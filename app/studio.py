"""Local Colab client. No local inference or weight downloads."""
import asyncio, base64, io, json, re, sys, uuid
from PIL import Image
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'studio'
OUT.mkdir(parents=True, exist_ok=True)
app = FastAPI(title='MeshPort')
app.mount('/static', StaticFiles(directory=ROOT/'app/static'), name='static')
connection = None
jobs = {}
active = set()

@app.middleware('http')
async def local_only(request, call_next):
    if request.headers.get('host','').split(':')[0] not in {'127.0.0.1','localhost','testserver'}:
        return JSONResponse({'detail':'Local access only'},403)
    origin=request.headers.get('origin')
    if origin and origin != str(request.base_url).rstrip('/'):
        return JSONResponse({'detail':'Cross-origin requests are not allowed'},403)
    return await call_next(request)

@app.get('/')
def home(): return FileResponse(ROOT/'app/static/studio.html')

class Connect(BaseModel):
    url: str

@app.post('/api/connect')
async def connect(body: Connect):
    global connection
    try:
        url=urlsplit(body.url.strip())
        port=url.port
    except ValueError:
        raise HTTPException(400,'Invalid connection URL')
    token=parse_qs(url.fragment).get('key',[''])[0]
    if url.scheme!='https' or not re.fullmatch(r'[a-z0-9-]+\.trycloudflare\.com',url.hostname or '') or port not in {None,443} or url.username or url.password or url.path not in {'','/'} or url.query or len(token)<32:
        raise HTTPException(400,'Paste the complete connection URL printed by the last notebook cell, including #key=…')
    candidate=(f'https://{url.hostname}', token)
    try:
        async with httpx.AsyncClient(timeout=20,follow_redirects=False) as client:
            response=await client.get(candidate[0]+'/health',headers={'Authorization':'Bearer '+token})
            response.raise_for_status(); info=response.json()
        if info.get('protocol')!='form-studio-v1' or not info.get('ready'):raise ValueError()
    except Exception:
        raise HTTPException(502,'Could not connect. Keep Colab running and copy its latest complete URL.')
    connection=candidate
    return info

@app.get('/api/status')
def status(): return {'connected':connection is not None}

async def work(job_id, body, remote):
    job=jobs[job_id]; folder=OUT/job_id; folder.mkdir()
    headers={'Authorization':'Bearer '+remote[1]}
    try:
        for index, encoded in enumerate(body['images']):
            raw=base64.b64decode(encoded,validate=True)
            image=Image.open(io.BytesIO(raw))
            suffix={'PNG':'png','JPEG':'jpg','WEBP':'webp'}[image.format]
            (folder/f'input-{index+1}.{suffix}').write_bytes(raw)
        async with httpx.AsyncClient(timeout=60,follow_redirects=False) as client:
            r=await client.post(remote[0]+'/jobs',json=body,headers=headers);r.raise_for_status()
            remote_id=r.json()['id']
            if not re.fullmatch(r'[a-f0-9]{32}',remote_id):raise ValueError('Invalid remote job identifier')
            # Polling timeout bounds a lost Colab session; it is not a generation quality setting.
            for _ in range(1800):
                r=await client.get(remote[0]+'/jobs/'+remote_id,headers=headers);r.raise_for_status(); state=r.json()
                job.update(stage=state.get('stage','Generating'),elapsed=state.get('elapsed',0))
                if state['status']=='failed':raise RuntimeError(state.get('error','Generation failed'))
                if state['status']=='complete':break
                await asyncio.sleep(2)
            else:raise RuntimeError('Colab did not finish within one hour. Check the notebook.')
            job['stage']='Saving model locally'
            size=0
            async with client.stream('GET',remote[0]+'/jobs/'+remote_id+'/model',headers=headers) as response:
                response.raise_for_status()
                with (folder/'model.glb').open('wb') as f:
                    async for chunk in response.aiter_bytes():
                        size+=len(chunk)
                        if size>256*1024*1024:raise RuntimeError('Model exceeds the 256 MB download limit')
                        f.write(chunk)
            job['glb']=f'/api/files/{job_id}/model.glb'
            job['stage']='Exporting textured FBX'
            proc=await asyncio.create_subprocess_exec(sys.executable,str(ROOT/'app/convert_fbx.py'),str(folder),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
            try: log,_=await asyncio.wait_for(proc.communicate(),300)
            except asyncio.TimeoutError:
                proc.kill();await proc.wait();raise RuntimeError('FBX conversion timed out; GLB is available.')
            (folder/'conversion.log').write_bytes(log)
            if proc.returncode:raise RuntimeError('FBX conversion failed; GLB is available. See conversion.log in the output folder.')
            job.update(status='complete',stage='Ready to download',fbx=f'/api/files/{job_id}/model.fbx',bundle=f'/api/files/{job_id}/model_fbx_with_textures.zip',details=state.get('details',{}))
    except Exception as exc:
        # Never persist or return transport credentials.
        message=str(exc).replace(remote[1],'[redacted]')
        job.update(status='failed',stage='Could not finish',error=message)
    finally:
        (folder/'job.json').write_text(json.dumps(job,indent=2))

@app.post('/api/jobs')
async def generate(request: Request):
    if connection is None:raise HTTPException(409,'Connect Colab first.')
    if any(j['status']=='running' for j in jobs.values()):raise HTTPException(409,'A model is already being generated.')
    raw=bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw)>32*1024*1024:raise HTTPException(413,'Upload limit is 24 MB total before encoding.')
    try:
        body=json.loads(raw)
        images=body['images']
        if not isinstance(images,list) or not 1<=len(images)<=6 or any(not isinstance(i,str) for i in images):raise ValueError()
        if set(body)!={'images'}:raise ValueError()
    except Exception:raise HTTPException(400,'Choose one to six images.')
    if any(j['status']=='running' for j in jobs.values()):raise HTTPException(409,'A model is already being generated.')
    job_id=uuid.uuid4().hex
    jobs[job_id]={'id':job_id,'status':'running','stage':'Sending images to Colab'}
    task=asyncio.create_task(work(job_id,body,connection));active.add(task);task.add_done_callback(active.discard)
    return jobs[job_id]

@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    if job_id not in jobs:raise HTTPException(404,'Job not found')
    return jobs[job_id]

@app.get('/api/files/{job_id}/{filename}')
def file(job_id: str,filename: str):
    if not re.fullmatch(r'[a-f0-9]{32}',job_id) or filename not in {'model.glb','model.fbx','model_fbx_with_textures.zip'}:raise HTTPException(404)
    path=OUT/job_id/filename
    if not path.is_file():raise HTTPException(404)
    return FileResponse(path,filename=filename)

@app.get('/api/notebook')
def notebook():
    return FileResponse(ROOT/'colab/meshport_colab.ipynb',filename='meshport_colab.ipynb')


SAMPLE = OUT / '9c17ad1483a000000000000000000000'

@app.get('/api/sample')
def get_sample():
    path = SAMPLE / 'job.json'
    if not path.is_file():
        raise HTTPException(404, 'Saved sample unavailable')
    return json.loads(path.read_text(encoding='utf-8'))

@app.get('/api/sample/input')
def get_sample_input():
    path = SAMPLE / 'input.png'
    if not path.is_file():
        raise HTTPException(404, 'Saved sample input unavailable')
    return FileResponse(path, media_type='image/png')
