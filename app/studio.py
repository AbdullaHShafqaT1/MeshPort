"""MeshPort's local interface and Colab bridge."""
import asyncio, base64, io, json, re, sys, uuid
from PIL import Image
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit, parse_qs
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .angle_preview import open_source, router as angle_preview
from .view_detection import ANGLES, EXTRA_ANGLES, ORIENTED_ANGLES, OLLAMA, DetectionError, detect_views, identify_angle
from .sheet_panels import split_gutters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'studio'
OUT.mkdir(parents=True, exist_ok=True)
app = FastAPI(title='MeshPort')
app.mount('/static', StaticFiles(directory=ROOT/'app/static'), name='static')
app.include_router(angle_preview)


class ColabConnection(NamedTuple):
    url: str
    token: str
    multi_view_conditioning: str | None = None


connection: ColabConnection | None = None
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
    if info.get('labeled_cameras') is not True:
        connection=None
        raise HTTPException(409,'This Colab worker ignores view labels. Run the updated notebook in a fresh runtime, then paste its new URL.')
    conditioning=info.get('multi_view_conditioning')
    if conditioning not in ('multi-image-multidiffusion','multi-image-stochastic'):
        conditioning=None
    connection=ColabConnection(*candidate,conditioning)
    return info

@app.get('/api/status')
def status():
    remote=connection
    result={'connected':remote is not None}
    if remote and remote.multi_view_conditioning:
        result['multi_view_conditioning']=remote.multi_view_conditioning
    return result

async def sheet_inputs(raw):
    with open_source(raw) as source:
        try:
            views = await asyncio.wait_for(detect_views(source), 180)
        except (httpx.RequestError, DetectionError, asyncio.TimeoutError):
            _, panels = await asyncio.to_thread(split_gutters, source)
            views = [('panel', box) for box in panels]
        if len(views) < 2:
            return None
        if len(views) > 6:
            raise ValueError('More than six views detected. Select up to six in the angle separator.')
        encoded = []
        total = 0
        for _, box in views:
            with source.crop(box) as crop, io.BytesIO() as data:
                crop.thumbnail((2048,2048), Image.Resampling.LANCZOS)
                crop.save(data, format='WEBP', lossless=True)
                total += data.tell()
                if total > 24*1024*1024:
                    raise ValueError('Sheet crops exceed the upload limit. Select fewer views in the angle separator.')
                encoded.append(base64.b64encode(data.getvalue()).decode('ascii'))
        return {'images': encoded, 'angles':[a if a!='panel' else 'unknown' for a,_ in views]}


async def work(job_id, body, remote: ColabConnection):
    job=jobs[job_id]; folder=OUT/job_id; folder.mkdir()
    headers={'Authorization':'Bearer '+remote.token}
    try:
        for index, encoded in enumerate(body['images']):
            raw=base64.b64decode(encoded,validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                suffix={'PNG':'png','JPEG':'jpg','WEBP':'webp'}[image.format]
            (folder/f'input-{index+1}.{suffix}').write_bytes(raw)
        if len(body['images']) == 1 and body.get('angles',['unknown'])[0]=='unknown':
            job['stage']='Checking image for multiple views'
            body = await sheet_inputs(raw) or body
            job['stage']='Sending reference images to Colab'
        labels = body.get('angles',['unknown']*len(body['images']))
        if any(label in ('unknown','oblique','front_three_quarter','rear_three_quarter') for label in labels):
            job['stage']='Identifying camera directions'
            async with httpx.AsyncClient(base_url=OLLAMA,timeout=httpx.Timeout(60,connect=3),trust_env=False) as detector:
                for index,label in enumerate(labels):
                    if label in ('unknown','oblique','front_three_quarter','rear_three_quarter'):
                        try:
                            with open_source(base64.b64decode(body['images'][index])) as source:
                                detected = await identify_angle(source,detector)
                            if label == 'oblique' and detected in ('front','rear'):
                                labels[index] = detected+'_three_quarter'
                            elif detected in ORIENTED_ANGLES and (label in ('unknown','oblique') or detected.startswith(label)):
                                labels[index]=detected
                            elif detected!='unknown' and (label=='unknown' or label=='oblique' and detected.endswith('three_quarter')):
                                labels[index]=detected
                        except (DetectionError,httpx.RequestError):
                            break
        body = dict(body,angles=labels)
        async with httpx.AsyncClient(timeout=60,follow_redirects=False) as client:
            deadline = asyncio.get_running_loop().time() + 3600
            while True:
                r = await client.post(remote.url+'/jobs',json=body,headers=headers)
                if r.status_code != 409:
                    r.raise_for_status()
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError('Colab stayed busy for one hour. Check the running job in the notebook.')
                job['stage'] = 'Waiting for the current Colab GPU job to finish'
                await asyncio.sleep(20)
            remote_id=r.json()['id']
            if not re.fullmatch(r'[a-f0-9]{32}',remote_id):raise ValueError('Invalid remote job identifier')
            job['remote_id'] = remote_id
            deadline = asyncio.get_running_loop().time() + 3600
            while asyncio.get_running_loop().time() < deadline:
                try:
                    r = await client.get(remote.url+'/jobs/'+remote_id,headers=headers)
                except httpx.TransportError:
                    r = None
                if r is None or r.status_code in {502,503,504}:
                    job['stage'] = 'Reconnecting to the accepted Colab job'
                    await asyncio.sleep(5)
                    continue
                r.raise_for_status()
                state=r.json()
                job.update(stage=state.get('stage','Generating'),elapsed=state.get('elapsed',0))
                if state['status']=='failed':raise RuntimeError(state.get('error','Generation failed'))
                if state['status']=='complete':break
                await asyncio.sleep(2)
            else:raise RuntimeError('Colab did not finish within one hour. Check the notebook.')
            details=state.get('details',{})
            if not isinstance(details,dict):
                raise ValueError('Colab returned invalid model details.')
            count=details.get('input_count')
            if count is not None and (type(count) is not int or count!=len(body['images'])):
                raise ValueError('Colab reported a different number of views than MeshPort submitted.')
            registrations = details.get('reference_registration')
            if (not isinstance(registrations,list) or
                [row.get('supplied_label') if isinstance(row,dict) else None for row in registrations] != labels):
                raise ValueError('Colab did not confirm the submitted view labels. Run the updated notebook and reconnect.')
            job['stage']='Saving model locally'
            size=0
            async with client.stream('GET',remote.url+'/jobs/'+remote_id+'/model',headers=headers) as response:
                response.raise_for_status()
                with (folder/'model.glb').open('wb') as f:
                    async for chunk in response.aiter_bytes():
                        size+=len(chunk)
                        if size>256*1024*1024:raise RuntimeError('Model exceeds the 256 MB download limit')
                        f.write(chunk)
            job['glb']=f'/api/files/{job_id}/model.glb'
            job['details']=details
            job['stage']='Exporting textured FBX'
            proc=await asyncio.create_subprocess_exec(sys.executable,str(ROOT/'app/convert_fbx.py'),str(folder),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
            try: log,_=await asyncio.wait_for(proc.communicate(),300)
            except asyncio.TimeoutError:
                proc.kill();await proc.wait();raise RuntimeError('FBX conversion timed out; GLB is available.')
            (folder/'conversion.log').write_bytes(log)
            if proc.returncode:raise RuntimeError('FBX conversion failed; GLB is available. See conversion.log in the output folder.')
            job.update(status='complete',stage='Ready to download',fbx=f'/api/files/{job_id}/model.fbx',bundle=f'/api/files/{job_id}/model_fbx_with_textures.zip')
    except Exception as exc:
        # Never persist or return transport credentials.
        message=(str(exc) or type(exc).__name__).replace(remote.token,'[redacted]')
        job.update(status='failed',stage='Could not finish',error=message)
    finally:
        (folder/'job.json').write_text(json.dumps(job,indent=2))

@app.post('/api/jobs')
async def generate(request: Request):
    remote=connection
    if remote is None:raise HTTPException(409,'Connect Colab first.')
    if any(j['status']=='running' for j in jobs.values()):raise HTTPException(409,'A model is already being generated.')
    raw=bytearray()
    async for chunk in request.stream():
        if len(raw)+len(chunk)>32*1024*1024+4096:raise HTTPException(413,'Upload limit is 24 MB total before encoding.')
        raw.extend(chunk)
    try:
        body=json.loads(raw)
        images=body['images']
        if not isinstance(images,list) or not 1<=len(images)<=6 or any(not isinstance(i,str) for i in images):raise ValueError()
        if set(body)-{'images','angles'}:raise ValueError()
        labels=body.get('angles',['unknown']*len(images))
        if not isinstance(labels,list) or len(labels)!=len(images) or any(label not in ANGLES+EXTRA_ANGLES+ORIENTED_ANGLES for label in labels):raise ValueError()
        total=0
        for encoded in images:
            data=base64.b64decode(encoded,validate=True);total+=len(data)
            if total>24*1024*1024:raise ValueError()
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {'PNG','JPEG','WEBP'} or image.width*image.height>16_000_000 or getattr(image,'n_frames',1)!=1:raise ValueError()
                image.verify()
    except Exception:raise HTTPException(400,'Choose one to six still PNG, JPEG or WebP images, at most 16 megapixels each and 24 MiB total.')
    if any(j['status']=='running' for j in jobs.values()):raise HTTPException(409,'A model is already being generated.')
    job_id=uuid.uuid4().hex
    jobs[job_id]={'id':job_id,'status':'running','stage':'Sending images to Colab'}
    task=asyncio.create_task(work(job_id,body,remote));active.add(task);task.add_done_callback(active.discard)
    return jobs[job_id]

@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    if job_id in jobs:
        return jobs[job_id]
    if re.fullmatch(r'[a-f0-9]{32}', job_id):
        path = OUT / job_id / 'job.json'
        if path.is_file():
            return json.loads(path.read_text(encoding='utf-8'))
    raise HTTPException(404, 'Job not found')

@app.get('/api/files/{job_id}/{filename}')
def file(job_id: str,filename: str):
    if not re.fullmatch(r'[a-f0-9]{32}',job_id) or filename not in {'model.glb','model.fbx','model_fbx_with_textures.zip'}:raise HTTPException(404)
    path=OUT/job_id/filename
    if not path.is_file():raise HTTPException(404)
    return FileResponse(path,filename=filename,headers={'Cache-Control':'no-store'})

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
