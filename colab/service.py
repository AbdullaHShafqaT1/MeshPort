"""Authenticated Colab service, embedded into the notebook by build_notebook.py."""
import os, io, base64, json, uuid, time, threading, hashlib
from pathlib import Path
os.environ['ATTN_BACKEND']='xformers'
os.environ['SPCONV_ALGO']='native'
import torch
from PIL import Image
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

TOKEN=os.environ['STUDIO_KEY']
ROOT=Path('/content/trellis-results'); ROOT.mkdir(exist_ok=True)
pipeline=TrellisImageTo3DPipeline.from_pretrained('microsoft/TRELLIS-image-large')
pipeline.cuda()
app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
jobs={}; lock=threading.Lock()
Image.MAX_IMAGE_PIXELS=16_000_000

@app.middleware('http')
async def auth(request,call_next):
    import secrets
    if not secrets.compare_digest(request.headers.get('authorization',''),'Bearer '+TOKEN):
        return JSONResponse({'detail':'Invalid connection key'},401)
    return await call_next(request)

@app.get('/health')
def health():
    return {'protocol':'form-studio-v1','ready':True,'model':'TRELLIS image large','gpu':torch.cuda.get_device_name(0),'steps':16,'texture_size':2048}

def generate(job_id,images):
    job=jobs[job_id]; start=time.monotonic(); folder=ROOT/job_id
    try:
        folder.mkdir()
        digests=[]
        for i,image in enumerate(images):
            path=folder/f'input-{i+1}.png';image.save(path);digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        job['stage']='Generating geometry and appearance'
        params=dict(seed=42,formats=['gaussian','mesh'],sparse_structure_sampler_params={'steps':16,'cfg_strength':7.5},slat_sampler_params={'steps':16,'cfg_strength':3.0})
        outputs=pipeline.run(images[0],**params) if len(images)==1 else pipeline.run_multi_image(images,**params)
        job['stage']='Baking 2048 texture'
        mesh=postprocessing_utils.to_glb(outputs['gaussian'][0],outputs['mesh'][0],simplify=0.95,texture_size=2048)
        mesh.export(folder/'model.glb')
        details={'model':'microsoft/TRELLIS-image-large','seed':42,'steps':16,'texture_size':2048,'input_sha256':digests,'faces':len(mesh.faces),'vertices':len(mesh.vertices),'watertight':bool(mesh.is_watertight),'seconds':round(time.monotonic()-start,1)}
        (folder/'manifest.json').write_text(json.dumps(details,indent=2))
        job.update(status='complete',stage='Model ready',details=details)
    except Exception as exc:
        job.update(status='failed',stage='Generation failed',error=str(exc).replace(TOKEN,'[redacted]'))
    finally:
        job['elapsed']=round(time.monotonic()-start,1)
        torch.cuda.empty_cache();lock.release()

@app.post('/jobs')
async def create(request:Request):
    if not lock.acquire(blocking=False):raise HTTPException(409,'GPU is busy. Try again after the current model finishes.')
    try:
        raw=bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw)>32*1024*1024:raise ValueError('Images exceed 24 MB total upload limit')
        body=json.loads(raw); encoded=body['images']
        if not isinstance(encoded,list) or not 1<=len(encoded)<=6:raise ValueError('Choose one to six views of one object')
        images=[]
        for item in encoded:
            image=Image.open(io.BytesIO(base64.b64decode(item,validate=True)))
            if image.width*image.height>16_000_000:raise ValueError('Each image must be at most 16 megapixels')
            if image.format not in {'PNG','JPEG','WEBP'}:raise ValueError('Use PNG, JPEG or WebP')
            image.load();images.append(image.convert('RGBA'))
        job_id=uuid.uuid4().hex
        jobs[job_id]={'id':job_id,'status':'running','stage':'Preparing images','elapsed':0}
        threading.Thread(target=generate,args=(job_id,images),daemon=True).start()
        return {'id':job_id}
    except Exception as exc:
        lock.release();raise HTTPException(400,str(exc))

@app.get('/jobs/{job_id}')
def status(job_id:str):
    if job_id not in jobs:raise HTTPException(404)
    return jobs[job_id]

@app.get('/jobs/{job_id}/model')
def model(job_id:str):
    if job_id not in jobs or jobs[job_id]['status']!='complete':raise HTTPException(404)
    return FileResponse(ROOT/job_id/'model.glb',media_type='model/gltf-binary')

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=7860,access_log=False)
