"""Authenticated Colab service, embedded into the notebook by build_notebook.py."""
import os, io, base64, json, uuid, time, threading, hashlib
from pathlib import Path
os.environ['ATTN_BACKEND']='xformers'
os.environ['SPCONV_ALGO']='native'
import torch
from PIL import Image, ImageOps
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
from photo_finish import refine, unlit

TOKEN=os.environ['STUDIO_KEY']
ROOT=Path('/content/trellis-results'); ROOT.mkdir(exist_ok=True)
pipeline=TrellisImageTo3DPipeline.from_pretrained('microsoft/TRELLIS-image-large')
pipeline.cuda()
app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
jobs={}; lock=threading.Lock()
Image.MAX_IMAGE_PIXELS=16_000_000
MULTI_VIEW_MODE='multidiffusion'
SIMPLIFY_RATIO = float(os.environ.get('MESHPORT_SIMPLIFY_RATIO', '0.50'))
if not 0 <= SIMPLIFY_RATIO < 1:
    raise ValueError('MESHPORT_SIMPLIFY_RATIO must be in [0, 1).')

@app.middleware('http')
async def auth(request,call_next):
    import secrets
    if not secrets.compare_digest(request.headers.get('authorization',''),'Bearer '+TOKEN):
        return JSONResponse({'detail':'Invalid connection key'},401)
    return await call_next(request)

@app.get('/health')
def health():
    return {'protocol':'form-studio-v1','ready':True,'labeled_cameras':True,'model':'TRELLIS image large','gpu':torch.cuda.get_device_name(0),'steps':16,'texture_size':2048,'multi_view_conditioning':f'multi-image-{MULTI_VIEW_MODE}'}

def generate(job_id,images,labels=None):
    job=jobs[job_id]; start=time.monotonic(); folder=ROOT/job_id
    try:
        folder.mkdir()
        digests=[]
        for i,image in enumerate(images):
            path=folder/f'input-{i+1}.png';image.save(path);digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        params=dict(num_samples=1,seed=42,formats=['gaussian','mesh'],sparse_structure_sampler_params={'steps':16,'cfg_strength':7.5},slat_sampler_params={'steps':16,'cfg_strength':3.0})
        if len(images)==1:
            job['stage']='Generating geometry and appearance from one reference image'
            outputs=pipeline.run(images[0],**params)
            conditioning='single-image'
        else:
            job['stage']=f'Blending {len(images)} views for geometry and appearance'
            samplers=(pipeline.sparse_structure_sampler,pipeline.slat_sampler)
            original_methods=[sampler._inference_model for sampler in samplers]
            try:
                outputs=pipeline.run_multi_image(images,mode=MULTI_VIEW_MODE,**params)
            except Exception:
                # TRELLIS can leave its injected sampler installed when sampling raises.
                for sampler,original in zip(samplers,original_methods):
                    if '_old_inference_model' in vars(sampler):
                        sampler._inference_model=original
                        del sampler._old_inference_model
                raise
            conditioning=f'multi-image-{MULTI_VIEW_MODE}'
        job['stage']='Baking 2048 texture'
        source_mesh = outputs['mesh'][0]
        source_faces = len(source_mesh.faces)
        mesh=postprocessing_utils.to_glb(outputs['gaussian'][0],source_mesh,simplify=SIMPLIFY_RATIO,texture_size=2048)
        job['stage']='Matching reference cameras and applying visible photo detail'
        finish = refine(mesh, images, labels)
        mesh.export(folder/'model.glb',tree_postprocessor=unlit)
        details={'model':'microsoft/TRELLIS-image-large','seed':42,'steps':16,'texture_size':2048,'input_count':len(images),'conditioning':conditioning,'input_sha256':digests,'faces':len(mesh.faces),'vertices':len(mesh.vertices),'watertight':bool(mesh.is_watertight),'seconds':round(time.monotonic()-start,1)}
        details.update(source_faces=source_faces, simplify_ratio=SIMPLIFY_RATIO, **finish)
        (folder/'manifest.json').write_text(json.dumps(details,indent=2))
        job.update(status='complete',stage='Model ready',details=details)
    except Exception as exc:
        job.update(status='failed',stage='Generation failed',error=str(exc).replace(TOKEN,'[redacted]'))
    finally:
        job['elapsed']=round(time.monotonic()-start,1)
        try:
            torch.cuda.empty_cache()
        finally:
            lock.release()

@app.post('/jobs')
async def create(request:Request):
    if not lock.acquire(blocking=False):raise HTTPException(409,'GPU is busy. Try again after the current model finishes.')
    try:
        raw=bytearray()
        async for chunk in request.stream():
            if len(raw)+len(chunk)>32*1024*1024+4096:raise ValueError('Images exceed 24 MB total upload limit')
            raw.extend(chunk)
        body=json.loads(raw); encoded=body['images']
        if not isinstance(encoded,list) or not 1<=len(encoded)<=6:raise ValueError('Choose one to six views of one object')
        labels=body.get('angles',['unknown']*len(encoded))
        if not isinstance(labels,list) or len(labels)!=len(encoded) or any(label not in ('front','rear','left','right','top','bottom','front_three_quarter','rear_three_quarter','front_three_quarter_nose_left','front_three_quarter_nose_right','rear_three_quarter_nose_left','rear_three_quarter_nose_right','oblique','unknown') for label in labels):
            raise ValueError('Expected one angle per image')
        images=[];total=0
        for index,item in enumerate(encoded,1):
            data=base64.b64decode(item,validate=True);total+=len(data)
            if total>24*1024*1024:raise ValueError('Images exceed 24 MB total upload limit')
            with Image.open(io.BytesIO(data),formats=['PNG','JPEG','WEBP']) as image:
                if image.width*image.height>16_000_000:raise ValueError(f'View {index} exceeds 16 megapixels')
                if getattr(image,'n_frames',1)!=1:raise ValueError(f'View {index} must be a still image')
                images.append(ImageOps.exif_transpose(image).convert('RGBA'))
        job_id=uuid.uuid4().hex
        jobs[job_id]={'id':job_id,'status':'running','stage':'Preparing images','elapsed':0}
        threading.Thread(target=generate,args=(job_id,images,labels),daemon=True).start()
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
