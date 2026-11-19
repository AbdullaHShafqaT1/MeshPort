from pathlib import Path
import json,ast
root=Path(__file__).resolve().parent
nb={'cells':[{'cell_type':'markdown','metadata':{},'source':[]}]+[{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':[]} for _ in range(4)],'metadata':{'kernelspec':{'display_name':'Python 3','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU','colab':{'gpuType':'A100'}},'nbformat':4,'nbformat_minor':2}
nb['cells'][1]['source']=(root/'verify_cell.py').read_text().splitlines(True)
nb['cells'][2]['source']=(root/'setup_cell.py').read_text().splitlines(True)
source=(root/'service.py').read_text()
ast.parse(source)
launch='''# @title 2. Start the image-to-3D service
import os, pathlib, subprocess, secrets, time, urllib.request
'''+ 'service_source = '+repr(source)+'\nphoto_source = '+repr((root/'photo_finish.py').read_text())+'\n'+'''
if 'studio_proc' in globals() and studio_proc.poll() is None:
    raise RuntimeError('The service is already running. Run the final cell to copy its URL, or restart the runtime for changed code.')
STUDIO_KEY=secrets.token_urlsafe(32)
path=pathlib.Path('/content/TRELLIS/form_service.py')
path.write_text(service_source)
path.with_name('photo_finish.py').write_text(photo_source)
env=os.environ.copy()
env.update(STUDIO_KEY=STUDIO_KEY, PATH='/content/trellis-env/bin:'+env['PATH'], CUDA_HOME='/content/trellis-env', CC='/content/trellis-env/bin/x86_64-conda-linux-gnu-cc', CXX='/content/trellis-env/bin/x86_64-conda-linux-gnu-c++', ATTN_BACKEND='xformers', SPCONV_ALGO='native', TORCH_CUDA_ARCH_LIST='8.0', MAX_JOBS='4')
env['CPATH']=':'.join(['/content/trellis-env/targets/x86_64-linux/include']+[str(p) for p in pathlib.Path('/content/trellis-env/lib/python3.10/site-packages/nvidia').glob('*/include')])
env['LIBRARY_PATH']='/content/trellis-env/targets/x86_64-linux/lib'
log=open('/content/form-service.log','w')
studio_proc=subprocess.Popen(['/content/trellis-env/bin/python','-u',str(path)],cwd='/content/TRELLIS',env=env,stdout=log,stderr=subprocess.STDOUT)
for attempt in range(600):
    if studio_proc.poll() is not None:
        raise RuntimeError(pathlib.Path('/content/form-service.log').read_text()[-6000:])
    try:
        request=urllib.request.Request('http://127.0.0.1:7860/health',headers={'Authorization':'Bearer '+STUDIO_KEY})
        with urllib.request.urlopen(request,timeout=2) as response:
            if response.status==200:break
    except Exception:time.sleep(2)
else:raise TimeoutError('Loading is taking longer than expected. Inspect /content/form-service.log before restarting.')
print('TRELLIS ready. Run the final cell to get your connection URL.')
'''
tunnel='''# @title 3. Copy this connection URL into localhost
import pathlib, subprocess, urllib.request, re, time
if 'studio_proc' not in globals() or studio_proc.poll() is not None:
    raise RuntimeError('Run the service cell first.')
if 'tunnel_proc' not in globals() or tunnel_proc.poll() is not None:
    binary=pathlib.Path('/content/cloudflared')
    if not binary.exists():
        urllib.request.urlretrieve('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64',binary)
        binary.chmod(0o755)
    tunnel_log=open('/content/form-tunnel.log','w')
    tunnel_proc=subprocess.Popen([str(binary),'tunnel','--url','http://127.0.0.1:7860','--no-autoupdate'],stdout=tunnel_log,stderr=subprocess.STDOUT)
for attempt in range(90):
    if tunnel_proc.poll() is not None:raise RuntimeError('Tunnel stopped; inspect /content/form-tunnel.log')
    match=re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com',pathlib.Path('/content/form-tunnel.log').read_text())
    if match:break
    time.sleep(1)
else:raise TimeoutError('Tunnel URL not ready; rerun this cell.')
print('Paste this entire URL into MeshPort (keep it private):')
print(match.group(0)+'/#key='+STUDIO_KEY)
print('Keep this Colab runtime connected. Reconnecting a new runtime creates a new URL.')
'''
nb['cells'][0]['source']=['# MeshPort · Colab A100\n\nRun all cells, then paste the complete URL printed by the final cell into the localhost app. Upload one image or up to six views of the same object in localhost. Generation runs here; preview and textured FBX conversion run locally.\n\nSelect an A100 GPU before running. First setup and downloads can take several minutes. The tunnel requires a random session key included in the URL fragment. Treat the complete URL as private; notebook outputs may contain that key. Clear outputs before sharing the notebook. Keep Colab connected.\n\nUses TRELLIS-image-large with seed 42, 16 sampling steps and 2048 textures. For multiple inputs, TRELLIS blends each view at every geometry and appearance step. This requires more GPU time as the number of views grows. Multi-image conditioning is experimental.\n']
for index,code in [(3,launch),(4,tunnel)]:
 ast.parse(code);nb['cells'][index]['source']=code.splitlines(True);nb['cells'][index]['outputs']=[];nb['cells'][index]['execution_count']=None
(root/'meshport_colab.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
print('Notebook generated; embedded service syntax verified.')
