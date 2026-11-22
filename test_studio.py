"""Bridge contract checks with synthetic geometry; no live inference claims."""
import asyncio, base64, io, json, tempfile, unittest, uuid
import subprocess, sys
from PIL import Image
from pathlib import Path
from unittest.mock import patch
import httpx
from fastapi.testclient import TestClient
from app import studio

class StudioTests(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(studio.app)
        studio.connection=None
        studio.local_enabled=True
        studio.jobs.clear()

    def test_worker_without_camera_labels_cannot_generate(self):
        original = httpx.AsyncClient
        for capable, expected in [(True,200),(False,409)]:
            health={'protocol':'form-studio-v1','ready':True,'labeled_cameras':capable}
            with patch.object(studio.httpx,'AsyncClient',lambda **kw:original(
                transport=httpx.MockTransport(lambda _:httpx.Response(200,json=health)),**kw)):
                response=self.client.post('/api/connect',json={'url':'https://test.trycloudflare.com/#key='+'a'*32})
            self.assertEqual(response.status_code,expected)
            self.assertEqual(self.client.get('/api/status').json()['connected'],capable)

    def test_boundaries(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertEqual(self.client.get('/api/notebook').status_code,200)
        self.assertEqual(self.client.get('/api/status',headers={'Host':'evil.example'}).status_code,403)
        self.assertEqual(self.client.post('/api/connect',json={'url':'x'},headers={'Origin':'https://evil.example'}).status_code,403)
        studio.local_enabled = False
        self.assertEqual(self.client.post('/api/jobs',json={'images':[]}).status_code,409)
        studio.local_enabled = True
        for url in ['https://example.com/#key='+'a'*32,'https://a.trycloudflare.com:bad/','https://a.trycloudflare.com/#key=short']:
            self.assertEqual(self.client.post('/api/connect',json={'url':url}).status_code,400)

    def test_local_default_mode_and_colab_toggle(self):
        from unittest.mock import AsyncMock
        original = httpx.AsyncClient
        with patch.object(studio, 'is_local_available', return_value=True):
            st = self.client.get('/api/status').json()
            self.assertEqual(st['mode'], 'local')
            self.assertTrue(st['local_available'])
            self.assertEqual(st['local_info']['target_faces'], 40000)
            self.assertEqual(st['local_info']['texture_size'], 4096)

            buffer = io.BytesIO()
            Image.new('RGB', (8, 8), 'red').save(buffer, format='PNG')
            encoded = [base64.b64encode(buffer.getvalue()).decode()]
            with patch.object(studio, 'work_local', new_callable=AsyncMock) as work_local:
                resp = self.client.post('/api/jobs', json={'images': encoded})
                self.assertEqual(resp.status_code, 200)
                work_local.assert_called_once()
            studio.jobs.clear()

            health = {'protocol': 'form-studio-v1', 'ready': True, 'labeled_cameras': True}
            with patch.object(studio.httpx, 'AsyncClient', lambda **kw: original(
                transport=httpx.MockTransport(lambda _: httpx.Response(200, json=health)), **kw)):
                self.assertEqual(self.client.post('/api/connect', json={'url': 'https://test.trycloudflare.com/#key=' + 'a' * 32}).status_code, 200)
            self.assertEqual(self.client.get('/api/status').json()['mode'], 'colab')
            disc = self.client.post('/api/disconnect')
            self.assertEqual(disc.status_code, 200)
            self.assertEqual(disc.json()['mode'], 'local')

    def test_multiview_submission_and_invalid_inputs(self):
        from unittest.mock import AsyncMock
        studio.connection=studio.ColabConnection('https://test.trycloudflare.com','k'*32)
        images=[]
        for color in ['red','green','blue']:
            buffer=io.BytesIO()
            Image.new('RGB',(8,8),color).save(buffer,format='PNG')
            images.append(base64.b64encode(buffer.getvalue()).decode())
        with patch.object(studio,'work',new_callable=AsyncMock) as work:
            response=self.client.post('/api/jobs',json={'images':images})
            self.assertEqual(response.status_code,200)
            self.assertEqual(work.call_args.args[1],{'images':images})
        studio.jobs.clear()
        for body in [{'images':[]},{'images':images*3},{'images':['bad']},
                     {'images':[base64.b64encode(b'not an image').decode()]},
                     {'images':images,'ignored':True}]:
            with self.subTest(body=body):
                self.assertEqual(self.client.post('/api/jobs',json=body).status_code,400)
                self.assertFalse(studio.jobs)

    def test_cloud_generation_uses_all_views(self):
        # Exercise the real Colab upload and generation functions with a stub pipeline.
        import ast, hashlib, threading, time
        from fastapi import FastAPI, HTTPException, Request
        from PIL import ImageOps
        from types import SimpleNamespace
        from unittest.mock import Mock

        source=ast.parse((studio.ROOT/'colab/service.py').read_text())
        handlers=[
            node for node in source.body
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in {'create','generate'}
        ]
        with tempfile.TemporaryDirectory() as temp:
            pipeline=Mock()
            pipeline.run.return_value=pipeline.run_multi_image.return_value={'gaussian':['g'],'mesh':[Mock(faces=[1,2])]}
            sparse_model, slat_model = object(), object()
            pipeline.sparse_structure_sampler = SimpleNamespace(_inference_model=sparse_model)
            pipeline.slat_sampler = SimpleNamespace(_inference_model=slat_model)
            mesh=Mock(faces=[1],vertices=[1,2],is_watertight=True)
            lock=threading.Lock()
            cloud=FastAPI()
            def inline_thread(*,target,args,daemon):
                return SimpleNamespace(start=lambda:target(*args))
            namespace=dict(ROOT=Path(temp),jobs={},pipeline=pipeline,lock=lock,
                           SIMPLIFY_RATIO=.5,refine=Mock(return_value={}),unlit=Mock(),
                           torch=Mock(),postprocessing_utils=SimpleNamespace(to_glb=Mock(return_value=mesh)),
                           time=time,hashlib=hashlib,json=json,TOKEN='secret',MULTI_VIEW_MODE='multidiffusion',
                           app=cloud,Request=Request,HTTPException=HTTPException,uuid=uuid,
                           io=io,base64=base64,Image=Image,ImageOps=ImageOps,
                           threading=SimpleNamespace(Thread=inline_thread))
            exec(compile(ast.Module(body=handlers,type_ignores=[]),'service.py','exec'),namespace)
            client=TestClient(cloud)

            encoded=[]
            expected=[]
            for index in range(6):
                original=Image.new('RGB',(20,8),(index*30,40,50))
                with io.BytesIO() as buffer:
                    if index==0:
                        exif=Image.Exif()
                        exif[274]=6
                        original.save(buffer,format='JPEG',exif=exif)
                    else:
                        original.save(buffer,format='PNG')
                    raw=buffer.getvalue()
                encoded.append(base64.b64encode(raw).decode('ascii'))
                with Image.open(io.BytesIO(raw)) as decoded:
                    oriented=ImageOps.exif_transpose(decoded).convert('RGBA')
                    expected.append((oriented.size,oriented.tobytes()))
            self.assertEqual(expected[0][0],(8,20))

            def fail_after_patching(*args, **kwargs):
                for sampler in (pipeline.sparse_structure_sampler,pipeline.slat_sampler):
                    sampler._old_inference_model = sampler._inference_model
                    sampler._inference_model = object()
                raise RuntimeError('Sampling failed')

            pipeline.run_multi_image.side_effect = fail_after_patching
            response = client.post('/jobs',json={'images':encoded[:3]})
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(namespace['jobs'][response.json()['id']]['status'],'failed')
            self.assertIs(pipeline.sparse_structure_sampler._inference_model,sparse_model)
            self.assertIs(pipeline.slat_sampler._inference_model,slat_model)
            self.assertFalse(hasattr(pipeline.sparse_structure_sampler,'_old_inference_model'))
            self.assertFalse(hasattr(pipeline.slat_sampler,'_old_inference_model'))
            self.assertFalse(lock.locked())
            pipeline.run_multi_image.side_effect = None

            for count in [1,3,6]:
                pipeline.reset_mock()
                response=client.post('/jobs',json={'images':encoded[:count]})
                self.assertEqual(response.status_code,200,response.text)
                job_id=response.json()['id']
                job=namespace['jobs'][job_id]
                self.assertEqual(job['status'],'complete',job)
                self.assertEqual(job['details']['input_count'],count)
                retained=[(Path(temp)/job_id/f'input-{i+1}.png').read_bytes() for i in range(count)]
                self.assertEqual(job['details']['input_sha256'],[hashlib.sha256(data).hexdigest() for data in retained])
                if count==1:
                    pipeline.run.assert_called_once()
                    self.assertEqual(job['details']['conditioning'],'single-image')
                    images=[pipeline.run.call_args.args[0]]
                    pipeline.run_multi_image.assert_not_called()
                else:
                    pipeline.run.assert_not_called()
                    pipeline.run_multi_image.assert_called_once()
                    self.assertEqual(job['details']['conditioning'],'multi-image-multidiffusion')
                    images=pipeline.run_multi_image.call_args.args[0]
                    self.assertEqual(pipeline.run_multi_image.call_args.kwargs['mode'],'multidiffusion')
                self.assertEqual([(image.size,image.tobytes()) for image in images],expected[:count])
                self.assertFalse(lock.locked())

    def test_mismatched_colab_view_count_stops_before_export(self):
        with io.BytesIO() as buffer:
            Image.new('RGB',(8,8),'white').save(buffer,format='PNG')
            encoded=base64.b64encode(buffer.getvalue()).decode('ascii')
        requested=[]

        def remote(request):
            requested.append(request.url.path)
            if len(requested)==1:
                return httpx.Response(409,json={'detail':'GPU busy'})
            if request.method=='POST':
                return httpx.Response(200,json={'id':'a'*32})
            if len(requested)==3:
                raise httpx.ReadTimeout('poll interrupted')
            return httpx.Response(200,json={'status':'complete','details':{'input_count':1}})

        original=httpx.AsyncClient
        job_id='b'*32
        studio.jobs[job_id]={'id':job_id,'status':'running'}
        connection=studio.ColabConnection('https://test.trycloudflare.com','k'*32)
        with tempfile.TemporaryDirectory() as temp, patch.object(studio,'OUT',Path(temp)), patch.object(
            studio.httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(remote),**kw)
        ), patch.object(studio.asyncio,'sleep'):
            asyncio.run(studio.work(job_id,{'images':[encoded,encoded],'angles':['front','rear']},connection))
            self.assertEqual(requested,['/jobs','/jobs','/jobs/'+'a'*32,'/jobs/'+'a'*32])
            job=studio.jobs[job_id]
            self.assertEqual(job['status'],'failed')
            self.assertIn('different number of views',job['error'])
            self.assertNotIn('glb',job)

    def test_generated_fixture_to_fbx(self):
        root=studio.ROOT
        with tempfile.TemporaryDirectory() as fixture:
            fixture_path=Path(fixture)/'fixture.glb'
            source = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()/'.fbx-tools'))
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add()
obj=bpy.context.object
material=bpy.data.materials.new('Fixture')
material.use_nodes=True
image=bpy.data.images.new('Fixture',width=8,height=8)
image.generated_color=(1.0,0.3,0.0,1.0)
image.filepath_raw=str(Path(sys.argv[1]).with_suffix('.png'))
image.file_format='PNG'
image.save()
image.pack()
node=material.node_tree.nodes.new('ShaderNodeTexImage')
node.image=image
material.node_tree.links.new(node.outputs['Color'],material.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
obj.data.materials.append(material)
bpy.ops.export_scene.gltf(filepath=sys.argv[1],export_format='GLB')
"""
            subprocess.run([studio.fbx_python(),'-c',source,str(fixture_path)],check=True,capture_output=True)
            recorded=fixture_path.read_bytes()
        views=[]
        for color in ('red','blue','green','orange','yellow','purple'):
            image_bytes=io.BytesIO()
            Image.new('RGB',(8,8),color).save(image_bytes,format='PNG')
            views.append(image_bytes.getvalue())
        encoded=[base64.b64encode(view).decode() for view in views]
        submissions=[]
        def remote(request):
            self.assertEqual(request.headers['Authorization'],'Bearer '+'k'*32)
            if request.method=='POST':
                submissions.append(json.loads(request.content)['images'])
                return httpx.Response(200,json={'id':'a'*32})
            if request.url.path.endswith('/model'):
                return httpx.Response(200,content=recorded)
            return httpx.Response(200,json={'status':'complete','details':{'faces':10916,'input_count':len(views),
                'reference_registration':[{'supplied_label':label} for label in body['angles']]}})
        original=httpx.AsyncClient
        with tempfile.TemporaryDirectory(dir=root/'outputs') as temp:
            job_id=uuid.uuid4().hex
            studio.jobs[job_id]={'id':job_id,'status':'running'}
            body={'images':encoded,'angles':['front','rear','left','right','top','bottom']}
            with patch.object(studio,'OUT',Path(temp)),patch.object(studio.httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(remote),**kw)):
                remote_connection=studio.ColabConnection('https://test.trycloudflare.com','k'*32)
                asyncio.run(studio.work(job_id,body,remote_connection))
                result=studio.jobs[job_id]
                self.assertEqual(result['status'],'complete',
                    (result, (Path(temp)/job_id/'conversion.log').read_text(errors='replace')))
                self.assertEqual(self.client.get(result['fbx']).status_code,200)
                report=json.loads((Path(temp)/job_id/'fbx_verification.json').read_text())
                self.assertEqual(report['before']['triangles'],report['after_reimport']['triangles'])
                self.assertTrue(report['embedded_textures'])
                self.assertEqual(submissions,[encoded])
                self.assertEqual(result['details']['input_count'],len(views))
                retained=sorted((Path(temp)/job_id).glob('input-*.png'))
                self.assertEqual([path.read_bytes() for path in retained],views)
                self.assertNotIn('k'*32,(Path(temp)/job_id/'job.json').read_text())

if __name__=='__main__':unittest.main()
