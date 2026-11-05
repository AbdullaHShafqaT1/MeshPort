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
        studio.jobs.clear()

    def test_boundaries(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertEqual(self.client.get('/api/notebook').status_code,200)
        self.assertEqual(self.client.get('/api/status',headers={'Host':'evil.example'}).status_code,403)
        self.assertEqual(self.client.post('/api/connect',json={'url':'x'},headers={'Origin':'https://evil.example'}).status_code,403)
        self.assertEqual(self.client.post('/api/jobs',json={'images':[]}).status_code,409)
        for url in ['https://example.com/#key='+'a'*32,'https://a.trycloudflare.com:bad/','https://a.trycloudflare.com/#key=short']:
            self.assertEqual(self.client.post('/api/connect',json={'url':url}).status_code,400)

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
            subprocess.run([sys.executable,'-c',source,str(fixture_path)],check=True,capture_output=True)
            recorded=fixture_path.read_bytes()
        image_bytes=io.BytesIO()
        Image.new('RGB',(8,8),'orange').save(image_bytes,format='PNG')
        def remote(request):
            self.assertEqual(request.headers['Authorization'],'Bearer '+'k'*32)
            if request.method=='POST':
                self.assertEqual(len(json.loads(request.content)['images']),1)
                return httpx.Response(200,json={'id':'a'*32})
            if request.url.path.endswith('/model'):
                return httpx.Response(200,content=recorded)
            return httpx.Response(200,json={'status':'complete','details':{'faces':10916}})
        original=httpx.AsyncClient
        with tempfile.TemporaryDirectory(dir=root/'outputs') as temp:
            job_id=uuid.uuid4().hex
            studio.jobs[job_id]={'id':job_id,'status':'running'}
            body={'images':[base64.b64encode(image_bytes.getvalue()).decode()]}
            with patch.object(studio,'OUT',Path(temp)),patch.object(studio.httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(remote),**kw)):
                asyncio.run(studio.work(job_id,body,('https://test.trycloudflare.com','k'*32)))
                result=studio.jobs[job_id]
                self.assertEqual(result['status'],'complete',
                    (result, (Path(temp)/job_id/'conversion.log').read_text(errors='replace')))
                self.assertEqual(self.client.get(result['fbx']).status_code,200)
                report=json.loads((Path(temp)/job_id/'fbx_verification.json').read_text())
                self.assertEqual(report['before']['triangles'],report['after_reimport']['triangles'])
                self.assertTrue(report['embedded_textures'])
                self.assertTrue((Path(temp)/job_id/'input-1.png').exists())
                self.assertNotIn('k'*32,(Path(temp)/job_id/'job.json').read_text())

if __name__=='__main__':unittest.main()
