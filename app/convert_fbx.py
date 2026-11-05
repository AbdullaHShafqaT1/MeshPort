"""Convert this generated GLB to a textured binary FBX and verify a round trip."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

result_dir = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '.fbx-tools'))
import bpy

source = result_dir / 'model.glb'
destination = result_dir / 'model.fbx'
texture_dir = result_dir / 'fbx-textures'
texture_dir.mkdir(exist_ok=True)

def scene_facts():
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    triangles = 0
    uv_loops = 0
    material_images = []
    points = []
    for obj in meshes:
        obj.data.calc_loop_triangles()
        triangles += len(obj.data.loop_triangles)
        uv_loops += len(obj.data.uv_layers.active.data) if obj.data.uv_layers.active else 0
        points.extend([obj.matrix_world @ vertex.co for vertex in obj.data.vertices])
        for material in obj.data.materials:
            if material and material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.image and node.outputs['Color'].is_linked:
                        material_images.append(node.image)
    assert meshes and points, 'No mesh geometry found'
    bounds = [[min(point[i] for point in points) for i in range(3)],
              [max(point[i] for point in points) for i in range(3)]]
    return {'mesh_objects': len(meshes), 'triangles': triangles, 'uv_loops': uv_loops,
            'bounds': bounds, 'texture_sizes': [list(image.size) for image in material_images]}, material_images

bpy.ops.wm.read_factory_settings(use_empty=True)
assert 'FINISHED' in bpy.ops.import_scene.gltf(filepath=str(source))
before, images = scene_facts()
assert images and before['uv_loops'], 'The source must have UVs and a linked texture'
textures = []
for index, image in enumerate(dict.fromkeys(images)):
    texture = texture_dir / f'basecolor_{index}.png'
    if image.packed_file:
        texture.write_bytes(bytes(image.packed_file.data))
    else:
        image.filepath_raw = str(texture)
        image.file_format = 'PNG'
        image.save()
    image.filepath_raw = str(texture)
    textures.append(texture)

bpy.ops.object.select_all(action='SELECT')
assert 'FINISHED' in bpy.ops.export_scene.fbx(
    filepath=str(destination), use_selection=True, object_types={'MESH'},
    path_mode='COPY', embed_textures=True, bake_anim=False,
    add_leaf_bones=False, axis_forward='-Z', axis_up='Y',
    use_tspace=True, mesh_smooth_type='FACE',
)
assert destination.read_bytes().startswith(b'Kaydara FBX Binary  \x00\x1a\x00')

bpy.ops.wm.read_factory_settings(use_empty=True)
assert 'FINISHED' in bpy.ops.import_scene.fbx(filepath=str(destination), use_image_search=True)
after, imported_images = scene_facts()
assert after['triangles'] == before['triangles'], (before, after)
assert after['uv_loops'] == before['uv_loops'], (before, after)
assert imported_images and all(image.has_data for image in imported_images)
assert after['texture_sizes'] == before['texture_sizes'], (before, after)
for original_bound, imported_bound in zip(before['bounds'], after['bounds']):
    assert max(abs(a-b) for a,b in zip(original_bound, imported_bound)) < 1e-5

verification = {
    'source_glb_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'fbx_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
    'fbx_bytes': destination.stat().st_size,
    'converter': 'Blender ' + bpy.app.version_string,
    'binary_fbx': True, 'embedded_textures': True,
    'before': before, 'after_reimport': after,
}
report = result_dir / 'fbx_verification.json'
report.write_text(json.dumps(verification, indent=2))
bundle = result_dir / 'model_fbx_with_textures.zip'
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
    archive.write(destination, destination.name)
    archive.write(report, report.name)
    for texture in textures:
        archive.write(texture, str(texture.relative_to(result_dir)))
print(json.dumps(verification, indent=2))
print('FBX:', destination)
print('Texture bundle:', bundle)
