"""Register foreground photos to a generated textured mesh and refine its UV atlas."""
import math

import cv2
import numpy as np
from numba import njit
from PIL import Image


@njit
def raster(screen, faces, height, width):
    """Nearest triangle and barycentric coordinates at each covered pixel."""
    owner = np.full((height, width), -1, np.int32)
    bary = np.zeros((height, width, 2), np.float32)
    depth = np.full((height, width), -np.inf, np.float32)
    for fi, (i, j, k) in enumerate(faces):
        ax, ay, az = screen[i]
        bx, by, bz = screen[j] - screen[i]
        cx, cy, cz = screen[k] - screen[i]
        det = bx*cy - by*cx
        if abs(det) < 1e-9:
            continue
        for y in range(max(0, math.ceil(ay+min(0., by, cy))), min(height, math.floor(ay+max(0., by, cy))+1)):
            for x in range(max(0, math.ceil(ax+min(0., bx, cx))), min(width, math.floor(ax+max(0., bx, cx))+1)):
                u = ((x-ax)*cy-(y-ay)*cx)/det
                v = (bx*(y-ay)-by*(x-ax))/det
                z = az+u*bz+v*cz
                if u >= 0 and v >= 0 and u+v <= 1 and z > depth[y,x]:
                    owner[y,x] = fi
                    bary[y,x,0], bary[y,x,1] = u,v
                    depth[y,x] = z
    return owner, bary, depth


def camera(vertices, yaw, elevation):
    a,b = np.deg2rad([yaw,elevation])
    right = np.array([np.cos(a),0,-np.sin(a)])
    direction = np.array([np.sin(a)*np.cos(b),np.sin(b),np.cos(a)*np.cos(b)])
    basis = np.stack([right,np.cross(right,direction),direction]).astype('f')
    projected = vertices @ basis.T
    extent = np.maximum(np.ptp(projected[:,:2],axis=0),1e-6)
    projected[:,:2] = (projected[:,:2]-projected[:,:2].min(0))/extent
    return projected, direction, extent[0]/extent[1]


def interpolate(values, triangles, weights):
    a,b,c = values[triangles].transpose(1,0,2)
    return a+(b-a)*weights[:,0:1]+(c-a)*weights[:,1:2]


def foreground(photo, session):
    photo = photo.copy()
    photo.thumbnail((1280,1280),Image.Resampling.LANCZOS)
    if photo.getchannel('A').getextrema()[0] == 255:
        from rembg import remove
        photo = remove(photo, session=session)
    bounds = photo.getchannel('A').point(lambda a: 255 if a>128 else 0).getbbox()
    if bounds is None or min(bounds[2]-bounds[0],bounds[3]-bounds[1]) < 8:
        return None, (False,)*4
    crop = np.asarray(photo.crop(bounds))
    mask = crop[:,:,3] > 128
    sides = (mask[:,0],mask[:,-1],mask[0],mask[-1])
    at_edge = (bounds[0]==0,bounds[2]==photo.width,bounds[1]==0,bounds[3]==photo.height)
    return crop, tuple(edge and side.mean()>.08 for edge,side in zip(at_edge,sides))


def visible_windows(edges, photo_aspect, model_aspect):
    """Candidate regions of the projected mesh visible inside a clipped photo."""
    left,right,top,bottom = edges
    horizontal,vertical = left or right,top or bottom
    if horizontal == vertical:
        yield (0.,1.,0.,1.)
        return
    span = min(1., photo_aspect/model_aspect if horizontal else model_aspect/photo_aspect)
    low,high = (left,right) if horizontal else (top,bottom)
    starts = np.linspace(0,1-span,13) if low and high else [1-span if low else 0.]
    for start in starts:
        yield (float(start),span,0.,1.) if horizontal else (0.,1.,float(start),span)


def joint_cameras(scores, labels):
    offsets = {'front': 0, 'right': -90, 'rear': 180, 'left': 90}

    def allowed(label, front, candidate):
        yaw,elev = candidate[1:3]
        label,_,nose = label.partition('_nose_')
        signed = (yaw-front-offsets.get(label,0)+180)%360-180
        delta = abs(signed)
        if nose=='left' and signed<=0 or nose=='right' and signed>=0:
            return False
        if label in offsets:
            return delta <= 15 and abs(elev) <= 12
        if label in ('top','bottom'):
            return elev >= 70 if label == 'top' else elev <= -70
        if label.endswith('three_quarter'):
            return (20 <= delta <= 70 if label.startswith('front') else 110 <= delta <= 160) and abs(elev) <= 50
        return True

    # The generated object's forward axis is unknown. Score complete assignments
    # around one shared heading, including the side profiles that reveal its nose.
    best = None
    for front in range(0,360,30):
        picks = [max((c for c in row if allowed(label,front,c)), default=None)
                 for row,label in zip(scores,labels)]
        quality = sum(p[0] if p else -1 for p in picks)
        if best is None or quality > best[0]:
            best = quality,picks
    return best[1]


def refine(mesh, images, labels=None):
    mat = mesh.visual.material
    atlas = np.asarray(mat.baseColorTexture.convert('RGB'))
    h,w = atlas.shape[:2]
    uv = np.asarray(mesh.visual.uv,dtype='f').copy()
    uv[:,1] = 1-uv[:,1]
    uv *= [w-1,h-1]
    vertices = np.asarray(mesh.vertices,dtype='f')
    faces = np.asarray(mesh.faces,dtype=np.int32)
    session = None
    if any(im.getchannel('A').getextrema()[0] == 255 for im in images):
        from rembg import new_session
        session = new_session('u2net', providers=['CPUExecutionProvider'])
    photos,edges = zip(*(foreground(im, session) for im in images))
    del session
    refs = [None if p is None else cv2.resize(p,(64,64),interpolation=cv2.INTER_AREA) for p in photos]
    scores = [[] for _ in photos]
    grid_y,grid_x = np.indices((64,64),dtype='f')

    def evaluate(yaw,elev,indices):
        screen,_,aspect = camera(vertices,yaw,elev)
        screen[:,:2] *= 63
        face,bary,_ = raster(screen,faces,64,64)
        hit = face>=0
        tex = interpolate(uv,faces[face[hit]],bary[hit]).round().astype(int)
        render = np.zeros((64,64,3),np.float32)
        render[hit] = atlas[tex[:,1].clip(0,h-1),tex[:,0].clip(0,w-1)]
        for i in indices:
            ref = refs[i]
            if ref is None:
                continue
            mask = ref[:,:,3]>128
            ratio = photos[i].shape[1]/photos[i].shape[0]
            best = None
            for window in visible_windows(edges[i],ratio,aspect):
                sx,sw,sy,sh = window
                if sw==sh==1:
                    visible,colors = hit,render
                else:
                    mx,my = grid_x*sw+63*sx,grid_y*sh+63*sy
                    visible = cv2.remap(hit.astype('uint8'),mx,my,cv2.INTER_NEAREST)>0
                    colors = cv2.remap(render,mx,my,cv2.INTER_LINEAR)
                overlap = mask & visible
                if not overlap.any():
                    continue
                iou = overlap.sum()/(mask|visible).sum()
                target = ref[:,:,:3][overlap].astype('f')
                projected = colors[overlap]
                gain = np.clip((target.mean()+1)/(projected.mean()+1),.6,1.7)
                error = np.abs(target-projected*gain).mean()/255
                aspect_error = abs(np.log(aspect*sw/(ratio*sh)))
                score = iou-.72*error-.18*min(aspect_error,2)
                candidate = (score,yaw,elev,iou,error,window)
                if best is None or score>best[0]:
                    best = candidate
            if best is not None:
                scores[i].append(best)

    for elev in (-90,-45,-20,0,20,45,90):
        for yaw in range(0,360,30):
            evaluate(yaw,elev,range(len(refs)))
    supplied_labels = labels or ['unknown']*len(images)
    labels = list(supplied_labels)
    for i,(ref,label) in enumerate(zip(refs,labels)):
        if ref is None or 'three_quarter' not in label:
            continue
        # Cross-check oblique predictions on bilaterally symmetric end views.
        gray = cv2.cvtColor(ref[:,:,:3],cv2.COLOR_RGB2GRAY).astype('f')
        mask = ref[:,:,3]>128
        overlap = mask & mask[:,::-1]
        values, mirror = gray[overlap],gray[:,::-1][overlap]
        if (overlap.sum() > .87*mask.sum() and values.size and values.std()>5
                and np.corrcoef(values,mirror)[0,1]>.72):
            labels[i] = label.split('_')[0]
    for i,match in enumerate(joint_cameras(scores,labels)):
        if match:
            a,b = match[1:3]
            for da in (-12,0,12):
                for db in (-8,0,8):
                    evaluate(a+da,np.clip(b+db,-90,90),[i])

    owner,bary,_ = raster(np.column_stack([uv,np.zeros(len(uv),'f')]),faces,h,w)
    normals = np.asarray(mesh.vertex_normals,dtype='f')
    weighted = np.zeros((h,w,3),'f')
    weights = np.zeros((h,w),'f')
    report = []
    for pic,match,label,candidates in zip(photos,joint_cameras(scores,labels),labels,scores):
        if not match:
            report.append(dict(applied=False,label=label,reason='no eligible camera'))
            continue
        score,yaw,elev,iou,error,window = match
        if label in ('unknown','oblique') and abs(elev)<65 and any(
            abs((other[1]-yaw+180)%360-180)>140 and abs(other[2]-elev)<25
            and other[0]>=score-.025 for other in candidates
        ):
            report.append(dict(applied=False,label=label,reason='ambiguous opposite directions'))
            continue
        applied = bool(iou>=.70 and score>=.52 and error<.36)
        report.append(dict(applied=applied,label=label,yaw=float(yaw),elevation=float(elev),silhouette_iou=round(float(iou),3),score=round(float(score),3)))
        if not applied:
            continue
        ph,pw = pic.shape[:2]
        projected,direction,_ = camera(vertices,yaw,elev)
        sx,sw,sy,sh = window
        projected[:,:2] = (projected[:,:2]-[sx,sy])/[sw,sh]
        report[-1]['visible_window'] = list(window)
        projected[:,:2] *= [pw-1,ph-1]
        _,_,depth = raster(projected,faces,ph,pw)
        alpha = cv2.erode(pic[:,:,3],np.ones((3,3),np.uint8))
        tolerance = np.linalg.norm(np.ptp(vertices,axis=0))*.009
        for row in range(0,h,64):
            block = owner[row:row+64]
            yy,xx = np.where(block>=0)
            yy += row
            tri = faces[owner[yy,xx]]
            frac = bary[yy,xx]
            q = interpolate(projected,tri,frac)
            px,py = np.rint(q[:,:2]).astype(int).T
            inside = (px>=0)&(px<pw)&(py>=0)&(py<ph)
            px,py = px.clip(0,pw-1),py.clip(0,ph-1)
            depth_ok = inside & (np.abs(depth[py,px]-q[:,2])<tolerance)
            facing = interpolate(normals,tri,frac) @ direction
            weight = np.maximum(facing,0)**8 * depth_ok * (alpha[py,px]>225) * score**4
            weighted[yy,xx] += weight[:,None]*pic[py,px,:3]
            weights[yy,xx] += weight
    mix = np.minimum(weights/.065,1)
    fused = atlas*(1-mix[:,:,None])+weighted*(mix/np.maximum(weights,1e-10))[:,:,None]
    mat.baseColorTexture = Image.fromarray(np.uint8(np.clip(fused,0,255)))
    for item,label in zip(report,supplied_labels):
        item['supplied_label'] = label
    return dict(reference_registration=report,photo_texture_fraction=round(float(np.count_nonzero(mix>.5)/max(1,(owner>=0).sum())),4))


def unlit(document):
    document.setdefault('extensionsUsed',[]).append('KHR_materials_unlit')
    for material in document['materials']:
        material.setdefault('extensions',{})['KHR_materials_unlit'] = {}
