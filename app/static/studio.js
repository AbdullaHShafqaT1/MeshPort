const $=id=>document.getElementById(id);let files=[],connected=false,localAvailable=false,mode='disconnected',running=false,model=null,scene,camera,renderer,controls;const urls=[];
const angles = new WeakMap();
let importing = false, connecting = false, modelLoading = false;
let autoTimer, pendingUpload = false;
let currentSheet = null, sheetPreviewUrl = null;
function openImagePreview(src, title) {
    if (!src) return;
    $('modalImg').src = src;
    $('modalTitle').textContent = title || 'Image preview';
    $('modalOpenTab').href = src;
    $('modalDownload').href = src;
    $('modalDownload').download = title || 'reference.png';
    $('imageModal').hidden = false;
}
function closeImagePreview() {
    $('imageModal').hidden = true;
    $('modalImg').src = '';
}
function showSheetFile(file) {
    if (!file) return;
    currentSheet = file;
    if (sheetPreviewUrl) URL.revokeObjectURL(sheetPreviewUrl);
    sheetPreviewUrl = URL.createObjectURL(file);
    $('sheetName').textContent = file.name;
    $('sheetPreviewImg').src = sheetPreviewUrl;
    $('sheetPreviewTitle').textContent = file.name;
    $('sheetPreviewTitle').title = file.name;
    $('sheetPreviewInfo').textContent = (file.size / 1024 ** 2).toFixed(2) + ' MiB';
    $('sheetPreview').hidden = false;
    $('sheetDrop').hidden = true;
}
function queueGeneration() {
    clearTimeout(autoTimer);
    if (!pendingUpload || !canGenerate() || running || importing || !files.length || !$('automatic').checked) return;
    message('Starting automatically. Add another view now or turn off automatic generation.');
    autoTimer = setTimeout(() => {
        if (pendingUpload && !$('generate').disabled && $('automatic').checked) $('generate').click();
    }, 2000);
}
$('automatic').onchange = queueGeneration;
$('connection').oninput = () => {
    if (!running && !$('connect').disabled && /^https:\/\/[a-z0-9-]+\.trycloudflare\.com\/#key=[\w-]{32,}$/.test($('connection').value.trim())) $('connect').click();
};
const conditioningLabels = new Map([
    ['single-image', 'Single image'],
    ['multi-image-stochastic', 'Multi-view · Stochastic'],
    ['multi-image-multidiffusion', 'Multi-view · MultiDiffusion'],
    ['local-mlx-pbr-photofinish', 'Local MLX · 4K PBR + Photo Finish'],
]);
function canGenerate(){return mode==='local'||mode==='colab';}
function applyBackendState(s){
    connected=Boolean(s.connected);
    localAvailable=Boolean(s.local_available);
    mode=s.mode||(connected?'colab':(localAvailable?'local':'disconnected'));
    $('disconnect').hidden=!(connected&&localAvailable);
    if(mode==='colab'){
        $('connectionBadge').textContent='● Colab connected';
        $('connectionBadge').classList.add('connected');
        $('connectionInfo').textContent=conditioningLabels.get(s.multi_view_conditioning)||'Connected to Colab GPU.';
        $('qualitySummary').textContent='16 sampling steps · 2K textures';
        $('qualitySub').textContent='TRELLIS image large · cloud generation · local exports.';
        $('heroSubtitle').textContent='From a reference image to a textured 3D model. Powered by your Colab GPU.';
    } else if(mode==='local'){
        const info=s.local_info||{};
        $('connectionBadge').textContent='● Local Mac (MLX)';
        $('connectionBadge').classList.add('connected');
        $('connectionInfo').textContent=[info.model||'Hunyuan3D-2.1 MLX',info.gpu||'Apple Silicon (Metal / MLX)',`${(info.target_faces||40000).toLocaleString()} faces · ${info.texture_size||4096}px PBR`].join(' · ');
        $('qualitySummary').textContent=`${(info.target_faces||40000).toLocaleString()} faces · ${Math.round((info.texture_size||4096)/1024)}K PBR textures + Photo Finish`;
        $('qualitySub').textContent='Hunyuan3D-2.1 MLX · local Apple Silicon · local exports.';
        $('heroSubtitle').textContent='From a reference image to a 4K PBR textured 3D model. Running locally on your Mac.';
    } else {
        $('connectionBadge').textContent='○ Engine disconnected';
        $('connectionBadge').classList.remove('connected');
        $('connectionInfo').textContent='Run setup.sh for local Apple Silicon generation or paste a Colab URL.';
    }
}
function message(text,error=false,loading=false){
    const el=$('message');
    if (loading && el.replaceChildren) {
        const sp=document.createElement('span');sp.className='spinner inline-spinner';
        const tx=document.createElement('span');tx.textContent=text;
        el.replaceChildren(sp,tx);
    } else {
        el.textContent=text;
    }
    el.classList.toggle('error',error);
}
function setSheetStatus(text,loading=false){
    const el=$('sheetStatus');
    if (loading && el.replaceChildren) {
        const sp=document.createElement('span');sp.className='spinner inline-spinner';
        const tx=document.createElement('span');tx.textContent=text;
        el.replaceChildren(sp,tx);
    } else {
        el.textContent=text;
    }
}
function ready(){
    const busy = running || importing;
    const hasSheet = Boolean($('sheet').files?.length || currentSheet);
    $('generate').disabled = !canGenerate() || !files.length || Boolean(validateReferences(files)) || busy;
    $('connect').disabled = busy || connecting;
    $('disconnect').disabled = busy || connecting;
    $('images').disabled = busy;
    $('clear').disabled = busy;
    $('sheet').disabled = busy;
    $('changeSheet').disabled = busy;
    $('extract').disabled = busy || !hasSheet;
    document.querySelectorAll('.remove-view').forEach(button => button.disabled = busy);
}
async function api(path,body){const response=await fetch(path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined});const value=await response.json();if(!response.ok)throw Error(typeof value.detail==='string'?value.detail:'The request failed. Please try again.');return value}
$('connect').onclick=async()=>{
    const button=$('connect');
    connecting=true;
    ready();
    button.innerHTML='<span class="spinner inline-spinner"></span> Connecting…';
    $('connectionBadge').innerHTML='<span class="spinner inline-spinner"></span> Connecting to Colab…';
    $('connectionBadge').classList.remove('connected');
    try{
        const info=await api('/api/connect',{url:$('connection').value});
        applyBackendState({connected:true,local_available:localAvailable,mode:'colab',multi_view_conditioning:info.multi_view_conditioning});
        $('connectionInfo').textContent=[info.model,info.gpu,conditioningLabels.get(info.multi_view_conditioning)].filter(Boolean).join(' · ');
        message(info.labeled_cameras ? (files.length?'Ready to create your model.':'Connected to Colab. Add reference images.') : 'Connected to an older notebook. Run the updated notebook for direction-aware texturing.');
    }catch(e){
        pendingUpload=false;
        const st=await api('/api/status').catch(()=>({connected:false,local_available:localAvailable}));
        applyBackendState(st);
        message(e.message,true);
    }finally{
        connecting=false;
        button.textContent='Connect ↗';
        ready();
        queueGeneration();
    }
};
$('disconnect').onclick=async()=>{
    connecting=true;
    ready();
    try{
        const st=await api('/api/disconnect',{});
        applyBackendState(st);
        message(files.length?'Switched to Local Mac (MLX). Ready to create your model.':'Switched to Local Mac (MLX). Add reference images.');
    }catch(e){
        message(e.message,true);
    }finally{
        connecting=false;
        ready();
        queueGeneration();
    }
};
function renderViews(){
    urls.splice(0).forEach(URL.revokeObjectURL);
    $('thumbnails').replaceChildren();
    files.forEach((file,index)=>{
        const card=document.createElement('div');card.className='reference-view';
        const img=document.createElement('img');img.src=URL.createObjectURL(file);urls.push(img.src);img.alt=file.name;
        img.title='Click to preview '+file.name;
        img.onclick=()=>openImagePreview(img.src,file.name);
        const name=document.createElement('span');name.textContent=file.name;name.title='Click to preview '+file.name;
        name.onclick=()=>openImagePreview(img.src,file.name);
        const remove=document.createElement('button');remove.type='button';remove.className='remove-view';remove.textContent='Remove';
        remove.setAttribute('aria-label','Remove '+file.name);remove.disabled=running || importing;
        remove.onclick=()=>{if(running || importing)return;files.splice(index,1);renderViews();queueGeneration()};
        card.append(img,name,remove);$('thumbnails').append(card);
    });
    $('clear').hidden=!files.length;
    $('viewSummary').textContent=files.length===0?'No views added.':`${files.length} / ${referenceLimit} views · ${files.length===1?'Single-image':'Multi-view'} generation`;
    message(files.length?'Click any image to preview, add more angles, or create your model.':canGenerate()?'Ready. Add reference images.':'Connect Colab or enable local engine to begin.');
    ready();
}
function choose(selected){
    if (running || importing || !selected.length) return false;
    const next = [...files, ...Array.from(selected)];
    const error = validateReferences(next);
    if (error) {
        message(error, true);
        return false;
    }
    files = next;
    pendingUpload = true;
    renderViews();
    queueGeneration();
    return true;
}
$('images').onchange=e=>{choose(e.target.files);e.target.value=''};
$('clear').onclick=()=>{if(running || importing)return;files=[];$('images').value='';renderViews()};
$('drop').ondragover=e=>{e.preventDefault();$('drop').classList.add('dragover')};
$('drop').ondragleave=()=>$('drop').classList.remove('dragover');
$('drop').ondrop=e=>{e.preventDefault();$('drop').classList.remove('dragover');choose(e.dataTransfer.files)};
$('sheet').onchange = () => {
    const file = $('sheet').files[0];
    if (file) showSheetFile(file);
    setSheetStatus('');
    clearTimeout(autoTimer);
    pendingUpload = false;
    ready();
    if (file && $('automatic').checked && !$('extract').disabled) $('extract').click();
};
$('extract').onclick = async () => {
    const sheet = $('sheet').files?.[0] || currentSheet;
    if (!sheet || running || importing) return;
    if (!referenceTypes.has(sheet.type) || sheet.size > 16 * 1024 ** 2) {
        setSheetStatus('Choose a PNG, JPG or WebP sheet up to 16 MiB.');
        return;
    }
    showSheetFile(sheet);
    importing = true;
    pendingUpload = false;
    clearTimeout(autoTimer);
    ready();
    $('sheetExtractingOverlay').hidden = false;
    $('extract').innerHTML = '<span class="spinner inline-spinner"></span> Extracting angles…';
    setSheetStatus('Finding views in the sheet…', true);
    try {
        const response = await fetch('/api/angle-preview', {
            method: 'POST', headers: {'Content-Type': sheet.type}, body: sheet
        });
        const result = await response.json();
        if (!response.ok) throw Error(result.detail || 'Could not extract angles.');
        if (!result.views.length) throw Error('No views found. Try another sheet or add individual photos below.');
        const crops = await Promise.all(result.views.map(async (view, index) => {
            if (!view.image?.startsWith('data:image/png;base64,')) throw Error('Invalid crop returned.');
            const blob = await (await fetch(view.image)).blob();
            const file = new File([blob], (index + 1) + ' ' + view.angle.replaceAll('_', ' ') + '.png', {type: 'image/png'});
            angles.set(file, view.angle);
            return file;
        }));
        files.push(...crops);
        pendingUpload = true;
        $('sheet').value = '';
        renderViews();
        setSheetStatus(crops.length + (crops.length === 1 ? ' view extracted below.' : ' views extracted below.'));
        const issue = validateReferences(files);
        message(issue ? issue + ' Remove extra views below.' : 'Review or preview the angles below, then click Create 3D model.', Boolean(issue));
    } catch (error) {
        setSheetStatus(error.message);
    } finally {
        importing = false;
        $('sheetExtractingOverlay').hidden = true;
        $('extract').textContent = currentSheet ? 'Extract angles again' : 'Extract angles';
        ready();
        queueGeneration();
    }
};
const encode=file=>new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=()=>reject(Error('Could not read '+file.name));reader.readAsDataURL(file)});

$('changeSheet').onclick = () => { if (!running && !importing) $('sheet').click(); };
$('sheetPreviewOpen').onclick = () => { if (sheetPreviewUrl) openImagePreview(sheetPreviewUrl, currentSheet?.name || 'Reference sheet'); };
$('previewSheetBtn').onclick = () => { if (sheetPreviewUrl) openImagePreview(sheetPreviewUrl, currentSheet?.name || 'Reference sheet'); };
$('sheetDrop').ondragover = e => { e.preventDefault(); $('sheetDrop').classList.add('dragover'); };
$('sheetDrop').ondragleave = () => $('sheetDrop').classList.remove('dragover');
$('sheetDrop').ondrop = e => {
    e.preventDefault();
    $('sheetDrop').classList.remove('dragover');
    if (running || importing || !e.dataTransfer.files?.length) return;
    showSheetFile(e.dataTransfer.files[0]);
    ready();
    if ($('automatic').checked && !$('extract').disabled) $('extract').click();
};
$('modalClose').onclick = closeImagePreview;
$('imageModal').onclick = e => { if (e.target === $('imageModal')) closeImagePreview(); };
window.addEventListener('keydown', e => { if (e.key === 'Escape' && !$('imageModal').hidden) closeImagePreview(); });

function applyTheme(mode) {
    const light = mode === 'light';
    document.body.classList.toggle('light-mode', light);
    $('themeToggle').textContent = light ? '☾ Dark mode' : '☀ Light mode';
    try { localStorage.setItem('meshport-theme', light ? 'light' : 'dark'); } catch {}
}
try { applyTheme(localStorage.getItem('meshport-theme') || 'dark'); } catch { applyTheme('dark'); }
$('themeToggle').onclick = () => applyTheme(document.body.classList.contains('light-mode') ? 'dark' : 'light');

$('generate').onclick=async()=>{
    pendingUpload=false;
    clearTimeout(autoTimer);
    running=true;
    ready();
    $('result').hidden=true;
    $('busy').hidden=false;
    $('busyNote').textContent=mode==='colab'?'You can leave this page open while Colab works.':'Running locally on Apple Silicon (MLX). Each stage frees memory automatically.';
    $('stage').textContent=mode==='colab'?'Sending images to Colab':'Preparing local Apple Silicon generation';
    $('generate').innerHTML='<span class="spinner inline-spinner"></span> Generating 3D model…';
    const start=Date.now();
    const timer=setInterval(()=>{
        const seconds=Math.floor((Date.now()-start)/1000);
        $('elapsed').textContent=Math.floor(seconds/60)+':'+String(seconds%60).padStart(2,'0');
    },1000);
    try{
        const labels=files.map(file=>angles.get(file)||'unknown');
        const payload={images:await Promise.all(files.map(encode))};
        if(labels.some(label=>label!=='unknown'))payload.angles=labels;
        const job=await api('/api/jobs',payload);
        let result;
        for(;;){
            result=await api('/api/jobs/'+job.id);
            $('stage').textContent=result.stage;
            if(result.status!=='running')break;
            await new Promise(r=>setTimeout(r,2000));
        }
        if(result.glb)history.replaceState(history.state,'','/?job='+encodeURIComponent(job.id));
        showResult(result);
        if(result.status==='failed')throw Error(result.error);
        message('Your model is ready. Rotate it above or download your FBX.');
    }catch(e){
        message(e.message,true);
    }finally{
        running=false;
        clearInterval(timer);
        if(!modelLoading)$('busy').hidden=true;
        $('generate').innerHTML='Create 3D model <span>↗</span>';
        ready();
    }
};
function frame(){if(!model)return;const box=new THREE.Box3().setFromObject(model),size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3());const distance=Math.max(size.x,size.y,size.z)*2;controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(distance,.65*distance,distance));camera.near=distance/1000;camera.far=distance*100;camera.updateProjectionMatrix();controls.update()}
function loadModel(url){
    if(!renderer){message('Preview is unavailable because WebGL could not start. Your download links still work.',true);return}
    modelLoading = true;
    $('busy').hidden = false;
    $('stage').textContent = 'Loading 3D model into viewport…';
    $('elapsed').textContent = '';
    $('busyNote').textContent = 'Preparing 3D geometry and photo texture.';
    const cacheBusted = url + (url.includes('?') ? '&' : '?') + 't=' + Date.now();
    new THREE.GLTFLoader().load(cacheBusted,gltf=>{
        modelLoading = false;
        if(!running)$('busy').hidden = true;
        if(model){scene.remove(model);model.traverse(o=>{if(o.geometry)o.geometry.dispose();const mats=Array.isArray(o.material)?o.material:[o.material];mats.filter(Boolean).forEach(m=>{for(const v of Object.values(m))if(v?.isTexture)v.dispose();m.dispose()})})}
        model=gltf.scene;
        scene.add(model);
        $('empty').hidden=true;
        $('wireframe').setAttribute('aria-pressed','false');
        frame();
    },undefined,()=>{
        modelLoading = false;
        if(!running)$('busy').hidden = true;
        message('The model was saved, but the preview could not load. Try the GLB download.',true);
    });
}
try{const container=$('viewport');scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(40,1,.01,100);renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputEncoding=THREE.sRGBEncoding;renderer.toneMapping=THREE.NoToneMapping;container.append(renderer.domElement);controls=new THREE.OrbitControls(camera,renderer.domElement);controls.enableDamping=true;scene.add(new THREE.HemisphereLight(0xffffff,0x7c886c,1.6));for(const [x,y,z,power] of [[4,6,4,1.8],[-4,2,-2,.8]]){const light=new THREE.DirectionalLight(0xffffff,power);light.position.set(x,y,z);scene.add(light)}new ResizeObserver(()=>{const w=container.clientWidth,h=container.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix()}).observe(container);function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera)}animate()}catch(e){$('empty').querySelector('p').textContent='3D preview needs WebGL. Downloads will still work.'}
$('reset').onclick=frame;$('wireframe').onclick=()=>{if(!model)return;const enabled=$('wireframe').getAttribute('aria-pressed')!=='true';$('wireframe').setAttribute('aria-pressed',String(enabled));model.traverse(o=>{const mats=Array.isArray(o.material)?o.material:[o.material];mats.filter(Boolean).forEach(m=>m.wireframe=enabled)})};
api('/api/status').then(s=>{
    applyBackendState(s);
    if(files.length===0 && canGenerate()) message(mode==='local'?'Local Apple Silicon engine ready. Add reference images.':'Colab connected. Add reference images.');
    ready();
    queueGeneration();
}).catch(()=>{
    applyBackendState({connected:false,local_available:false,mode:'disconnected'});
});

async function importCrops(token) {
    importing = true;
    ready();
    message('Loading the selected crops…', false, true);
    try {
        const incoming = await referenceDrafts.take(token);
        if (!incoming?.length) throw new Error('The selected crops expired. Open the angle separator to choose them again.');
        importing = false;
        const restored = incoming.map(item => {
            if (!item.file) return item;
            angles.set(item.file,item.angle);
            return item.file;
        });
        if (choose(restored)) message('Views loaded from the angle separator. Click any thumbnail to preview.');
    } catch (error) {
        message(error.message || 'Could not load the selected crops. Open the angle separator to try again.', true);
    } finally {
        importing = false;
        history.replaceState(history.state, '', location.pathname + location.search);
        ready();
    }
}

if (location.hash.startsWith('#views=')) importCrops(location.hash.slice('#views='.length));

function showResult(result) {
    if (!result.glb) return;
    $('result').hidden = false;
    $('glb').href = result.glb;
    $('fbx').hidden = !result.fbx;
    $('bundle').hidden = !result.bundle;
    if (result.fbx) $('fbx').href = result.fbx;
    if (result.bundle) $('bundle').href = result.bundle;
    loadModel(result.glb);
    $('modelLabel').textContent = 'Your generated model';
    const details = result.details || {};
    $('details').textContent = [
        details.input_count && `${details.input_count} reference ${details.input_count === 1 ? 'image' : 'views'}`,
        conditioningLabels.get(details.conditioning),
        details.reference_registration && `${details.reference_registration.filter(view=>view.applied).length} / ${details.reference_registration.length} photo views matched`,
        details.faces ? `${details.faces.toLocaleString()} faces` : 'GLB saved locally.',
        details.texture_size && `${details.texture_size} texture`,
        typeof details.watertight === 'boolean' && (details.watertight ? 'Watertight mesh' : 'Open mesh'),
    ].filter(Boolean).join(' · ');
}

if (new URLSearchParams(location.search).get('sample') === 'car') {
    message('Loading sample model…', false, true);
    api('/api/sample').then(result => {
        showResult(result);
        $('modelLabel').textContent = 'Car from TRELLIS on Colab';
        const input = document.createElement('img');
        input.src = '/api/sample/input';
        input.alt = 'Original image used for this car';
        input.title = 'Click to preview sample input';
        input.onclick = () => openImagePreview(input.src, 'Sample reference image');
        $('thumbnails').append(input);
        message('Sample model loaded. Drag to rotate and scroll to zoom.');
    }).catch(error => message(error.message, true));
}

const savedJob = new URLSearchParams(location.search).get('job');
if (savedJob && /^[a-f0-9]{32}$/.test(savedJob)) {
    message('Loading saved model…', false, true);
    api('/api/jobs/'+savedJob).then(result=>{
        showResult(result);
        if(result.status==='complete') message('Saved model loaded. Drag to rotate and scroll to zoom.');
    }).catch(error=>message(error.message,true));
}
