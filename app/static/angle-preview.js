const byId = (id) => document.getElementById(id);
const fileInput = byId('file');
const pick = byId('pick');
const analyze = byId('analyze');
const remove = byId('remove');
const changeFile = byId('change-file');
const source = byId('source-image');
const boxes = byId('crop-outlines');
const toggle = byId('toggle-boxes');
const crops = byId('crops');
const sendToStudio = byId('send-to-studio');
const pngPrefix = 'data:image/png;base64,';
const labels = new Map([
    ['front', 'Front'],
    ['rear', 'Rear'],
    ['left', 'Left side'],
    ['right', 'Right side'],
    ['top', 'Top'],
    ['bottom', 'Bottom'],
    ['front_three_quarter', 'Front three-quarter'],
    ['rear_three_quarter', 'Rear three-quarter'],
    ['perspective', 'Perspective'],
    ['oblique', 'Perspective'],
]);
let file = null;
let sourceUrl = null;
let running = false;
let staging = false;
let selections = [];

function openImagePreview(src, title) {
    if (!src || !byId('imageModal')) return;
    byId('modalImg').src = src;
    byId('modalTitle').textContent = title || 'Image preview';
    byId('modalOpenTab').href = src;
    byId('modalDownload').href = src;
    byId('modalDownload').download = title || 'crop.png';
    byId('imageModal').hidden = false;
}
function closeImagePreview() {
    if (!byId('imageModal')) return;
    byId('imageModal').hidden = true;
    byId('modalImg').src = '';
}
if (byId('modalClose')) {
    byId('modalClose').addEventListener('click', closeImagePreview);
    byId('imageModal').addEventListener('click', (e) => { if (e.target === byId('imageModal')) closeImagePreview(); });
    window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !byId('imageModal').hidden) closeImagePreview(); });
}
function applyTheme(mode) {
    const light = mode === 'light';
    document.body.classList.toggle('light-mode', light);
    if (byId('themeToggle')) byId('themeToggle').textContent = light ? '☾ Dark mode' : '☀ Light mode';
    try { localStorage.setItem('meshport-theme', light ? 'light' : 'dark'); } catch {}
}
try { applyTheme(localStorage.getItem('meshport-theme') || 'dark'); } catch { applyTheme('dark'); }
if (byId('themeToggle')) {
    byId('themeToggle').addEventListener('click', () => applyTheme(document.body.classList.contains('light-mode') ? 'dark' : 'light'));
}

function setStatus(message, error = false) {
    byId('status-text').textContent = message;
    byId('status').classList.toggle('error', error);
}

function updateControls() {
    pick.disabled = running || staging;
    remove.disabled = running || staging;
    if (changeFile) changeFile.disabled = running || staging;
    analyze.disabled = running || staging || !file;
    analyze.innerHTML = running
        ? '<span class="spinner" aria-hidden="true"></span> Separating angles…'
        : 'Separate angles <span aria-hidden="true">↗</span>';
    byId('status').querySelector('.spinner').hidden = !running && !staging;
    byId('results-title').closest('section').setAttribute('aria-busy', String(running));
    updateHandoff();
}

function updateHandoff() {
    const selected = selections.filter(view => view.checkbox.checked);
    const size = selected.reduce((bytes, view) => bytes + view.size, 0);
    const error = validateReferences(selected);
    byId('handoff').hidden = !selections.length;
    byId('selected-summary').textContent = `${selected.length} of ${selections.length} views selected · ${(size / 1024 ** 2).toFixed(2)} MiB`;
    byId('handoff-error').textContent = error ? error + ' Deselect views to continue.' : '';
    sendToStudio.disabled = staging || running || !selected.length || Boolean(error);
    sendToStudio.innerHTML = staging
        ? '<span class="spinner" aria-hidden="true"></span> Opening in studio…'
        : 'Use selected views in studio <span aria-hidden="true">↗</span>';
    for (const view of selections) view.checkbox.disabled = staging;
}

function showOutlines(visible) {
    boxes.hidden = !visible;
    toggle.setAttribute('aria-pressed', String(visible));
}

function clearResults(message = 'Crops will appear here after analysis.') {
    selections = [];
    crops.replaceChildren();
    boxes.replaceChildren();
    byId('missing').replaceChildren();
    byId('coverage').hidden = true;
    byId('result-meta').textContent = '';
    byId('results-empty').textContent = message;
    byId('results-empty').hidden = false;
    toggle.hidden = true;
    showOutlines(true);
    updateHandoff();
}

function replaceFile(next) {
    if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    file = next;
    sourceUrl = file ? URL.createObjectURL(file) : null;
    if (sourceUrl) source.src = sourceUrl;
    else source.removeAttribute('src');
    byId('source-frame').hidden = !file;
    byId('source-empty').hidden = Boolean(file);
    byId('file-info').hidden = !file;
    byId('file-name').textContent = file?.name ?? '';
    byId('file-name').title = file?.name ?? '';
    byId('file-size').textContent = file ? `${(file.size / 1024 ** 2).toFixed(2)} MiB` : '';
    clearResults();
    updateControls();
    setStatus(file ? 'Ready to separate the views.' : 'Add an image to get started.');
}

function choose(files) {
    if (running || staging || !files.length) return;
    if (files.length !== 1) {
        setStatus('Choose one reference sheet at a time.', true);
        return;
    }
    const candidate = files[0];
    if (!referenceTypes.has(candidate.type)) {
        setStatus('Choose a PNG, JPG or WebP image.', true);
        return;
    }
    if (!candidate.size || candidate.size > 16 * 1024 ** 2) {
        setStatus('Choose a nonempty image up to 16 MiB.', true);
        return;
    }
    replaceFile(candidate);
}

function renderCrop(view, number) {
    const label = labels.get(view.angle) ?? 'Unclassified view';
    const card = document.createElement('figure');
    card.className = 'crop-card';
    const image = document.createElement('img');
    image.src = view.image;
    image.alt = `${label} crop`;
    image.title = `Click to preview ${label} crop`;
    image.addEventListener('click', () => openImagePreview(view.image, `${number}. ${label}`));
    const caption = document.createElement('figcaption');
    const heading = document.createElement('div');
    heading.className = 'crop-label';
    const badge = document.createElement('span');
    badge.className = 'crop-index';
    badge.textContent = number;
    heading.append(badge, label);
    const download = document.createElement('a');
    download.className = 'crop-download';
    download.href = view.image;
    download.download = `meshport-${labels.has(view.angle) ? view.angle.replaceAll('_', '-') : 'view'}-${number}.png`;
    download.textContent = 'Download PNG ↓';
    download.setAttribute('aria-label', `Download crop ${number}, ${label}, as PNG`);
    const choice = document.createElement('label');
    choice.className = 'crop-choice';
    const checkbox = document.createElement('input');
    checkbox.className = 'crop-select';
    checkbox.type = 'checkbox';
    checkbox.checked = true;
    checkbox.setAttribute('aria-label', `Use crop ${number}, ${label}, in the model`);
    checkbox.addEventListener('change', updateHandoff);
    choice.append(checkbox, 'Use in model');
    const padding = view.image.endsWith('==') ? 2 : Number(view.image.endsWith('='));
    const size = (view.image.length - pngPrefix.length) * 3 / 4 - padding;
    selections.push({ angle: view.angle, image: view.image, name: download.download, size, type: 'image/png', checkbox });
    caption.append(heading, download, choice);
    card.append(image, caption);
    return card;
}

function renderOutline([left, top, right, bottom], number, width, height) {
    const outline = document.createElement('div');
    outline.className = 'crop-box';
    outline.style.left = `${100 * left / width}%`;
    outline.style.top = `${100 * top / height}%`;
    outline.style.width = `${100 * (right - left) / width}%`;
    outline.style.height = `${100 * (bottom - top) / height}%`;
    const badge = document.createElement('span');
    badge.textContent = number;
    outline.append(badge);
    return outline;
}

function showResult({ views, missing, model, width, height }) {
    const count = views.length;
    views.forEach((view, index) => {
        crops.append(renderCrop(view, index + 1));
        boxes.append(renderOutline(view.box, index + 1, width, height));
    });
    toggle.hidden = !count;
    byId('result-meta').textContent = model || '';
    byId('results-empty').hidden = Boolean(count);
    byId('results-empty').textContent = 'No views were detected in this image.';
    for (const angle of missing) {
        const chip = document.createElement('li');
        chip.textContent = labels.get(angle) ?? angle;
        byId('missing').append(chip);
    }
    byId('coverage').hidden = !missing.length;
    setStatus(count ? `${count} ${count === 1 ? 'crop is' : 'crops are'} ready to review. Click any image to preview.` : 'Analysis finished without finding a view.');
}

pick.addEventListener('click', () => fileInput.click());
if (changeFile) changeFile.addEventListener('click', () => fileInput.click());
source.addEventListener('click', () => { if (sourceUrl) openImagePreview(sourceUrl, file?.name || 'Original sheet'); });
fileInput.addEventListener('change', () => {
    choose(fileInput.files);
    fileInput.value = '';
});
pick.addEventListener('dragover', (event) => {
    event.preventDefault();
    if (!running && !staging) pick.classList.add('dragover');
});
pick.addEventListener('dragleave', () => pick.classList.remove('dragover'));
pick.addEventListener('drop', (event) => {
    event.preventDefault();
    pick.classList.remove('dragover');
    choose(event.dataTransfer.files);
});
remove.addEventListener('click', () => {
    replaceFile(null);
    pick.focus();
});
toggle.addEventListener('click', () => showOutlines(boxes.hidden));

sendToStudio.addEventListener('click', async () => {
    if (sendToStudio.disabled) return;
    staging = true;
    updateControls();
    try {
        const selected = selections.filter(view => view.checkbox.checked);
        const files = await Promise.all(selected.map(async view => {
            const blob = await (await fetch(view.image)).blob();
            return {file: new File([blob], view.name, { type: view.type }), angle: view.angle};
        }));
        const token = await referenceDrafts.stage(files);
        location.assign('/#views=' + token);
    } catch {
        staging = false;
        updateControls();
        byId('handoff-error').textContent = 'Could not open the views in studio. Try again or download the crops.';
    }
});

window.addEventListener('pageshow', event => {
    if (!event.persisted) return;
    staging = false;
    updateControls();
});

analyze.addEventListener('click', async () => {
    if (running || staging || !file) return;
    running = true;
    updateControls();
    clearResults('Looking for views in your reference sheet…');
    setStatus('Analyzing locally with Ollama. The first run can take longer while the model loads.');
    try {
        const response = await fetch('/api/angle-preview', {
            method: 'POST',
            headers: { 'Content-Type': file.type },
            body: file,
        });
        const result = await response.json().catch(() => null);
        if (!response.ok) {
            throw new Error(typeof result?.detail === 'string' ? result.detail : 'Could not analyze this image. Please try again.');
        }
        if (!result || !Array.isArray(result.views) || !Array.isArray(result.missing) ||
            result.views.some(view => !view || typeof view.image !== 'string' || !view.image.startsWith(pngPrefix))) {
            throw new Error('The preview received an unexpected response.');
        }
        showResult(result);
    } catch (error) {
        clearResults('No crops to show. You can run the analysis again.');
        setStatus(error.message, true);
    } finally {
        running = false;
        updateControls();
    }
});
