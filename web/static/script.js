// ==========================
// GreenMind script.js (fixed)
// ==========================

// Controls for CAM
const camMethodSel  = document.getElementById('cam-method');
const opacitySlider = document.getElementById('opacity-slider');
const opacityVal    = document.getElementById('opacity-val');
const downloadCam   = document.getElementById('download-cam');

let lastOverlayImg = null; // heatmap (transparent PNG) returned from /explain
let stream = null;

const video  = document.getElementById('video');
const canvas = document.getElementById('frame-canvas');

const startBtn = document.getElementById('start-cam');
const stopBtn  = document.getElementById('stop-cam');
const snapBtn  = document.getElementById('snap');
const identifyFromCamBtn = document.getElementById('identify-from-cam');

const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const dropzone  = document.getElementById('dropzone');
const preview   = document.getElementById('preview');
const previewImg= document.getElementById('preview-img');

const identifyBtn = document.getElementById('identify-btn');
const explainBtn  = document.getElementById('explain-btn');
const clearBtn    = document.getElementById('clear-btn');

const loading     = document.getElementById('loading');
const resultWrap  = document.getElementById('result');
const bestLabelEl = document.getElementById('best-label');
const bestScoreEl = document.getElementById('best-score');
const confBar     = document.getElementById('confidence-fill');
const altChips    = document.getElementById('alt-chips');
const factsHtml   = document.getElementById('facts-html');
const explainArea = document.getElementById('explain-area');
const yearEl      = document.getElementById('year');
const toast       = document.getElementById('toast');

// Feedback elements
const fbBlock      = document.getElementById('feedback-block');
const fbCorrectBtn = document.getElementById('fb-correct');
const fbWrongBtn   = document.getElementById('fb-wrong');
const fbCorrection = document.getElementById('fb-correction');
const fbTrueLabel  = document.getElementById('fb-true-label');
const fbNotes      = document.getElementById('fb-notes');
const fbSubmit     = document.getElementById('fb-submit');
const fbToast      = document.getElementById('fb-toast');

let lastIdentifyPayload = null; // store server's /identify JSON
let currentBlob = null;         // original image blob for identify/explain

// Safe no-op: avoid breaking if setSourceBadge is not defined
if (typeof window.setSourceBadge !== 'function') {
  window.setSourceBadge = function(){};
}

if (yearEl) yearEl.textContent = new Date().getFullYear();

// -------- Helpers --------
function showToast(msg, isError=false){
  toast.textContent = msg;
  toast.classList.toggle('error', isError);
  toast.classList.add('show');
  setTimeout(()=> toast.classList.remove('show'), 2600);
}

function sanitize(html){
  return String(html).replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, "");
}

function fileToDataURL(file){
  return new Promise((resolve, reject)=>{
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function setPreview(file){
  currentBlob = file;
  fileToDataURL(file).then(url=>{
    previewImg.src = url;
    preview.classList.remove('hidden');
    resultWrap.classList.add('hidden');
    explainArea.classList.add('hidden');
    explainBtn.disabled = false; // allow Explain right away
    lastOverlayImg = null;       // clear previous overlay
    // Hide stale feedback when a new image is chosen
    hideFeedbackUI();
  }).catch(()=> showToast("Could not preview image", true));
}

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/static/sw.js").catch(console.error);
  });
}

// Draw preview + overlay using opacity slider (multiplies PNG alpha)
function drawCamWithOpacity(){
  const canvasEl = document.getElementById('cam-canvas');
  if (!canvasEl || !lastOverlayImg) return;

  const w = (previewImg.naturalWidth  || lastOverlayImg.width)  || 512;
  const h = (previewImg.naturalHeight || lastOverlayImg.height) || 512;
  canvasEl.width = w; canvasEl.height = h;

  const ctx = canvasEl.getContext('2d');
  try { ctx.drawImage(previewImg, 0, 0, w, h); } catch {}
  const alpha = Math.max(0, Math.min(1, (Number(opacitySlider?.value || 55) / 100)));
  ctx.globalAlpha = alpha;
  ctx.drawImage(lastOverlayImg, 0, 0, w, h);
  ctx.globalAlpha = 1.0;
}

// -------- Camera support check --------
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  startBtn.disabled = true;
  snapBtn.disabled = true;
  identifyFromCamBtn.disabled = true;
  stopBtn.disabled = true;
  showToast('Camera not supported in this browser', true);
}

// -------- Camera functions --------
async function startCamera(){
  try{
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' } }, audio: false
    });
    video.srcObject = stream;
    video.hidden = false;
    await video.play().catch(()=>{});
    snapBtn.disabled = false;
    identifyFromCamBtn.disabled = false;
    stopBtn.disabled = false;
    showToast('Camera started');
  }catch(err){
    showToast('Camera error: ' + err.message, true);
  }
}

function stopCamera(){
  if(stream){
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  video.hidden = true;
  snapBtn.disabled = true;
  identifyFromCamBtn.disabled = true;
  stopBtn.disabled = true;
  showToast('Camera stopped');
}

function captureFrameToBlob(){
  return new Promise((resolve, reject)=>{
    if(!video || video.readyState < 2) return reject(new Error('Camera not ready'));
    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, w, h);
    canvas.toBlob(blob => {
      if(!blob) return reject(new Error('Capture failed'));
      resolve(blob);
    }, 'image/jpeg', 0.9);
  });
}

// -------- Upload / drag & drop --------
browseBtn.addEventListener('click', ()=> fileInput.click());
fileInput.addEventListener('change', e=>{
  const f = e.target.files?.[0];
  if (f) setPreview(f);
});

['dragenter','dragover'].forEach(evt => dropzone.addEventListener(evt, e=>{
  e.preventDefault(); e.stopPropagation(); dropzone.classList.add('hover');
}));
['dragleave','drop'].forEach(evt => dropzone.addEventListener(evt, e=>{
  e.preventDefault(); e.stopPropagation(); dropzone.classList.remove('hover');
}));
dropzone.addEventListener('drop', e=>{
  const f = e.dataTransfer.files?.[0];
  if (f) setPreview(f);
});

// Avoid browser opening file when dropped outside
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => e.preventDefault());
window.addEventListener('dragleave', () => dropzone.classList.remove('hover'));

// ---------- FEEDBACK UI HELPERS ----------
function showFeedbackUI({ openCorrection = false } = {}) {
  if (!fbBlock) return;
  fbBlock.style.display = 'block';
  if (fbCorrection) fbCorrection.classList.toggle('hidden', !openCorrection);
  if (fbTrueLabel) fbTrueLabel.value = '';
  if (fbNotes) fbNotes.value = '';
  if (fbToast) fbToast.style.display = 'none';
}
function hideFeedbackUI() {
  if (!fbBlock) return;
  fbBlock.style.display = 'none';
  if (fbToast) fbToast.style.display = 'none';
  if (fbCorrection) fbCorrection.classList.add('hidden');
}

// -------- Identify --------
identifyBtn.addEventListener('click', async ()=>{
  if (!currentBlob) return showToast("Choose an image first", true);
  identifyBtn.disabled = true;
  loading.classList.remove('hidden');
  resultWrap.classList.add('hidden');
  hideFeedbackUI(); // clear stale feedback while we fetch

  const fd = new FormData();
  fd.append('file', currentBlob, 'query.jpg');

  try{
    const res = await fetch('/identify', { method:'POST', body: fd });
    const j = await res.json();
    if (!res.ok || j.error) throw new Error(j.error || 'Identification failed');

    // --- gates from server ---
    if (j.no_plant) {
      // store minimal payload so feedback can still be sent
      lastIdentifyPayload = { model: j.model || 'unknown', best: null, alternatives: [] };

      resultWrap.classList.remove('hidden');
      bestLabelEl.textContent = 'No plant detected';
      bestScoreEl.textContent = '—';
      confBar.style.width = '0%';
      altChips.innerHTML = '';
      factsHtml.innerHTML = `<p>${sanitize(j.message || 'No plant-like object was detected.')}</p>`;

      // show feedback first so a later error can’t block it
      showFeedbackUI({ openCorrection: true });

      // safe badge update
      setSourceBadge('local');

      showToast(j.message || 'No plant detected');
      return;
    }

    if (j.low_confidence) {
      lastIdentifyPayload = j;
      const best = j.best || {};
      resultWrap.classList.remove('hidden');
      bestLabelEl.textContent = best.label || 'Unknown';
      bestScoreEl.textContent = best.score != null ? `${(best.score*100).toFixed(1)}%` : '—';
      confBar.style.width = best.score != null ? `${Math.max(2, best.score*100)}%` : '2%';
      altChips.innerHTML = '';
      (j.alternatives || []).slice(1).forEach(a=>{
        const span = document.createElement('span');
        span.className = 'chip';
        span.textContent = `${a.label} ${(a.score*100).toFixed(0)}%`;
        altChips.appendChild(span);
      });
      factsHtml.innerHTML = `<p>${sanitize(j.message || 'Low-confidence ID. Try another photo.')}</p>`;

      // show feedback before badge
      showFeedbackUI({ openCorrection: true });

      setSourceBadge(j.model || 'local');
      showToast(j.message || 'Low-confidence ID');
      return;
    }

    // --- normal success path ---
    lastIdentifyPayload = j;
    const best = j.best || {};
    resultWrap.classList.remove('hidden');
    bestLabelEl.textContent = best.label || 'Unknown';
    bestScoreEl.textContent = best.score != null ? `${(best.score*100).toFixed(1)}%` : '—';
    confBar.style.width = best.score != null ? `${Math.max(2, best.score*100)}%` : '2%';

    altChips.innerHTML = '';
    (j.alternatives || []).slice(1).forEach(a=>{
      const span = document.createElement('span');
      span.className = 'chip';
      span.textContent = `${a.label} ${(a.score*100).toFixed(0)}%`;
      altChips.appendChild(span);
    });

    factsHtml.innerHTML = sanitize(j.facts_html || '<p>No facts available.</p>');

    // show feedback before badge
    showFeedbackUI({ openCorrection: false });

    setSourceBadge(j.model || 'local');
    explainBtn.disabled = false;
    showToast("Identified ✓");
  }catch(err){
    showToast(err.message, true);
  }finally{
    loading.classList.add('hidden');
    identifyBtn.disabled = false;
  }
});

// -------- Explain (Grad-CAM / EigenCAM) --------
explainBtn.addEventListener('click', async ()=>{
  if (!currentBlob) return showToast("Choose or capture an image first", true);

  explainBtn.disabled = true;
  loading.classList.remove('hidden');

  try{
    const fd = new FormData();
    fd.append('file', currentBlob, 'query.jpg');

    const method = camMethodSel ? camMethodSel.value : 'eigen';
    const res = await fetch(`/explain?method=${encodeURIComponent(method)}`, { method:'POST', body: fd });
    if (!res.ok) {
      const j = await res.json().catch(()=>({error:'Explain failed'}));
      throw new Error(j.error || 'Explain failed');
    }
    const blob = await res.blob();

    lastOverlayImg = new Image();
    lastOverlayImg.onload = () => {
      drawCamWithOpacity();
      explainArea.classList.remove('hidden');
      showToast(`Explanation: ${method} ✓`);
    };
    lastOverlayImg.src = URL.createObjectURL(blob);
  }catch(err){
    showToast(err.message, true);
  }finally{
    loading.classList.add('hidden');
    explainBtn.disabled = false; // re-enable
  }
});

// Live opacity updates
if (opacitySlider && opacityVal) {
  opacitySlider.addEventListener('input', () => {
    opacityVal.textContent = `${opacitySlider.value}%`;
    drawCamWithOpacity();
  });
}

// Download canvas as PNG
if (downloadCam) {
  downloadCam.addEventListener('click', ()=>{
    const c = document.getElementById('cam-canvas');
    if (!c) return;
    const url = c.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url; a.download = 'explanation.png';
    document.body.appendChild(a);
    a.click();
    a.remove();
  });
}

// -------- Feedback wiring --------
async function sendFeedback({ verdict, true_label=null, notes=null }){
  if (!currentBlob || !lastIdentifyPayload) {
    showToast('Nothing to send. Try identifying again.', true);
    return;
  }

  const fd = new FormData();
  fd.append('file', currentBlob, 'feedback_image.jpg');

  const meta = {
    verdict,                        // 'correct' | 'wrong'
    true_label: true_label || null, // optional
    notes: notes || null,           // optional
    model: lastIdentifyPayload.model || 'unknown',
    best: lastIdentifyPayload.best || null,
    alternatives: lastIdentifyPayload.alternatives || [],
    time: new Date().toISOString(),
    app_version: 'greenmind-1.0.0'
  };
  fd.append('meta', JSON.stringify(meta));

  try{
    const res = await fetch('/feedback', { method:'POST', body: fd });
    const j = await res.json().catch(()=> ({}));
    if (!res.ok) throw new Error(j.error || 'Feedback failed');

    if (fbToast) {
      fbToast.style.display = 'block';
      fbToast.textContent = 'Thanks! Your feedback was saved.';
    }
    showToast('Feedback saved ✓');
    if (fbCorrection) fbCorrection.classList.add('hidden');
  }catch(err){
    showToast(err.message, true);
  }
}

if (fbCorrectBtn) {
  fbCorrectBtn.addEventListener('click', async ()=>{
    await sendFeedback({ verdict: 'correct' });
  });
}
if (fbWrongBtn) {
  fbWrongBtn.addEventListener('click', ()=>{
    if (fbCorrection) fbCorrection.classList.remove('hidden');
  });
}
if (fbSubmit) {
  fbSubmit.addEventListener('click', async ()=>{
    const trueLabel = fbTrueLabel ? fbTrueLabel.value.trim() : '';
    const notes = fbNotes ? fbNotes.value.trim() : '';
    await sendFeedback({ verdict: 'wrong', true_label: trueLabel, notes });
  });
}

// -------- Camera buttons --------
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);

snapBtn.addEventListener('click', async ()=>{
  try{
    const blob = await captureFrameToBlob();
    currentBlob = blob;
    fileToDataURL(blob).then(url=>{
      previewImg.src = url;
      preview.classList.remove('hidden');
      resultWrap.classList.add('hidden');
      explainArea.classList.add('hidden');
      explainBtn.disabled = false; // allow Explain after capture
      lastOverlayImg = null;
      hideFeedbackUI();
      showToast('Captured frame ✓');
    });
  }catch(err){
    showToast(err.message, true);
  }
});

identifyFromCamBtn.addEventListener('click', async ()=>{
  try{
    const blob = await captureFrameToBlob();
    currentBlob = blob;
    identifyBtn.click();
  }catch(err){
    showToast(err.message, true);
  }
});

// Stop camera when navigating away
window.addEventListener('beforeunload', stopCamera);

// -------- Clear --------
clearBtn.addEventListener('click', ()=>{
  fileInput.value = '';
  currentBlob = null;
  preview.classList.add('hidden');
  resultWrap.classList.add('hidden');
  explainArea.classList.add('hidden');
  explainBtn.disabled = true;
  lastOverlayImg = null;
  hideFeedbackUI();
});
