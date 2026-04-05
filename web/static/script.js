// ==========================
// GreenMind script.js
// ==========================

// ---------- Shared DOM ----------
const yearEl = document.getElementById('year');
const toast  = document.getElementById('toast');
if (yearEl) yearEl.textContent = new Date().getFullYear();

// Safe no-op: avoid breaking if setSourceBadge is not defined
if (typeof window.setSourceBadge !== 'function') window.setSourceBadge = function(){};

// Service worker
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/static/sw.js").catch(console.error);
  });
}

// ---------- Helpers ----------
function showToast(msg, isError=false){
  if (!toast) return;
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
    const r = new FileReader();
    r.onload = ()=> resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}
function fmtPct(p){ return (Number(p)*100).toFixed(1)+'%'; }

// POST image helper (Health API)
async function postImage(url, file){
  const fd = new FormData();
  fd.append('image', file);
  const res = await fetch(url, { method:'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// =======================================================
// IDENTIFY PAGE (upload/dragdrop + camera + Grad/Eigen)
// =======================================================

// Explain controls (Identify page)
const camMethodSel  = document.getElementById('cam-method');
const opacitySlider = document.getElementById('opacity-slider');
const opacityVal    = document.getElementById('opacity-val');
const downloadCam   = document.getElementById('download-cam');

let lastOverlayImg = null; // PNG from /explain
let stream = null;

// Camera elements (shared by both pages)
const video  = document.getElementById('video');
const canvas = document.getElementById('frame-canvas');
const startBtn = document.getElementById('start-cam');
const stopBtn  = document.getElementById('stop-cam');
const snapBtn  = document.getElementById('snap');
const identifyFromCamBtn = document.getElementById('identify-from-cam');

// Identify page upload elements
const fileInput  = document.getElementById('file-input');
const browseBtn  = document.getElementById('browse-btn');
const dropzone   = document.getElementById('drop-zone');
const preview    = document.getElementById('preview');
const previewImg = document.getElementById('preview-img');

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

// Feedback
const fbBlock      = document.getElementById('feedback-block');
const fbCorrectBtn = document.getElementById('fb-correct');
const fbWrongBtn   = document.getElementById('fb-wrong');
const fbCorrection = document.getElementById('fb-correction');
const fbTrueLabel  = document.getElementById('fb-true-label');
const fbNotes      = document.getElementById('fb-notes');
const fbSubmit     = document.getElementById('fb-submit');
const fbToast      = document.getElementById('fb-toast');

let lastIdentifyPayload = null;
let currentBlob = null;

// ---- Identify: preview / drag & drop ----
function setPreview(file){
  currentBlob = file;
  fileToDataURL(file).then(url=>{
    if (!preview || !previewImg) return;
    previewImg.src = url;
    preview.classList.remove('hidden');
    resultWrap?.classList.add('hidden');
    explainArea?.classList.add('hidden');
    if (explainBtn) explainBtn.disabled = false;
    lastOverlayImg = null;
    hideFeedbackUI();
  }).catch(()=> showToast("Could not preview image", true));
}

if (browseBtn && fileInput) {
  browseBtn.addEventListener('click', ()=> fileInput.click());
  fileInput.addEventListener('change', e=>{
    const f = e.target.files?.[0];
    if (f) setPreview(f);
  });
}
if (dropzone) {
  ['dragenter', 'dragover'].forEach(evt => {
    dropzone.addEventListener(evt, e => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('hover');
    });
  });

  ['dragleave', 'drop'].forEach(evt => {
    dropzone.addEventListener(evt, e => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('hover');
    });
  });

  dropzone.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      setPreview(files[0]);
    }
  });

  // Prevent browser from opening the image in a new tab
  window.addEventListener('dragover', e => e.preventDefault());
  window.addEventListener('drop', e => e.preventDefault());
}

// if (dropzone) {
//   ['dragenter','dragover'].forEach(evt => dropzone.addEventListener(evt, e=>{
//     e.preventDefault(); e.stopPropagation(); dropzone.classList.add('hover');
//   }));
//   ['dragleave','drop'].forEach(evt => dropzone.addEventListener(evt, e=>{
//     e.preventDefault(); e.stopPropagation(); dropzone.classList.remove('hover');
//   }));
//   dropzone.addEventListener('drop', e=>{
//     const f = e.dataTransfer.files?.[0];
//     if (f) setPreview(f);
//   });
//   window.addEventListener('dragover', e => e.preventDefault());
//   window.addEventListener('drop', e => e.preventDefault());
//   window.addEventListener('dragleave', () => dropzone.classList.remove('hover'));
// }
// if (window.PLANT_ID && typeof healthPreviewImg !== 'undefined' && healthPreviewImg && healthPreviewImg.src) {
//   (async () => {
//     try {
//       const imageBase64 = healthPreviewImg.src;
//       await fetch('/api/plant/image', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({
//           plant_id: window.PLANT_ID,
//           image: imageBase64
//         })
//       });
//     } catch (err) {
//       console.warn('Failed to auto-upload plant image', err);
//     }
//   })();
// }
// ---- Identify: feedback helpers ----
function showFeedbackUI({ openCorrection = false } = {}) {
  if (!fbBlock) return;
  fbBlock.style.display = 'block';
  fbCorrection?.classList.toggle('hidden', !openCorrection);
  if (fbTrueLabel) fbTrueLabel.value = '';
  if (fbNotes) fbNotes.value = '';
  if (fbToast) fbToast.style.display = 'none';
}
function hideFeedbackUI() {
  if (!fbBlock) return;
  fbBlock.style.display = 'none';
  if (fbToast) fbToast.style.display = 'none';
  fbCorrection?.classList.add('hidden');
}

// ---- Identify: main action ----
if (identifyBtn) {
  identifyBtn.addEventListener('click', async ()=>{
    if (!currentBlob) return showToast("Choose an image first", true);
    identifyBtn.disabled = true;
    loading?.classList.remove('hidden');
    resultWrap?.classList.add('hidden');
    hideFeedbackUI();

    const fd = new FormData();
    fd.append('file', currentBlob, 'query.jpg');

    try{
      const res = await fetch('/identify', { method:'POST', body: fd });
      const j = await res.json();
      if (!res.ok || j.error) throw new Error(j.error || 'Identification failed');

      // gates
      if (j.no_plant) {
        lastIdentifyPayload = { model: j.model || 'unknown', best: null, alternatives: [] };
        resultWrap?.classList.remove('hidden');
        bestLabelEl && (bestLabelEl.textContent = 'No plant detected');
        bestScoreEl && (bestScoreEl.textContent = '—');
        confBar && (confBar.style.width = '0%');
        if (altChips) altChips.innerHTML = '';
        if (factsHtml) factsHtml.innerHTML = `<p>${sanitize(j.message || 'No plant-like object was detected.')}</p>`;
        showFeedbackUI({ openCorrection: true });
        setSourceBadge('local');
        showToast(j.message || 'No plant detected');
        return;
      }

      if (j.low_confidence) {
        lastIdentifyPayload = j;
        const best = j.best || {};
        resultWrap?.classList.remove('hidden');
        bestLabelEl && (bestLabelEl.textContent = best.label || 'Unknown');
        bestScoreEl && (bestScoreEl.textContent = best.score != null ? `${(best.score*100).toFixed(1)}%` : '—');
        confBar && (confBar.style.width = best.score != null ? `${Math.max(2, best.score*100)}%` : '2%');
        if (altChips) {
          altChips.innerHTML = '';
          (j.alternatives || []).slice(1).forEach(a=>{
            const span = document.createElement('span');
            span.className = 'chip';
            span.textContent = `${a.label} ${(a.score*100).toFixed(0)}%`;
            altChips.appendChild(span);
          });
        }
        if (factsHtml) factsHtml.innerHTML = `<p>${sanitize(j.message || 'Low-confidence ID. Try another photo.')}</p>`;
        showFeedbackUI({ openCorrection: true });
        setSourceBadge(j.model || 'local');
        showToast(j.message || 'Low-confidence ID');
        return;
      }

      // success
      lastIdentifyPayload = j;
      const best = j.best || {};
      resultWrap?.classList.remove('hidden');
      bestLabelEl && (bestLabelEl.textContent = best.label || 'Unknown');
      bestScoreEl && (bestScoreEl.textContent = best.score != null ? `${(best.score*100).toFixed(1)}%` : '—');
      confBar && (confBar.style.width = best.score != null ? `${Math.max(2, best.score*100)}%` : '2%');
      if (altChips) {
        altChips.innerHTML = '';
        (j.alternatives || []).slice(1).forEach(a=>{
          const span = document.createElement('span');
          span.className = 'chip';
          span.textContent = `${a.label} ${(a.score*100).toFixed(0)}%`;
          altChips.appendChild(span);
        });
      }
      if (factsHtml) factsHtml.innerHTML = sanitize(j.facts_html || '<p>No facts available.</p>');
      showFeedbackUI({ openCorrection: false });
      setSourceBadge(j.model || 'local');
      if (explainBtn) explainBtn.disabled = false;
      showToast("Identified ✓");
    }catch(err){
      showToast(err.message, true);
    }finally{
      loading?.classList.add('hidden');
      identifyBtn.disabled = false;
    }
  });
}

// ---- Identify: Explain (Grad/Eigen) ----
if (explainBtn) {
  explainBtn.addEventListener('click', async ()=>{
    if (!currentBlob) return showToast("Choose or capture an image first", true);
    explainBtn.disabled = true;
    loading?.classList.remove('hidden');

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
        explainArea?.classList.remove('hidden');
        showToast(`Explanation: ${method} ✓`);
      };
      lastOverlayImg.src = URL.createObjectURL(blob);
    }catch(err){
      showToast(err.message, true);
    }finally{
      loading?.classList.add('hidden');
      explainBtn.disabled = false;
    }
  });
}

function drawCamWithOpacity() {
  const canvasEl = document.getElementById('cam-canvas');
  if (!canvasEl || !lastOverlayImg || !previewImg) return;
  const ctx = canvasEl.getContext('2d');
  // Get original image size
  const originalWidth  = previewImg.naturalWidth  || lastOverlayImg.width  || 512;
  const originalHeight = previewImg.naturalHeight || lastOverlayImg.height || 512;
  // 🔥 CONTROL MAX WIDTH (important)
  const maxWidth = 600; // you can adjust (500–800 recommended)
  const scale = Math.min(1, maxWidth / originalWidth);
  const drawWidth  = originalWidth * scale;
  const drawHeight = originalHeight * scale;
  // Set canvas size (scaled)
  canvasEl.width  = drawWidth;
  canvasEl.height = drawHeight;
  // Clear canvas
  ctx.clearRect(0, 0, drawWidth, drawHeight);
  // Draw original image
  try {
    ctx.drawImage(previewImg, 0, 0, drawWidth, drawHeight);
  } catch (e) {
    console.warn("Preview image draw failed", e);
  }
  // Apply opacity safely
  const alpha = Math.max(0, Math.min(1, (Number(opacitySlider?.value || 55) / 100)));
  // Draw overlay (Grad-CAM)
  ctx.globalAlpha = alpha;
  ctx.drawImage(lastOverlayImg, 0, 0, drawWidth, drawHeight);
  // Reset alpha
  ctx.globalAlpha = 1;
}
if (opacitySlider && opacityVal) {
  opacitySlider.addEventListener('input', () => {
    opacityVal.textContent = `${opacitySlider.value}%`;
    drawCamWithOpacity();
  });
}
if (downloadCam) {
  downloadCam.addEventListener('click', ()=>{
    const c = document.getElementById('cam-canvas');
    if (!c) return;
    const a = document.createElement('a');
    a.href = c.toDataURL('image/png'); a.download = 'explanation.png';
    document.body.appendChild(a); a.click(); a.remove();
  });
}

// ---- Shared camera (works on both pages) ----
if (startBtn && snapBtn && identifyFromCamBtn && stopBtn) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    startBtn.disabled = true; snapBtn.disabled = true;
    identifyFromCamBtn.disabled = true; stopBtn.disabled = true;
    showToast('Camera not supported in this browser', true);
  }
}
async function startCamera(){
  try{
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode:{ ideal:'environment'} }, audio:false });
    if (video) { video.srcObject = stream; video.hidden = false; await video.play().catch(()=>{}); }
    if (snapBtn) snapBtn.disabled = false;
    if (identifyFromCamBtn) identifyFromCamBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = false;
    showToast('Camera started');
  }catch(err){ showToast('Camera error: ' + err.message, true); }
}
function stopCamera(){
  if(stream){ stream.getTracks().forEach(t=>t.stop()); stream = null; }
  if (video) { video.srcObject = null; video.hidden = true; }
  if (snapBtn) snapBtn.disabled = true;
  if (identifyFromCamBtn) identifyFromCamBtn.disabled = true;
  if (stopBtn) stopBtn.disabled = true;
  showToast('Camera stopped');
}
function captureFrameToBlob(){
  return new Promise((resolve, reject)=>{
    if(!video || video.readyState < 2) return reject(new Error('Camera not ready'));
    const w = video.videoWidth || 640, h = video.videoHeight || 480;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d'); ctx.drawImage(video, 0, 0, w, h);
    canvas.toBlob(b=> b ? resolve(b) : reject(new Error('Capture failed')), 'image/jpeg', 0.9);
  });
}
if (startBtn) startBtn.addEventListener('click', startCamera);
if (stopBtn)  stopBtn.addEventListener('click', stopCamera);
if (snapBtn) {
  snapBtn.addEventListener('click', async ()=>{
    try{
      const blob = await captureFrameToBlob();
      currentBlob = blob;
      fileToDataURL(blob).then(url=>{
        if (!preview || !previewImg) return;
        previewImg.src = url;
        preview.classList.remove('hidden');
        resultWrap?.classList.add('hidden');
        explainArea?.classList.add('hidden');
        if (explainBtn) explainBtn.disabled = false;
        lastOverlayImg = null; hideFeedbackUI();
        showToast('Captured frame ✓');
      });
    }catch(err){ showToast(err.message, true); }
  });
}
if (identifyFromCamBtn) {
  identifyFromCamBtn.addEventListener('click', async ()=>{
    try{
      const blob = await captureFrameToBlob();
      currentBlob = blob;
      // If Identify button exists, run Identify; else we're on Health → run health
      if (document.getElementById('identify-btn')) {
        document.getElementById('identify-btn').click();
      } else {
        runHealth(blob); // health page path
      }
    }catch(err){ showToast(err.message, true); }
  });
}
window.addEventListener('beforeunload', stopCamera);
// Clear button handler
if (clearBtn) {
  clearBtn.addEventListener('click', ()=>{
    if (fileInput) fileInput.value='';
    currentBlob=null;
    preview?.classList.add('hidden');
    resultWrap?.classList.add('hidden');
    explainArea?.classList.add('hidden');
    if (explainBtn) explainBtn.disabled = true;
    lastOverlayImg=null; 
    hideFeedbackUI();
  });
}
// =======================================================
// FEEDBACK HANDLERS
// =======================================================
if (fbCorrectBtn) {
  fbCorrectBtn.addEventListener('click', async () => {
    if (!lastIdentifyPayload || !currentBlob) {
      showToast('No identification to provide feedback on', true);
      return;
    }
    
    const meta = {
      verdict: 'correct',
      model: lastIdentifyPayload.model || 'unknown',
      best: lastIdentifyPayload.best,
      alternatives: lastIdentifyPayload.alternatives
    };
    
    const fd = new FormData();
    fd.append('file', currentBlob, 'feedback.jpg');
    fd.append('meta', JSON.stringify(meta));
    
    try {
      const res = await fetch('/feedback', { method: 'POST', body: fd });
      const j = await res.json();
      
      if (res.ok) {
        if (fbToast) {
          fbToast.textContent = '✓ Thanks for your feedback!';
          fbToast.style.display = 'block';
          fbToast.style.color = 'var(--accent)';
        }
        showToast('Feedback submitted ✓');
        
        // Hide feedback UI after 2 seconds
        setTimeout(() => {
          hideFeedbackUI();
        }, 2000);
      } else {
        throw new Error(j.error || 'Feedback submission failed');
      }
    } catch (err) {
      showToast('Feedback failed: ' + err.message, true);
      if (fbToast) {
        fbToast.textContent = '✗ Failed to submit feedback';
        fbToast.style.display = 'block';
        fbToast.style.color = 'var(--danger)';
      }
    }
  });
}

// "No, wrong" button - shows correction form
if (fbWrongBtn) {
  fbWrongBtn.addEventListener('click', () => {
    fbCorrection?.classList.remove('hidden');
    fbTrueLabel?.focus(); // Focus on input for better UX
  });
}

// Submit correction button
if (fbSubmit) {
  fbSubmit.addEventListener('click', async () => {
    const trueLabel = fbTrueLabel?.value?.trim();
    
    if (!trueLabel) {
      showToast('Please enter the correct plant name', true);
      fbTrueLabel?.focus();
      return;
    }
    
    if (!lastIdentifyPayload || !currentBlob) {
      showToast('No identification to provide feedback on', true);
      return;
    }
    
    const meta = {
      verdict: 'wrong',
      true_label: trueLabel,
      notes: fbNotes?.value?.trim() || '',
      model: lastIdentifyPayload.model || 'unknown',
      best: lastIdentifyPayload.best,
      alternatives: lastIdentifyPayload.alternatives
    };
    
    const fd = new FormData();
    fd.append('file', currentBlob, 'feedback.jpg');
    fd.append('meta', JSON.stringify(meta));
    
    // Disable button during submission
    fbSubmit.disabled = true;
    fbSubmit.textContent = 'Submitting...';
    
    try {
      const res = await fetch('/feedback', { method: 'POST', body: fd });
      const j = await res.json();
      
      if (res.ok) {
        if (fbToast) {
          fbToast.textContent = '✓ Feedback submitted! Thank you for helping improve the model.';
          fbToast.style.display = 'block';
          fbToast.style.color = 'var(--accent)';
        }
        showToast('Thank you for your correction! ✓');
        
        // Hide feedback UI after 2.5 seconds
        setTimeout(() => {
          hideFeedbackUI();
        }, 2500);
      } else {
        throw new Error(j.error || 'Feedback submission failed');
      }
    } catch (err) {
      showToast('Feedback failed: ' + err.message, true);
      if (fbToast) {
        fbToast.textContent = '✗ Failed to submit feedback. Please try again.';
        fbToast.style.display = 'block';
        fbToast.style.color = 'var(--danger)';
      }
    } finally {
      // Re-enable button
      fbSubmit.disabled = false;
      fbSubmit.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Submit feedback';
    }
  });
}

// Optional: Allow Enter key to submit correction
if (fbTrueLabel) {
  fbTrueLabel.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      fbSubmit?.click();
    }
  });
}


// =======================================================
// HEALTH SECONDARY PAGE (upload + camera + XAI grid)
// =======================================================

// Elements (exist only on /health-secondary)
const btnRunHealth   = document.getElementById('btnRunHealth');
const healthInput    = document.getElementById('healthImageInput') || document.getElementById('imageInput');
const healthLoading  = document.getElementById('health-loading');
const healthEmpty    = document.getElementById('health-empty');
const healthResult   = document.getElementById('health-result');
const btnHealthHeatmaps = document.getElementById('btnHealthHeatmaps');

// Health preview + explain UI elements (IDs must exist in health.html)
const healthPreview     = document.getElementById('health-preview');
const healthPreviewImg  = document.getElementById('health-preview-img');
const healthCamMethod   = document.getElementById('health-cam-method');
let   healthOpacity     = document.getElementById('health-opacity-slider');
const healthOpacityVal  = document.getElementById('health-opacity-val');
const healthDownloadCam = document.getElementById('health-download-cam');
const healthExplainArea = document.getElementById('health-explain-area');
const healthCamCanvas   = document.getElementById('health-cam-canvas');

const healthStatusEl = document.getElementById('health-status');
const healthScoreEl  = document.getElementById('health-score');
const healthMeter    = document.getElementById('health-meter');
const healthChips    = document.getElementById('health-chips');

const xaiBlock   = document.getElementById('xai-block');
const xaiGrid    = document.getElementById('xai-grid');
const healthXaiMethodSel = document.getElementById('health-xai-method');
const plantidBlk = document.getElementById('plantid-block');
const plantidHtml= document.getElementById('plantid-html');
const careHtml   = document.getElementById('care-html');

// Buttons
if (btnRunHealth) btnRunHealth.addEventListener('click', ()=> runHealth());

// Health runner (accepts optional blob from camera)
async function runHealth(blobOverride = null){
  let imageFile = null;
  if (blobOverride instanceof Blob) {
    imageFile = new File([blobOverride], 'camera.jpg', { type: 'image/jpeg' });
  } else if (healthInput && healthInput.files && healthInput.files[0]) {
    imageFile = healthInput.files[0];
  }
  if (!imageFile) {
    alert('Choose or capture an image first.');
    return;
  }

  healthLoading?.classList.remove('hidden');
  healthEmpty?.classList.add('hidden');
  healthResult?.classList.add('hidden');

  try {
    const data = await postImage('/api/health', imageFile);

    // Topline (status + meter)
    const conf = Number(data.confidence || 0);
    healthStatusEl && (healthStatusEl.textContent = (data.status || 'unknown').toUpperCase());
    healthScoreEl  && (healthScoreEl.textContent = fmtPct(conf));
    healthMeter    && (healthMeter.style.width = Math.max(2, conf*100) + '%');
    const plantNameEl = document.getElementById('health-plant');
    if (plantNameEl) plantNameEl.textContent = (data.plant && data.plant.name) ? data.plant.name : 'Unknown';

    // Disease chips
    if (healthChips) {
      healthChips.innerHTML = '';
      (data.diseases || []).slice(0,5).forEach(d=>{
        const name = typeof d === 'string' ? d : (d.name || 'unknown');
        const p = (typeof d === 'object' && d.prob != null) ? ` ${(d.prob*100).toFixed(0)}%` : '';
        const span = document.createElement('span');
        span.className = 'chip';
        span.textContent = `${name}${p}`;
        healthChips.appendChild(span);
      });
    }

    // XAI images (server-side health overlays)
    if (xaiGrid && xaiBlock) {
      xaiGrid.innerHTML = '';
      let any = false;

      (data.crops || []).forEach(c => {
        (c.xai || []).forEach(p => {
          any = true;
          const src = p.startsWith('runs/') ? `/${p}` : p;

          // infer method from filename
          const method = src.includes('eigencam') ? 'eigencam' : 'gradcam';
          const label  = method === 'eigencam' ? 'EigenCAM' : 'Grad-CAM';

          const card = document.createElement('div');
          card.className = 'card-sub';
          card.dataset.method = method; // used by filter

          card.innerHTML = `
            <img src="${src}" style="width:100%; border-radius:10px; display:block;">
            <div class="small muted" style="margin-top:4px;">${label}</div>
          `;
          xaiGrid.appendChild(card);
        });
      });

      xaiBlock.style.display = any ? 'block' : 'none';
      if (any) applyHealthXaiFilter();  // apply current filter
    }

    // Plant.id (optional)
    if (plantidBlk && plantidHtml) {
      if (data.external && data.external.plantid) {
        const ext = data.external.plantid;
        const healthy = String(ext.is_healthy);
        const list = Array.isArray(ext.diseases)
          ? ext.diseases.map(d => (d.name || d.id || 'unknown')).join(', ')
          : '—';
        plantidHtml.innerHTML = `<p><b>Is healthy:</b> ${healthy}<br><b>Diseases:</b> ${list}</p>`;
        plantidBlk.style.display = 'block';
      } else {
        plantidBlk.style.display = 'none';
      }
    }

// Care tips (optional endpoint)
if (careHtml) {
  careHtml.innerHTML = '<p class="muted">Generating care tips…</p>';
  try {
    const careRes = await fetch('/api/health/care', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: data.status, diseases: data.diseases || [] })
    });
    const careJ = await careRes.json().catch(()=>({}));
    careHtml.innerHTML = (careRes.ok && careJ.html) ? careJ.html : '<p>Care tips unavailable right now.</p>';
  } catch {
    careHtml.innerHTML = '<p>Care tips unavailable right now.</p>';
  }
}
if (window.PLANT_ID) {
    try {
        await fetch('/api/health/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                plant_id: window.PLANT_ID,
                payload: data   // the whole health result
            })
        });
        console.log("✓ Health auto-saved");
    } catch (err) {
        console.warn("Failed to auto-save health:", err);
    }
}

healthResult?.classList.remove('hidden');

/* ============================================================
   📌 SAVE HEALTH RESULT TO DATABASE
   This block runs AFTER health results are shown.
   ------------------------------------------------------------
   - Asks user if they want to save
   - Shows list of plants OR creates a new one
   - Sends image + health JSON to /api/save-scan
   ============================================================ */
try {
    let plant_id = window.PLANT_ID || null;

    if (!plant_id) {
        const autoName = data.plant?.name || data.plant_name || "My Plant";
        const createFD = new FormData();
        createFD.append("plant_name", autoName);
        const createRes = await fetch("/api/save-scan", { method: "POST", body: createFD });
        const createJ = await createRes.json();
        if (!createRes.ok) throw new Error(createJ.error || "Could not create plant");
        plant_id = createJ.plant_id;
        window.PLANT_ID = plant_id;
    }

    const fd = new FormData();
    fd.append("plant_id", plant_id);
    fd.append("plant_name", data.plant?.name || "");
    fd.append("species", data.plant?.name || "");   // <── FIX
    fd.append("health_json", JSON.stringify(data));
    if (imageFile) fd.append("image", imageFile, "health.jpg");


    const saveRes = await fetch("/api/save-scan", { method: "POST", body: fd });
    const saveJ = await saveRes.json();
    if (!saveRes.ok) throw new Error(saveJ.error || "Save failed");

    console.log("✓ Auto-saved health scan.");
    showToast("Health scan saved ✓");
    window.dispatchEvent(new Event("plant-saved"));
}
catch (err) {
    console.error(err);
    showToast("Save failed: " + err.message, true);
}
/* ============================================================ */

} catch (e) {
  const panel = document.getElementById('healthResult') || healthEmpty;
  if (panel) panel.innerHTML = 'Health analysis failed: ' + e.message;
} finally {
  healthLoading?.classList.add('hidden');
}
}


// -------------------- Health explain overlay (Identify-style) --------------------

// separate overlay image for health canvas so Identify overlay is not disturbed
let healthLastOverlayImg = null;

// draw overlay blended onto the health preview canvas
function drawHealthCamWithOpacity() {
  const canvasEl = healthCamCanvas || document.getElementById('health-cam-canvas');
  const overlay = healthLastOverlayImg;
  const preview = healthPreviewImg || previewImg; // fallback to identify preview if missing
  if (!canvasEl || !overlay || !preview) return;

  const w = (preview.naturalWidth || overlay.width) || 512;
  const h = (preview.naturalHeight || overlay.height) || 512;
  canvasEl.width = w; canvasEl.height = h;
  const ctx = canvasEl.getContext('2d');
  try { ctx.drawImage(preview, 0, 0, w, h); } catch (e) {}
  const alpha = Math.max(0, Math.min(1, (Number(healthOpacity?.value || 55) / 100)));
  ctx.globalAlpha = alpha;
  ctx.drawImage(overlay, 0, 0, w, h);
  ctx.globalAlpha = 1;
}

// wire opacity slider (use oninput to avoid duplicate listeners)
if (healthOpacity && healthOpacityVal) {
  healthOpacity.oninput = () => {
    healthOpacityVal.textContent = `${healthOpacity.value}%`;
    drawHealthCamWithOpacity();
  };
}

// download blended canvas
if (healthDownloadCam) {
  healthDownloadCam.addEventListener('click', () => {
    const c = healthCamCanvas || document.getElementById('health-cam-canvas');
    if (!c) return;
    const a = document.createElement('a');
    a.href = c.toDataURL('image/png');
    a.download = 'health_explanation.png';
    document.body.appendChild(a); a.click(); a.remove();
  });
}

// Heatmaps button: run health (if needed), then request single overlay and show blended view
if (btnHealthHeatmaps) {
  btnHealthHeatmaps.addEventListener('click', async () => {
    try {
      // ensure health summary ran first
      if (!healthResult || healthResult.classList.contains('hidden')) {
        await runHealth();
      }

      // pick blob to send: prefer camera capture (currentBlob) else health input file
      let blob = null;
      if (typeof currentBlob !== 'undefined' && currentBlob instanceof Blob) {
        blob = currentBlob;
      } else if (healthInput && healthInput.files && healthInput.files[0]) {
        blob = healthInput.files[0];
      } else {
        showToast('Choose or capture an image first', true);
        return;
      }

      // method selection: health UI > identify UI fallback > default eigen
      const method = (healthCamMethod && healthCamMethod.value) ? healthCamMethod.value : (camMethodSel ? camMethodSel.value : 'eigen');

      // call existing /explain endpoint (same as Identify)
      const fd = new FormData();
      fd.append('file', blob, 'query.jpg');
      const res = await fetch(`/explain?method=${encodeURIComponent(method)}`, { method: 'POST', body: fd });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ error: 'Explain failed' }));
        throw new Error(body.error || 'Explain failed');
      }

      const overlayBlob = await res.blob();

      // create overlay image and draw
      healthLastOverlayImg = new Image();
      healthLastOverlayImg.onload = () => {
        if (healthExplainArea) healthExplainArea.classList.remove('hidden');
        if (healthOpacityVal && healthOpacity) healthOpacityVal.textContent = `${healthOpacity.value}%`;
        drawHealthCamWithOpacity();
        showToast(`Explanation (${method}) ready`);
      };
      healthLastOverlayImg.src = URL.createObjectURL(overlayBlob);
    } catch (err) {
      showToast(err.message || 'Explain failed', true);
    }
  });
}

// Filter health heatmaps by method (keeps thumbnail grid fallback)
function applyHealthXaiFilter() {
  if (!xaiGrid || !healthXaiMethodSel) return;
  const val = healthXaiMethodSel.value || 'all';
  const cards = xaiGrid.querySelectorAll('div.card-sub');
  cards.forEach(card => {
    const m = card.dataset.method || 'gradcam';
    card.style.display = (val === 'all' || val === m) ? 'block' : 'none';
  });
}
if (healthXaiMethodSel) healthXaiMethodSel.addEventListener('change', applyHealthXaiFilter);

// legacy hook if present (backwards compat)
const btnHealthLegacy = document.getElementById('btnHealth');
if (btnHealthLegacy) btnHealthLegacy.addEventListener('click', ()=> runHealth());
