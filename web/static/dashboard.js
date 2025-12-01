// /static/dashboard.js
// Modern grouped dashboard + animated weather background
(() => {
  const groupsContainer = document.getElementById('groups-container');
  const groupsEmpty = document.getElementById('groups-empty');
  const plantCount = document.getElementById('plant-count');
  const statGroups = document.getElementById('stat-groups');
  const statTotal = document.getElementById('stat-total');
  const statHealth = document.getElementById('stat-health');

  // detail area elements — may be missing; guard all uses
  const detailEmpty = document.getElementById('detail-empty');
  const detailView = document.getElementById('detail-view');
  const detailName = document.getElementById('detail-name');
  const detailDesc = document.getElementById('detail-desc');
  const detailImg = document.getElementById('detail-img');
  const healthSummaryBody = document.getElementById('health-summary-body');
  const historyList = document.getElementById('history-list');
  const galleryBlock = document.getElementById('gallery-block');
  const galleryThumbs = document.getElementById('gallery-thumbs');

  const weatherCard = document.getElementById('weather-card');
  const weatherCanvas = document.getElementById('weather-canvas');
  const weatherTemp = document.getElementById('weather-temp');
  const weatherCond = document.getElementById('weather-cond');
  const weatherMeta = document.getElementById('weather-meta');
  const weatherLocation = document.getElementById('weather-location');
  const weatherForecast = document.getElementById('weather-forecast');

  const precautionsBlock = document.getElementById('precautions-block');
  const precautionsBody = document.getElementById('precautions-body');

  const btnAddPlant = document.getElementById('btnAddPlant');
  const btnRunCheck = document.getElementById('btnRunCheck');
  const btnFetchPrecautions = document.getElementById('btnFetchPrecautions');

  // weather-note (right column)
  const noteBlock = document.getElementById("weather-note");
  const noteBody  = document.getElementById("weather-note-body");

  let plants = [];
  let selectedPlant = null;
  let geoCache = null;

  // ---------- utilities ----------
  function el(tag, cls, html){ const e = document.createElement(tag); if(cls) e.className = cls; if(html!=null) e.innerHTML = html; return e; }
  function escapeHtml(s=''){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
  window.addEventListener("plant-saved", () => {
      loadPlants();  // reload dashboard groups & history
  });
async function loadFullHealthResult(plant_id) {
    const box = document.getElementById("detail-health-full");
    if (!box) return;
    box.innerHTML = "Loading full health details...";

    try {
        const res = await fetch(`/api/get_latest_health?plant_id=${plant_id}`);
        const data = await res.json();

        if (!data || data.error) {
            box.innerHTML = `<div class="muted">No detailed health result found.</div>`;
            return;
        }

        const status = data.status || "Unknown";
        const score = data.score || 0;
        const diseases = (data.diseases || []).map(d => {
            if (typeof d === "string") return `<span class="chip">${d}</span>`;
            if (d.name) return `<span class="chip">${d.name}</span>`;
            if (d.label) return `<span class="chip">${d.label}</span>`;
            return "";
        }).join("");

        box.innerHTML = `
            <div class="topline">
                <h3 class="match-title">
                    Status: ${status}
                    <span class="badge">${score}%</span>
                </h3>
                <div class="meter"><div class="meter-fill" style="width:${score}%"></div></div>
            </div>

            <div class="alts">
                <span class="alts-title">Likely diseases:</span>
                <div class="chips">${diseases || "None detected"}</div>
            </div>

            <div class="card-sub" style="margin-top:14px;">
                <h3><i class="fa-solid fa-book-medical"></i> Care Tips</h3>
                <div class="facts-body">${data.care_html || "No tips available"}</div>
            </div>
        `;
    }
    catch (err) {
        console.error(err);
        box.innerHTML = "<div class='muted'>Error loading health details.</div>";
    }
}

  // ---------- fetch and group ----------
  async function loadPlants(){
    if (!groupsContainer) return;
    groupsContainer.innerHTML = '';
    if (groupsEmpty) { groupsEmpty.style.display = ''; groupsEmpty.innerText = 'Loading your plants...'; }
    try {
      const res = await fetch('/api/myplants');
      const j = await res.json();
      plants = j.plants || [];
      const count = j.count || plants.length;
      if (plantCount) plantCount.textContent = count;
      if (statTotal) statTotal.textContent = count;
      // compute avg health
      const avg = computeAvgHealth(plants);
      if (statHealth) statHealth.textContent = avg ? (Math.round(avg) + '%') : '--%';

      // group by: common_group (preferred), species, type, then name first word
      const groups = {};
      plants.forEach(p=>{
        const key = (p.group && p.group.trim()) || (p.common_group && p.common_group.trim()) || (p.species && p.species.trim()) || (p.type && p.type.trim()) || (p.name && p.name.split(' ')[0]) || 'Unknown';
        const normKey = String(key).trim() || 'Unknown';
        if(!groups[normKey]) groups[normKey] = [];
        groups[normKey].push(p);
      });

      renderGroups(groups);
    } catch (e){
      console.error('loadPlants', e);
      if (groupsEmpty) groupsEmpty.textContent = 'Could not load plants.';
    }
  }

  function computeAvgHealth(plantsArr){
    let total=0,count=0;
    (plantsArr||[]).forEach(p=>{
      // prefer last_health.confidence (0..1), else fallback to numeric fields
      if (p.last_health && typeof p.last_health.confidence === 'number') {
        total += (p.last_health.confidence*100);
        count++;
      } else if (typeof p.last_health_score === 'number') { total += p.last_health_score; count++; }
      else if (typeof p.health_score === 'number') { total += p.health_score; count++; }
    });
    return count ? (total/count) : null;
  }

  function renderGroups(groups){
    if (!groupsContainer) return;
    groupsContainer.innerHTML = '';
    const keys = Object.keys(groups).sort((a,b)=> groups[b].length - groups[a].length);
    if (!keys.length) {
      if (groupsEmpty) {
        groupsEmpty.style.display = '';
        groupsEmpty.innerText = 'No plants yet — try Identify or Add plant.';
      }
      return;
    }
    if (statGroups) statGroups.textContent = keys.length;

    keys.forEach(name=>{
      const items = groups[name];
      const groupCard = el('div','group-card');
      // header
      const head = el('div','group-head');
      const titleBlock = el('div','group-title');
      const icon = el('div','group-icon', `<i class="fa-solid fa-seedling"></i>`);
      const titleText = el('div','', `<div style="font-weight:700">${escapeHtml(name)}</div><div class="group-meta">${items.length} plants</div>`);
      titleBlock.appendChild(icon); titleBlock.appendChild(titleText);

      const chevron = el('div','group-meta','<i class="fa-solid fa-chevron-down"></i>');
      head.appendChild(titleBlock); head.appendChild(chevron);
      groupCard.appendChild(head);

      const body = el('div','group-body');
      // populate plants
      items.forEach(p=>{
        const item = el('div','plant-item');
        item.dataset.pid = p.id;
        const thumb = el('div','plant-thumb');
        const img = document.createElement('img');
        img.src = p.last_image ? (p.last_image.startsWith('/') ? p.last_image : '/' + p.last_image) : '/static/icons/icon-192.png';
        img.loading = 'lazy';
        thumb.appendChild(img);

        const info = el('div','plant-info');
        const title = el('div','plant-title', escapeHtml(p.name || 'Untitled'));
        const sub = el('div','plant-sub', `<span class="muted">${escapeHtml(p.location || p.place || '')}</span>`);
        const healthLine = el('div','plant-health');
        const meter = el('div','health-meter');
        const inner = el('i'); // width controlled below
        meter.appendChild(inner);
        const healthPct = Math.round((p.last_health && p.last_health.confidence ? p.last_health.confidence*100 : (p.last_health_score||p.health_score||0))) || 0;
        const healthVal = el('div','', `<strong>${healthPct}%</strong>`);
        healthLine.appendChild(meter);
        healthLine.appendChild(healthVal);

        info.appendChild(title); info.appendChild(sub); info.appendChild(healthLine);

        item.appendChild(thumb); item.appendChild(info);

        // clicking item selects and opens details
        item.addEventListener('click', ()=> {
          selectPlant(p.id);
      });

        // animate meter width
        setTimeout(()=> {
          inner.style.width = Math.max(2, Math.min(100, healthPct)) + '%';
        }, 80);

        body.appendChild(item);
      });

      // toggle behavior
      head.addEventListener('click', ()=>{
        const open = body.style.display === 'flex';
        document.querySelectorAll('.group-body').forEach(el=>el.style.display='none');
        document.querySelectorAll('.group-head .fa-chevron-down').forEach(ic=>ic.style.transform='rotate(0deg)');
        if (!open) {
          body.style.display = 'flex';
          body.style.flexDirection = 'column';
          try { chevron.querySelector('i').style.transform = 'rotate(180deg)'; } catch(e){}
        } else {
          body.style.display = 'none';
        }
      });

      groupCard.appendChild(body);
      groupsContainer.appendChild(groupCard);
      // open first group by default
      if (groupsContainer.children.length === 1) {
        head.click();
      }
    });
  }

  // ---------- selection details ----------
  async function selectPlant(id){
    selectedPlant = plants.find(p=> String(p.id) === String(id));
    if (!selectedPlant) return;

    // show/hide detail area if present
    if (detailEmpty) detailEmpty.style.display = 'none';
    if (detailView) detailView.classList.remove('hidden');

    if (detailName) detailName.textContent = selectedPlant.name || 'Untitled';
    if (detailDesc) detailDesc.textContent = selectedPlant.description || (selectedPlant.species || '');
    if (detailImg) detailImg.src = selectedPlant.last_image ? (selectedPlant.last_image.startsWith('/') ? selectedPlant.last_image : '/' + selectedPlant.last_image) : '/static/icons/icon-192.png';

    // meta
    const metaLine = document.getElementById('detail-meta-line');
    if (metaLine) metaLine.textContent = `Group: ${selectedPlant.group || selectedPlant.common_group || selectedPlant.species || 'Unknown'} • Added: ${selectedPlant.added_at ? new Date(selectedPlant.added_at).toLocaleDateString() : '—'}`;

    // health summary (uses last_health payload if exists)
    if (healthSummaryBody) {
      if (selectedPlant.last_health) {
        // last_health might be an object or a parsed payload depending on backend; handle both
        const lh = selectedPlant.last_health;
        const status = lh.status || (lh.payload && lh.payload.status) || (lh.raw && lh.raw.status) || 'unknown';
        const confidence = (typeof lh.confidence === 'number') ? Math.round(lh.confidence*100) : (typeof lh.payload === 'object' && typeof lh.payload.confidence === 'number' ? Math.round(lh.payload.confidence*100) : null);
        healthSummaryBody.innerHTML = `<div><strong>Status:</strong> ${escapeHtml(status)} ${confidence!=null ? `• Confidence: ${confidence}%` : ''}</div>`;
      } else {
        healthSummaryBody.innerHTML = `<div class="muted">No health scans yet.</div>`;
      }
    }

    // history: use selectedPlant.history if provided by /api/myplants, else fetch from API
    try {
      await populateHistory(selectedPlant.id, selectedPlant.history || null);
    } catch (err) {
      console.warn('history populate fail', err);
      if (historyList) historyList.innerHTML = '<div class="muted">Could not load history.</div>';
    }

    // gallery
    if (galleryThumbs) galleryThumbs.innerHTML = '';
    if (galleryBlock) galleryBlock.style.display = 'none';
    await loadGallery(selectedPlant.id);
    // Load FULL health result card too
    loadFullHealthResult(selectedPlant.id);

    // weather & precautions
    fetchWeatherAndPrecautions(selectedPlant.id);
  }

  async function populateHistory(plant_id, providedHistory){
    // If providedHistory is an array, use it. Otherwise fetch from API: /api/plant/history (we don't have that endpoint; use /api/myplants data or /api/plant/photos)
    if (!historyList) return;
    historyList.innerHTML = '';
    let rows = providedHistory;
    if (!Array.isArray(rows)) {
      // fallback: try to fetch the plant health rows via API endpoint (if available)
      try {
        const res = await fetch(`/api/myplants`);
        if (res.ok) {
          const j = await res.json();
          const p = (j.plants || []).find(x => String(x.id) === String(plant_id));
          rows = p ? (p.history || []) : [];
        } else {
          rows = [];
        }
      } catch (e) {
        rows = [];
      }
    }

    if (!rows || rows.length === 0) {
      historyList.innerHTML = '<div class="muted">No past scans</div>';
      return;
    }

    // Build list UI: show time, status, confidence and a small thumbnail if present
    rows.forEach(r => {
      // r may be { id, time, payload } (from your api_myplants) or a raw DB row; handle flexibly
      const payload = (r.payload && typeof r.payload === 'object') ? r.payload : (typeof r.payload === 'string' ? tryParseJSON(r.payload) : (r.payload || {}));
      const t = r.time || r.created_at || r.ts || null;
      const timeText = t ? (new Date(t).toLocaleString()) : '';
      const status = payload.status || payload.health_status || payload.result || 'unknown';
      const conf = (typeof payload.confidence === 'number') ? Math.round(payload.confidence*100) : (typeof payload.confidence === 'string' && !isNaN(payload.confidence) ? Math.round(Number(payload.confidence)*100) : (payload.confidence_pct || payload.confidence_percent || null));
      const diseases = (payload.diseases && Array.isArray(payload.diseases)) ? payload.diseases.map(d => (typeof d === 'string' ? d : (d.name || d.id))).join(', ') : '';

      const row = el('div','history-row');
      const left = el('div','history-left');
      const info = el('div','history-info', `<div class="history-time"><strong>${escapeHtml(timeText)}</strong></div>
        <div class="muted small history-meta">Status: ${escapeHtml(String(status))} ${conf!=null ? `• ${conf}%` : ''}${diseases ? ` • ${escapeHtml(diseases)}` : ''}</div>`);
      left.appendChild(info);

      // small thumbnail if payload.photo or payload.image exists
      const thumbContainer = el('div','history-thumb');
      let thumbSrc = payload.photo || payload.image || null;
      if (thumbSrc && typeof thumbSrc === 'string') {
        if (!thumbSrc.startsWith('/') && !thumbSrc.startsWith('http')) thumbSrc = '/' + thumbSrc;
        const img = document.createElement('img');
        img.src = thumbSrc;
        img.loading = 'lazy';
        img.addEventListener('click', ()=> openLightbox(thumbSrc));
        thumbContainer.appendChild(img);
      } else {
        // fallback: use plant last image small icon if available
        if (selectedPlant && selectedPlant.last_image) {
          const img = document.createElement('img');
          img.src = selectedPlant.last_image.startsWith('/') ? selectedPlant.last_image : '/' + selectedPlant.last_image;
          img.loading = 'lazy';
          img.style.opacity = '0.6';
          thumbContainer.appendChild(img);
        }
      }

      row.appendChild(thumbContainer);
      row.appendChild(left);
      historyList.appendChild(row);
    });
  }

  function tryParseJSON(s){
    try { return JSON.parse(s); } catch(e) { return {}; }
  }

  async function loadGallery(plant_id){
    if (!galleryThumbs) return;
    galleryThumbs.innerHTML = '';
    try {
      const res = await fetch(`/api/plant/photos?plant_id=${plant_id}`);
      if (!res.ok) return;
      const j = await res.json();
      if (!j.photos || !j.photos.length) return;
      if (galleryBlock) galleryBlock.style.display = 'block';
      j.photos.forEach(p=>{
        const img = document.createElement('img');
        let src = p.url || p.path || '';
        if (!src.startsWith('/') && !src.startsWith('http')) src = '/' + src;
        img.src = src;
        img.loading = 'lazy';
        img.addEventListener('click', ()=> openLightbox(src));
        galleryThumbs.appendChild(img);
      });
    } catch (e) { console.warn('gallery', e); }
  }

  // ---------- lightbox ----------
  function openLightbox(src){
    let lb = document.getElementById('global-lightbox');
    if (!lb) {
      const wrapper = document.createElement('div');
      wrapper.id = 'global-lightbox';
      wrapper.style = 'position:fixed; inset:0; display:flex; align-items:center; justify-content:center; background:rgba(0,0,0,0.85); z-index:9999;';
      const img = document.createElement('img');
      img.id = 'global-lightbox-img';
      img.style = 'max-width:90%; max-height:90%; border-radius:12px;';
      wrapper.appendChild(img);
      document.body.appendChild(wrapper);
      wrapper.addEventListener('click', ()=> { wrapper.remove(); });
    }
    const img = document.getElementById('global-lightbox-img');
    img.src = src;
  }

  // ---------- weather & precautions ----------
  function getCachedGeo(){
    if (geoCache && (Date.now() - geoCache.ts < 30000)) return Promise.resolve(geoCache.coords);
    return new Promise((resolve, reject)=>{
      if(!navigator.geolocation) return reject(new Error('No geo'));
      navigator.geolocation.getCurrentPosition(pos=>{
        geoCache = { coords: pos.coords, ts: Date.now() };
        resolve(pos.coords);
      }, err=> reject(err), { timeout: 8000 });
    });
  }

  async function fetchWeatherAndPrecautions(plant_id){
    let weather = null;
    try {
      const coords = await getCachedGeo().catch(()=>null);
      if (coords) {
        const res = await fetch(`/api/weather?lat=${coords.latitude}&lon=${coords.longitude}`);
        if (res.ok) weather = await res.json();
      }
    } catch(e){ console.warn('weather fetch', e); }

    if (weather && (weather.temp != null)) {
      updateWeatherCard(weather);
      updateWeatherNote(weather);
    } else {
      // hide if no weather
      if (weatherCard) weatherCard.style.display = 'none';
      if (noteBlock) noteBlock.classList.add('hidden');
    }

    // precautions call (with weather)
    try {
      const q = new URLSearchParams();
      q.set('plant_id', plant_id);
      if (weather) q.set('weather', JSON.stringify(weather));
      const res = await fetch('/api/plant_precautions?' + q.toString());
      if (res.ok) {
        const j = await res.json();
        if (j.html) {
          if (precautionsBlock && precautionsBody) {
            precautionsBlock.style.display = 'block';
            precautionsBody.innerHTML = j.html;
          }
          // also show global precaution area if present
          const globalPrecautions = document.getElementById('precautions-global-section');
          const globalPrecautionsBody = document.getElementById('precautions-global-body');
          if (globalPrecautions && globalPrecautionsBody) {
            globalPrecautions.classList.remove('hidden');
            globalPrecautionsBody.innerHTML = j.html;
          }
        } else {
          if (precautionsBlock) precautionsBlock.style.display = 'none';
          const globalPrecautions = document.getElementById('precautions-global-section');
          if (globalPrecautions) globalPrecautions.classList.add('hidden');
        }
      }
    } catch(e){ console.warn('precautions', e); }
  }

  function updateWeatherCard(w){
    if (!weatherCard) return;
    weatherCard.style.display = 'block';
    weatherTemp.textContent = Math.round(w.temp) + '°C';
    weatherCond.textContent = w.condition || (w.weather && w.weather[0] && w.weather[0].description) || '';
    weatherMeta.textContent = `Humidity: ${w.humidity ?? (w.main && w.main.humidity) ?? ''}% • Wind: ${w.wind_speed ?? (w.wind && w.wind.speed) ?? ''} km/h`;
    weatherLocation.textContent = w.name || 'Your area';

    // forecast chips (if present)
    weatherForecast.innerHTML = '';
    if (w.forecast && Array.isArray(w.forecast)) {
      w.forecast.slice(0,4).forEach(fc => {
        const chip = el('div','fc-chip', `<strong>${escapeHtml(fc.day || '')}</strong><div class="muted small">${Math.round(fc.temp)}°</div>`);
        weatherForecast.appendChild(chip);
      });
    }

    // animate canvas and gradient based on condition
    const cond = (w.condition || (w.weather && w.weather[0] && w.weather[0].main) || '').toLowerCase();
    let theme = 'clear';
    if (cond.includes('rain') || cond.includes('drizzle')) theme = 'rain';
    else if (cond.includes('cloud')) theme = 'cloud';
    else if (cond.includes('snow')) theme = 'snow';
    else if (cond.includes('storm') || cond.includes('thunder')) theme = 'storm';
    else if (cond.includes('fog') || cond.includes('mist') || cond.includes('haze')) theme = 'fog';
    else theme = 'clear';

    startWeatherAnimation(theme, w.icon || '');
  }

  // ---------- weather note ----------
  function updateWeatherNote(w){
    if (!noteBlock || !noteBody) return;
    const noteText = w.short_note || w.note || w.recommendation || null;
    if (noteText) {
      noteBody.innerHTML = noteText;
      noteBlock.classList.remove('hidden');
    } else {
      noteBlock.classList.add('hidden');
    }
  }

  // ---------- weather canvas animation ----------
  // Lightweight particle + gradient engine (no libs).
  let wAnim = { running:false, ctx:null, particles:[], w:0, h:0, theme:'clear' };
  function startWeatherAnimation(theme, iconCode){
    if (!weatherCanvas) return;
    const canvas = weatherCanvas;
    const ctx = canvas.getContext('2d');
    wAnim.ctx = ctx;
    wAnim.theme = theme;
    resizeCanvas();
    if (!wAnim.running) {
      wAnim.running = true;
      requestAnimationFrame(loop);
    }
    // set CSS gradient of parent based on theme
    const parent = weatherCard;
    if (!parent) return;
    parent.style.background = themeGradient(theme);
    // show small condition icon
    const iconEl = document.getElementById('weather-icon');
    if (iconCode && iconEl) {
      iconEl.innerHTML = `<img src="https://openweathermap.org/img/wn/${iconCode}@2x.png" style="width:54px; height:54px;">`;
    } else if (iconEl) {
      iconEl.innerHTML = '';
    }

    function resizeCanvas(){
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width));
      canvas.height = Math.max(1, Math.floor(rect.height));
      wAnim.w = canvas.width; wAnim.h = canvas.height;
      // seed particles
      wAnim.particles = [];
      const count = Math.max(8, Math.round((wAnim.w * wAnim.h) / 40000));
      for (let i=0;i<count;i++) wAnim.particles.push(makeParticle());
    }

    function makeParticle(){
      const p = { x: Math.random()*wAnim.w, y: Math.random()*wAnim.h, vx: (Math.random()-0.5) * (theme==='cloud'?0.3:1.2), vy: (Math.random()*0.6 + 0.2), size: Math.random()*2.5 + (theme==='snow'?1.7:0.8), alpha: Math.random()*0.6 + 0.15 };
      if (theme==='rain') { p.vy = Math.random()*6 + 4; p.vx = (Math.random()-0.2)*1.2; p.size = Math.random()*1.2 + 1; }
      if (theme==='cloud') { p.vy = Math.random()*0.6 + 0.1; p.size = Math.random()*8 + 18; p.alpha = 0.06; }
      if (theme==='snow') { p.vy = Math.random()*1 + 0.4; p.vx = (Math.random()-0.5)*0.4; p.size = Math.random()*3 + 2; }
      if (theme==='storm') { p.vy = Math.random()*8 + 4; p.size = Math.random()*2 + 1; }
      return p;
    }

    function loop(){
      const ctx = wAnim.ctx;
      if (!ctx) return;
      ctx.clearRect(0,0,wAnim.w,wAnim.h);
      // gradient overlay
      const g = ctx.createLinearGradient(0,0,0,wAnim.h);
      const colors = gradientColors(theme);
      g.addColorStop(0, colors[0]); g.addColorStop(1, colors[1]);
      ctx.fillStyle = g; ctx.fillRect(0,0,wAnim.w,wAnim.h);

      // draw particles (rain/snow/cloud mist)
      for (let i=0;i<wAnim.particles.length;i++){
        const p = wAnim.particles[i];
        ctx.globalAlpha = p.alpha;
        if (theme === 'rain' || theme === 'storm') {
          ctx.strokeStyle = 'rgba(255,255,255,0.55)';
          ctx.lineWidth = p.size;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(p.x + p.vx*2.5, p.y + p.vy*4.5);
          ctx.stroke();
        } else if (theme === 'snow') {
          ctx.fillStyle = 'rgba(255,255,255,0.9)';
          ctx.beginPath(); ctx.arc(p.x,p.y,p.size,0,Math.PI*2); ctx.fill();
        } else if (theme === 'cloud') {
          ctx.fillStyle = 'rgba(255,255,255,0.09)';
          ctx.beginPath(); ctx.ellipse(p.x,p.y,p.size*6,p.size*3,0,0,Math.PI*2); ctx.fill();
        } else { // clear - gentle shimmer
          ctx.fillStyle = 'rgba(255,255,255,0.04)';
          ctx.beginPath(); ctx.arc(p.x,p.y,p.size*2,0,Math.PI*2); ctx.fill();
        }

        // update
        p.x += p.vx; p.y += p.vy;
        if (p.y > wAnim.h + 20 || p.x < -50 || p.x > wAnim.w + 50) {
          // reset
          wAnim.particles[i] = makeParticle();
          wAnim.particles[i].x = Math.random()*wAnim.w;
          wAnim.particles[i].y = -10;
        }
      }

      ctx.globalAlpha = 1;
      requestAnimationFrame(loop);
    }

    // helpers for colors
    function gradientColors(t){
      switch(t){
        case 'rain': return ['rgba(45,118,255,0.95)','rgba(10,50,100,0.95)'];
        case 'storm': return ['rgba(20,35,60,0.95)','rgba(4,12,28,0.95)'];
        case 'cloud': return ['rgba(120,160,200,0.95)','rgba(40,80,130,0.95)'];
        case 'snow': return ['rgba(180,210,240,0.95)','rgba(100,140,180,0.95)'];
        case 'fog': return ['rgba(170,180,190,0.9)','rgba(100,110,120,0.9)'];
        default: return ['rgba(40,200,140,0.93)','rgba(30,130,255,0.95)']; // clear/sunny green-blue
      }
    }
    function themeGradient(t){
      const c = gradientColors(t);
      return `linear-gradient(135deg, ${c[0]}, ${c[1]})`;
    }

    // trigger resize on window
    function onResize(){ canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight; wAnim.w = canvas.width; wAnim.h = canvas.height; }
    window.removeEventListener('resize', onResize);
    window.addEventListener('resize', onResize);
    onResize();
  }

  // ---------- createPlantIcon flow ----------
  async function createPlantIcon(plantId, name){
    try {
      const res = await fetch('/api/create-plant-icon', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ plant_id: plantId, name })
      });
      if (!res.ok) throw new Error('icon create failed');
      const j = await res.json();
      if (j.ok && j.icon_url) {
        // find matching tile and insert small icon
        const tile = document.querySelector(`.plant-item[data-pid="${plantId}"]`);
        if (tile) {
          let ov = tile.querySelector('.plant-icon');
          if (!ov) {
            ov = el('div','plant-icon');
            ov.style.cssText = 'width:40px;height:40px;border-radius:8px;overflow:hidden;margin-left:8px;flex:0 0 40px;';
            tile.appendChild(ov);
          }
          ov.innerHTML = `<img src="${j.icon_url}" style="width:100%;height:100%;object-fit:cover">`;
        }
      }
    } catch(e){ console.warn('create icon', e); }
  }

  // ---------- buttons ----------
  if (btnAddPlant) {
    btnAddPlant.addEventListener('click', async ()=>{
      const name = prompt('Plant name (common or scientific)') || '';
      if (!name) return;
      try {
        const fd = new FormData();
        fd.append('plant_name', name);
        const res = await fetch('/api/save-scan', { method:'POST', body:fd });
        const j = await res.json();
        if (res.ok) {
          alert('Plant added!');
          await loadPlants();
        } else {
          alert('Could not add plant: ' + (j.error||'unknown'));
        }
      } catch(e){ alert('Add failed'); console.error(e); }
    });
  }
  if (btnRunCheck) {
    btnRunCheck.addEventListener('click', ()=> {
      if (!selectedPlant) return alert('Select a plant first.');
      // create icon (non-blocking)
      createPlantIcon(selectedPlant.id, selectedPlant.name).catch(()=>{});
      // go to health flow
      window.location.href = `/health-secondary?plant_id=${selectedPlant.id}`;
    });
  }
  if (btnFetchPrecautions) {
    btnFetchPrecautions.addEventListener('click', ()=> {
      if (!selectedPlant) return alert('Select a plant first.');
      fetchWeatherAndPrecautions(selectedPlant.id);
    });
  }

  // ---------- small helper: auto-select first plant on load ----------
  async function run(){
    await loadPlants();
    // fetch global weather so widget shows even before selecting plant
    try {
      const coords = await getCachedGeo().catch(()=>null);
      if (coords) {
        const res = await fetch(`/api/weather?lat=${coords.latitude}&lon=${coords.longitude}`);
        if (res.ok) {
          const w = await res.json();
          if (w) {
            updateWeatherCard(w);
            updateWeatherNote(w);
          }
        }
      }
    } catch(e){ /* ignore */ }

    // pick first plant if any — prefer the first group’s first plant
    if (plants && plants.length) {
      setTimeout(()=> {
        // prefer the first plant that appears in the grouped view (keeps UI consistent)
        const firstPlant = plants[0];
        if (firstPlant) selectPlant(firstPlant.id);
      }, 350);
    }
  }

  // start
  run();
})();
