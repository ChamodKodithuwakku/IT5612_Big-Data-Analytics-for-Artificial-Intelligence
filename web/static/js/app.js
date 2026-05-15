/* ══════════════════════════════════════════════════
   TripGraph — Frontend Application
   ══════════════════════════════════════════════════ */

'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  cities:           [],
  categories:       [],
  budgetLabels:     {},
  activeTab:        'nl',
  selectedBudget:   '',
  selectedCats:     new Set(['Restaurants', 'Arts & Entertainment', 'Coffee & Tea']),
  lastResponse:     null,
  leafletMap:       null,
};

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await loadConfig();
  setupNavScroll();
  lucide.createIcons();
});

async function loadConfig() {
  try {
    const res  = await fetch('/api/config');
    const data = await res.json();

    state.cities       = data.cities       || [];
    state.categories   = data.categories   || [];
    state.budgetLabels = data.budget_labels || {};

    populateCitySelect();
    populateCategoriesGrid();
  } catch (err) {
    console.warn('Could not load config from server:', err);
  }
}

// ── Navigation ────────────────────────────────────────────────────────────────
function setupNavScroll() {
  const nav = document.querySelector('.hero-nav');
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 60);
  }, { passive: true });
}

function scrollToPlanner() {
  document.getElementById('planner').scrollIntoView({ behavior: 'smooth' });
}

// ── Form population ───────────────────────────────────────────────────────────
function populateCitySelect() {
  const sel = document.getElementById('city-select');
  sel.innerHTML = '';
  state.cities.forEach(city => {
    const opt   = document.createElement('option');
    opt.value   = city;
    opt.textContent = city;
    sel.appendChild(opt);
  });
}

function populateCategoriesGrid() {
  const grid = document.getElementById('categories-grid');
  grid.innerHTML = '';
  state.categories.forEach(cat => {
    const btn       = document.createElement('button');
    btn.className   = 'cat-btn' + (state.selectedCats.has(cat) ? ' active' : '');
    btn.dataset.cat = cat;
    btn.textContent = cat;
    btn.type        = 'button';
    btn.onclick     = () => toggleCat(cat, btn);
    grid.appendChild(btn);
  });
}

function toggleCat(cat, btn) {
  if (state.selectedCats.has(cat)) {
    state.selectedCats.delete(cat);
    btn.classList.remove('active');
  } else {
    state.selectedCats.add(cat);
    btn.classList.add('active');
  }
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function switchTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll('.tab').forEach(t =>
    t.classList.toggle('active', t.dataset.tab === tab)
  );
  document.querySelectorAll('.tab-pane').forEach(p =>
    p.classList.toggle('active', p.id === `tab-${tab}`)
  );
}

// ── Slider ────────────────────────────────────────────────────────────────────
function updateDaysLabel(input) {
  const v = parseInt(input.value);
  document.getElementById('days-value').textContent = `${v} day${v !== 1 ? 's' : ''}`;
}

// ── Budget ────────────────────────────────────────────────────────────────────
function selectBudget(btn) {
  document.querySelectorAll('.budget-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  state.selectedBudget = btn.dataset.budget;
}

// ── Example pills ─────────────────────────────────────────────────────────────
function fillExample(btn) {
  document.getElementById('nl-prompt').value = btn.textContent;
  document.getElementById('nl-prompt').focus();
  if (state.activeTab !== 'nl') switchTab('nl');
}

// ── Error display ─────────────────────────────────────────────────────────────
function showError(msg) {
  const banner = document.getElementById('error-banner');
  document.getElementById('error-text').textContent = msg;
  banner.hidden = false;
  banner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  lucide.createIcons();
}

function hideError() {
  document.getElementById('error-banner').hidden = true;
}

// ── Loading overlay ───────────────────────────────────────────────────────────
let _loadingTimer = null;
let _loadingAbort = null; // set by planTrip so Cancel can abort the fetch

function showLoading(msg) {
  document.getElementById('loading-msg').textContent = msg || 'Building your itinerary…';
  document.getElementById('loading-overlay').hidden = false;
  document.body.style.overflow = 'hidden';

  // Live elapsed timer
  const startTime = Date.now();
  const timerEl   = document.getElementById('loading-timer');
  const subEl     = document.getElementById('loading-sub');
  timerEl.textContent = '';
  subEl.textContent   = '';
  clearInterval(_loadingTimer);
  _loadingTimer = setInterval(() => {
    const secs = Math.floor((Date.now() - startTime) / 1000);
    timerEl.textContent = `${secs}s elapsed`;
    if (secs > 10)  subEl.textContent = 'Loading businesses from Neo4j…';
    if (secs > 25)  subEl.textContent = 'Running hybrid recommendations…';
    if (secs > 45)  subEl.textContent = 'Clustering stops and building itinerary…';
    if (secs > 90)  subEl.textContent = 'Taking longer than expected — check Neo4j connectivity.';
  }, 1000);
}

function hideLoading() {
  clearInterval(_loadingTimer);
  document.getElementById('loading-overlay').hidden = true;
  document.body.style.overflow = '';
}

function cancelLoading() {
  if (_loadingAbort) _loadingAbort.abort();
  hideLoading();
  document.getElementById('btn-plan').disabled = false;
}

// ── Plan trip ─────────────────────────────────────────────────────────────────
async function planTrip() {
  hideError();

  const payload = {};

  if (state.activeTab === 'nl') {
    const text = document.getElementById('nl-prompt').value.trim();
    if (!text) {
      showError('Please enter a travel description before planning.');
      return;
    }
    payload.text = text;
  } else {
    const city       = document.getElementById('city-select').value;
    const days       = parseInt(document.getElementById('days-slider').value);
    const categories = [...state.selectedCats];

    if (!city) {
      showError('Please select a destination city.');
      return;
    }
    if (categories.length === 0) {
      showError('Please select at least one interest category.');
      return;
    }

    payload.city       = city;
    payload.days       = days;
    payload.categories = categories;
    payload.budget     = state.selectedBudget || null;
  }

  const btn = document.getElementById('btn-plan');
  btn.disabled = true;
  showLoading('Connecting to Neo4j…');

  const controller = new AbortController();
  _loadingAbort    = controller;
  const timeoutId  = setTimeout(() => controller.abort(), 120_000); // 2-min hard timeout

  try {
    const res = await fetch('/api/plan', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
      signal:  controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errData.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    state.lastResponse = data;

    hideLoading();
    renderResults(data);
  } catch (err) {
    clearTimeout(timeoutId);
    hideLoading();
    if (err.name === 'AbortError') {
      showError('Request timed out after 2 minutes. Check that your Neo4j credentials are correct and the database is reachable.');
    } else {
      showError(err.message);
    }
  } finally {
    btn.disabled = false;
  }
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data) {
  const section = document.getElementById('results');
  section.hidden = false;

  renderCityBanner(data);
  renderStats(data.stats);
  renderIntentCard(data.intent_summary);
  renderItinerary(data.itinerary);
  renderMap(data.itinerary);
  renderSummary(data.itinerary);

  lucide.createIcons();

  setTimeout(() => {
    section.scrollIntoView({ behavior: 'smooth' });
  }, 150);
}

function renderCityBanner(data) {
  const budgetLabel = data.budget ? ` · ${data.budget} budget` : '';
  document.getElementById('results-title').textContent =
    `${data.city} — ${data.days}-Day Itinerary`;
  document.getElementById('results-subtitle').textContent =
    (data.categories || []).slice(0, 4).join(', ') + budgetLabel;
}

function renderStats(stats) {
  const total   = stats?.total_businesses ?? 0;
  const rating  = stats?.avg_rating       ?? 0;
  const reviews = stats?.total_reviews    ?? 0;

  document.getElementById('stats-bar').innerHTML = `
    <div class="stat-card">
      <span class="stat-icon">🏢</span>
      <span class="stat-number">${Number(total).toLocaleString()}</span>
      <span class="stat-label">Businesses in graph</span>
    </div>
    <div class="stat-card">
      <span class="stat-icon">⭐</span>
      <span class="stat-number">${Number(rating).toFixed(2)}</span>
      <span class="stat-label">Average rating</span>
    </div>
    <div class="stat-card">
      <span class="stat-icon">💬</span>
      <span class="stat-number">${Number(reviews).toLocaleString()}</span>
      <span class="stat-label">Total reviews</span>
    </div>
  `;
}

function renderIntentCard(summary) {
  const card = document.getElementById('intent-card');
  if (!summary) {
    card.hidden = true;
    return;
  }
  document.getElementById('intent-text').innerHTML = markdownToHtml(summary);
  card.hidden = false;
}

// ── Itinerary cards ───────────────────────────────────────────────────────────
function renderItinerary(itinerary) {
  const container = document.getElementById('days-container');
  container.innerHTML = '';

  for (const day of itinerary) {
    container.appendChild(buildDaySection(day));
  }
}

function buildDaySection(day) {
  const section = document.createElement('div');
  section.className = 'day-section expanded';

  // Header
  const header = document.createElement('div');
  header.className = 'day-header';
  header.innerHTML = `
    <span class="day-pin" style="background:${day.color}">Day ${day.day}</span>
    <span class="day-meta">
      ${day.places.length} stops
      <span class="sep">·</span>
      ~${day.total_km.toFixed(1)} km
      <span class="sep">·</span>
      ~${Math.round(day.total_min)} min travel
    </span>
    <span class="day-chevron">▾</span>
  `;
  header.addEventListener('click', () => section.classList.toggle('expanded'));

  // Body
  const body = document.createElement('div');
  body.className = 'day-body';

  for (let i = 0; i < day.places.length; i++) {
    body.appendChild(buildPlaceCard(day.places[i], day.color));
    if (i < day.legs.length) {
      body.appendChild(buildLegConnector(day.legs[i], day.color));
    }
  }

  section.appendChild(header);
  section.appendChild(body);
  return section;
}

function buildPlaceCard(place, color) {
  const card = document.createElement('div');
  card.className = 'place-card';
  card.style.setProperty('--card-color', color);

  const ratingNum = place.rating || 0;
  const full      = Math.round(ratingNum);
  const stars     = '★'.repeat(full) + '☆'.repeat(Math.max(0, 5 - full));

  const explanation = markdownToHtml(place.explanation || '');
  const cats = (place.categories || '').split(',').slice(0, 3).join(', ');

  card.innerHTML = `
    <div class="place-card-main">
      <div class="place-stop-num" style="background:${color}">${place.stop}</div>
      <div class="place-info">
        <div class="place-name" title="${escHtml(place.name)}">${escHtml(place.name)}</div>
        <div class="place-meta">
          <span class="stars">${stars}</span>
          <span class="rating">${ratingNum.toFixed(1)}</span>
          ${place.price ? `<span class="price-badge">${escHtml(place.price)}</span>` : ''}
          <span>💬 ${place.review_count.toLocaleString()}</span>
          <span class="score">🎯 ${place.score.toFixed(3)}</span>
        </div>
        <div class="place-cats">${escHtml(cats)}</div>
      </div>
    </div>
    <details class="place-why">
      <summary>Why this place?</summary>
      <p>${explanation}</p>
    </details>
  `;
  return card;
}

function buildLegConnector(leg, color) {
  const el = document.createElement('div');
  el.className = 'leg-connector';
  el.innerHTML = `
    <div class="leg-line" style="background:${color}"></div>
    <div class="leg-info">↓ ${leg.distance_km} km · ~${Math.round(leg.duration_min)} min drive</div>
  `;
  return el;
}

// ── Map ───────────────────────────────────────────────────────────────────────
function renderMap(itinerary) {
  // Destroy previous instance
  if (state.leafletMap) {
    state.leafletMap.remove();
    state.leafletMap = null;
  }

  const allCoords = [];
  for (const day of itinerary) {
    for (const p of day.places) {
      if (p.lat && p.lng) allCoords.push([p.lat, p.lng]);
    }
  }
  if (allCoords.length === 0) return;

  const centerLat = allCoords.reduce((s, c) => s + c[0], 0) / allCoords.length;
  const centerLng = allCoords.reduce((s, c) => s + c[1], 0) / allCoords.length;

  const map = L.map('map').setView([centerLat, centerLng], 13);
  state.leafletMap = map;

  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 19,
  }).addTo(map);

  for (const day of itinerary) {
    const dayCoords = [];

    for (let i = 0; i < day.places.length; i++) {
      const p = day.places[i];
      if (!p.lat || !p.lng) continue;

      dayCoords.push([p.lat, p.lng]);

      // Outer ring
      L.circleMarker([p.lat, p.lng], {
        radius:      11,
        color:       'white',
        weight:      2.5,
        fillColor:   day.color,
        fillOpacity: 0.92,
      })
        .bindPopup(buildPopupHtml(p, day))
        .addTo(map);

      // Stop number label
      const numIcon = L.divIcon({
        html: `<div style="
          background:${day.color};
          color:white;
          border-radius:50%;
          width:18px;height:18px;
          display:flex;align-items:center;justify-content:center;
          font-size:10px;font-weight:700;
          font-family:Inter,sans-serif;
          border:2px solid white;
          box-shadow:0 1px 4px rgba(0,0,0,.25);
        ">${i + 1}</div>`,
        className:  '',
        iconSize:   [18, 18],
        iconAnchor: [9, 9],
      });
      L.marker([p.lat, p.lng], { icon: numIcon, interactive: false }).addTo(map);
    }

    // Route polyline
    if (dayCoords.length > 1) {
      L.polyline(dayCoords, {
        color:     day.color,
        weight:    2.5,
        opacity:   0.65,
        dashArray: '6, 8',
      }).addTo(map);
    }
  }

  // Fit bounds with padding
  if (allCoords.length > 1) {
    map.fitBounds(L.latLngBounds(allCoords), { padding: [28, 28] });
  }
}

function buildPopupHtml(place, day) {
  const stars    = place.rating ? place.rating.toFixed(1) + '★' : '?';
  const reviews  = place.review_count.toLocaleString();
  const price    = place.price ? `&nbsp;·&nbsp;${escHtml(place.price)}` : '';
  return `
    <div style="font-family:Inter,sans-serif;min-width:170px">
      <b style="color:${day.color};font-size:.9rem">${escHtml(place.name)}</b><br>
      <span style="font-size:.82rem;color:#64748b">
        ${stars} &nbsp; 💬 ${reviews} reviews${price}
      </span><br>
      <span style="font-size:.78rem;color:#94a3b8">Day ${day.day} · Stop ${place.stop}</span>
    </div>
  `;
}

// ── Summary stats ─────────────────────────────────────────────────────────────
function renderSummary(itinerary) {
  const totalStops = itinerary.reduce((s, d) => s + d.places.length, 0);
  const totalKm    = itinerary.reduce((s, d) => s + d.total_km, 0);
  const totalMin   = itinerary.reduce((s, d) => s + d.total_min, 0);
  const days       = itinerary.length;

  document.getElementById('summary-bar').innerHTML = `
    <div class="summary-card">
      <span class="summary-number">${totalStops}</span>
      <span class="summary-label">Total stops</span>
    </div>
    <div class="summary-card">
      <span class="summary-number">${totalKm.toFixed(1)}</span>
      <span class="summary-label">km total</span>
    </div>
    <div class="summary-card">
      <span class="summary-number">~${Math.round(totalMin)}</span>
      <span class="summary-label">min travel</span>
    </div>
    <div class="summary-card">
      <span class="summary-number">${days}</span>
      <span class="summary-label">days planned</span>
    </div>
  `;
}

// ── Export ────────────────────────────────────────────────────────────────────
async function exportItinerary() {
  if (!state.lastResponse) return;

  try {
    const res = await fetch('/api/export', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        city:      state.lastResponse.city,
        days:      state.lastResponse.days,
        itinerary: state.lastResponse.itinerary,
      }),
    });

    if (!res.ok) throw new Error('Export failed');

    const text = await res.text();
    const blob = new Blob([text], { type: 'text/plain' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    const slug = state.lastResponse.city.toLowerCase().replace(/\s+/g, '_');
    a.href     = url;
    a.download = `tripgraph_${slug}_${state.lastResponse.days}days.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    showError('Could not export: ' + err.message);
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function markdownToHtml(text) {
  // Convert **bold** and basic newlines
  return String(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}
