# Webapp v2 — Status & Remaining Work

## Architecture

```
nwc_webapp_v2/
├── backend/                    # FastAPI (Python) — serves API + static files
│   ├── main.py                # FastAPI app entry point
│   ├── api/                   # REST API endpoints
│   │   ├── config.py          # GET /api/config/* (models, environment)
│   │   ├── data.py            # GET /api/data/* (SRI, predictions, calendar)
│   │   ├── jobs.py            # POST /api/jobs/* (submit, status, error logs)
│   │   ├── rendering.py       # GET /api/render/* (overlays, figures, GIFs)
│   │   ├── metrics.py         # POST /api/metrics/* (CSI, POD, FAR, FSS, fit diagrams)
│   │   └── realtime.py        # POST/GET /api/realtime/* (start, stop, status)
│   ├── services/
│   │   └── realtime.py        # Singleton background loop (HPC + local mock)
│   └── static/                # Built frontend (output of npm run build)
│
├── frontend/                   # Vue 3 + Leaflet.js + Tailwind CSS
│   ├── src/
│   │   ├── App.vue            # Layout: navbar + router-view
│   │   ├── router.js          # 3 routes: /realtime, /comparison, /metrics
│   │   ├── api.js             # Centralized API calls (fetch wrappers)
│   │   ├── stores/config.js   # Pinia store for app config + models
│   │   ├── components/
│   │   │   ├── RadarMap.vue   # Leaflet map with preloaded frames + radar markers
│   │   │   ├── ColorBar.vue   # Precipitation legend
│   │   │   └── DataTable.vue  # Color-coded metric tables
│   │   └── views/
│   │       ├── RealTimeView.vue
│   │       ├── ModelComparisonView.vue
│   │       └── MetricsView.vue
│   └── public/
│       ├── radar.png          # Radar station icon
│       └── favicon.png
```

## How to Run

```bash
# 1. Activate conda environment
conda activate nwc_webapp

# 2. Install the nwc_webapp package (backend imports from it)
cd /path/to/Nowcasting_webapp
pip install -e .

# 3. Install backend dependencies
pip install -r nwc_webapp_v2/backend/requirements.txt

# 4. Build frontend
cd nwc_webapp_v2/frontend
npm install
npm run build

# 5. Start the server (must be in backend/ directory)
cd ../backend
uvicorn main:app --port 8000

# Open http://localhost:8000
# API docs at http://localhost:8000/docs
```

For development with hot reload:
```bash
# Terminal 1: Backend
cd nwc_webapp_v2/backend && uvicorn main:app --port 8000 --reload

# Terminal 2: Frontend (Vite dev server, proxies /api/* to backend)
cd nwc_webapp_v2/frontend && npx vite --port 5173
```

---

## Completed Features

### Pages
- [x] **Real-Time Predictions** — Leaflet map, 25-frame animation (-60 to +60 min), model selector, start/stop backend loop, SRI polling, groundtruth preload without model, prediction check on RT start
- [x] **Model Comparison** — Side-by-side lead time comparison (12 rows), synchronized zoom, CSI tables per row, calendar date highlighting, timestamp availability panel
- [x] **Metrics Analysis** — 6 tabs (CSI, POD, FAR, FSS, RMSE, Fit Diagrams), per-model tables + Chart.js charts, overall performance tables, collapsible metric formulas, fit diagram rendering

### Map
- [x] Leaflet with 4 base maps (Dark, OSM, Satellite, Terrain)
- [x] Preloaded frame strategy (25 frames loaded in parallel, instant switching)
- [x] Radar station markers (26 stations) with hover tooltips
- [x] Icons invert color on dark/satellite maps
- [x] Geocoder search bar
- [x] Precipitation color bar legend

### Data & Availability
- [x] Calendar date highlighting (dates with predictions highlighted green)
- [x] Timestamp-level availability panel (per-model green/red badges)
- [x] Availability shows on date range selection (before model pick in Metrics)
- [x] Test model filtered from Comparison and Metrics pages

### Backend
- [x] Config API (models, environment, thresholds)
- [x] Data API (SRI, predictions check, calendar availability, day detail)
- [x] Jobs API (submit PBS/mock, status polling, error logs)
- [x] Rendering API (overlays, figures, GIF check/serve)
- [x] Metrics API (compute CSI/POD/FAR/FSS/RMSE, comparison, fit diagrams)
- [x] Real-time service (singleton, HPC loop with smart polling, local mock loop)
- [x] Pre-check predictions on RT start, process existing SRI immediately

### UI/UX
- [x] Date format DD/MM/YYYY (VueDatePicker v12 `formats` prop)
- [x] Export/download buttons (CSV per model, chart PNG, export all)
- [x] Dark-themed date pickers matching top bar
- [x] Notification toasts on Real-Time page

---

## Remaining Work

### Responsive Layout
- [ ] **Sidebar collapse on small screens** — Real-Time sidebar (w-72) should collapse or become a drawer on narrow viewports
- [ ] **Metrics/Comparison top bar wrapping** — Controls row wraps awkwardly on smaller screens, needs better flex/grid breakpoints
- [ ] **Table horizontal scroll** — DataTables with many columns (12 lead times) need horizontal scroll on small screens
- [ ] **Mobile touch support** — Timeline slider and map controls should work well on tablets

### HPC Deployment
- [ ] **Create `start_webapp.sh` launch script** — Single script to activate conda, cd to backend, start uvicorn
- [ ] **Test with real data** — Verify all tabs work with real SRI files and PBS job submission
- [ ] **Test real-time HPC loop** — Verify smart polling detects new SRI files and submits PBS jobs
- [ ] **Verify prediction paths** — Ensure cfg.yaml HPC paths are correct for all models

### Future Enhancements (Nice to Have)
- [ ] **WebSocket for real-time updates** — Replace 3s polling with WebSocket push (instant updates, lower overhead)
- [ ] **Admin/user roles** — Restrict who can start real-time, submit jobs, etc.
- [ ] **Global error toasts** — Unified notification system across all pages (RT page already has one)
- [ ] **Job queue visualization** — Show PBS queue status for all models in a dedicated panel
