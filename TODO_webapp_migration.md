# Webapp Migration Plan: Streamlit → FastAPI + Vue.js + Leaflet

## Overview

Migrate the nowcasting web application from Streamlit to a modern **FastAPI backend + Vue 3 frontend + Leaflet.js map** architecture. This will give us:
- Instant UI interactions (no page reruns)
- Smooth map with client-side rendering
- Proper background task handling
- Beautiful, responsive design
- Same deployment model (localhost + SSH tunnel)

## Architecture

```
nwc_webapp_v2/
├── backend/                    # FastAPI (Python) - serves API + static files
│   ├── main.py                # FastAPI app entry point
│   ├── config/                # Config system (reused from current)
│   ├── api/                   # REST API endpoints
│   │   ├── predictions.py     # GET/POST prediction data
│   │   ├── models.py          # GET available models, status
│   │   ├── jobs.py            # POST submit jobs, GET job status
│   │   ├── data.py            # GET radar data, groundtruth
│   │   ├── gifs.py            # GET/POST gif creation and retrieval
│   │   └── metrics.py         # GET/POST CSI, FSS computation
│   ├── services/              # Business logic (mostly reused from current)
│   │   ├── prediction.py      # Prediction checking, loading
│   │   ├── job_manager.py     # Background job submission + monitoring
│   │   ├── gif_service.py     # GIF creation service
│   │   ├── metrics_service.py # CSI/FSS computation
│   │   ├── realtime.py        # Real-time file monitoring
│   │   └── mock_service.py    # Mock data for local dev
│   ├── ws/                    # WebSocket endpoints
│   │   ├── realtime.py        # Push real-time updates to clients
│   │   └── jobs.py            # Push job progress updates
│   ├── core/                  # Reused core modules
│   │   ├── data/              # Data loading (reused as-is)
│   │   ├── rendering/         # Figure generation (reused as-is)
│   │   ├── evaluation/        # Metrics (reused as-is)
│   │   ├── geo/               # Geo utilities (reused as-is)
│   │   ├── hpc/               # PBS integration (reused as-is)
│   │   └── mock/              # Mock data generation (reused as-is)
│   └── static/                # Built frontend files (served by FastAPI)
│
├── frontend/                   # Vue 3 + Leaflet.js + Tailwind CSS
│   ├── index.html
│   ├── src/
│   │   ├── App.vue            # Main app layout
│   │   ├── main.js            # Vue app entry
│   │   ├── router.js          # Client-side routing (tabs → routes)
│   │   ├── stores/            # Pinia stores (replaces session_state)
│   │   │   ├── app.js         # Global app state
│   │   │   ├── realtime.js    # Real-time prediction state
│   │   │   ├── predictions.js # Prediction data state
│   │   │   └── jobs.js        # Job status state
│   │   ├── components/        # Reusable UI components
│   │   │   ├── AppSidebar.vue
│   │   │   ├── RadarMap.vue   # Leaflet map with radar overlay
│   │   │   ├── StatusPanel.vue
│   │   │   ├── JobMonitor.vue
│   │   │   ├── DateTimePicker.vue
│   │   │   ├── ModelSelector.vue
│   │   │   └── GifViewer.vue
│   │   ├── views/             # Page views (replaces pages/)
│   │   │   ├── RealTimeView.vue
│   │   │   ├── NowcastingView.vue
│   │   │   ├── PredictionByDateView.vue
│   │   │   ├── ModelComparisonView.vue
│   │   │   └── MetricsView.vue
│   │   ├── composables/       # Shared logic
│   │   │   ├── useWebSocket.js
│   │   │   ├── useApi.js
│   │   │   └── useMap.js
│   │   └── assets/            # CSS, images, etc.
│   ├── package.json
│   └── vite.config.js
│
└── resources/                  # Shared resources (reused from current)
    ├── cfg/
    ├── legends/
    ├── mask/
    └── shapefiles/
```

## How to run (final goal)

```bash
# Terminal 1: Start backend
cd nwc_webapp_v2/backend
uvicorn main:app --port 8000

# Open browser: http://localhost:8000
# (Frontend is served as static files by FastAPI - no separate server needed)

# For development only (hot reload):
# Terminal 1: Backend with auto-reload
uvicorn main:app --port 8000 --reload
# Terminal 2: Frontend dev server with hot reload
cd frontend && npm run dev
```

---

## Module Reusability Analysis

### Fully Reusable (no Streamlit dependency) - copy as-is
- `config/config.py` - Config system with dataclasses
- `config/environment.py` - HPC vs local detection
- `config/constants.py` - Application constants
- `data/checking.py` - Prediction existence checks
- `data/dataset.py` - Dataset creation
- `data/groundtruth.py` - Groundtruth loading
- `data/predictions.py` - Prediction data loading
- `evaluation/metrics.py` - CSI, FSS calculations
- `evaluation/plots.py` - Metric visualization
- `geo/coordinates.py` - Coordinate transforms
- `geo/shapefiles.py` - Shapefile loading
- `geo/warping.py` - Projection utilities
- `rendering/colormaps.py` - Colormap configuration
- `rendering/figures.py` - Matplotlib figure generation
- `rendering/fit_diagram.py` - Fit diagram visualization
- `hpc/pbs.py` - PBS job submission (minor `st` import to remove)
- `hpc/jobs.py` - Job submission helpers
- `mock/generator.py` - Mock data generation
- `mock/predictions.py` - Mock prediction logic
- `mock/realtime.py` - Mock real-time service
- `logging_config.py` - Logging system
- `pages/csi_helpers.py` - CSI computation helpers

### Needs Streamlit Removal (logic reusable, UI code removed)
- `data/loaders.py` - Uses `st.session_state` for caching → remove st, use function params
- `rendering/gifs.py` - Uses `st.progress` → remove progress, return results
- `rendering/visualization.py` - Heavily Streamlit → extract pure logic only
- `core/workers.py` - Streamlit threading hacks → rewrite as async tasks
- `core/jobs.py` - Uses `st.status/progress` → extract submission logic
- `core/session.py` - Pure Streamlit → DELETE (replaced by Pinia stores)

### Full Rewrite (Streamlit UI → Vue.js)
- `app.py` → `backend/main.py` (FastAPI app)
- `pages/real_time.py` → `frontend/views/RealTimeView.vue` + `backend/api/predictions.py`
- `pages/nowcasting.py` → `frontend/views/NowcastingView.vue` + `backend/api/gifs.py`
- `pages/prediction_by_date.py` → `frontend/views/PredictionByDateView.vue`
- `pages/model_comparison.py` → `frontend/views/ModelComparisonView.vue`
- `pages/csi_analysis.py` → `frontend/views/MetricsView.vue` + `backend/api/metrics.py`
- `ui/components.py` → Vue components
- `ui/maps.py` → `frontend/components/RadarMap.vue` (Leaflet.js)

---

## Step-by-Step Migration Plan

### Phase 0: Project Setup & Learning
> Goal: Set up the new project, install dependencies, understand the basics

#### 0.1 Create project structure
- [ ] Create `nwc_webapp_v2/` directory alongside existing code
- [ ] Create `backend/` and `frontend/` subdirectories
- [ ] Set up Python virtual environment for backend
- [ ] Set up Node.js project for frontend

#### 0.2 Backend setup (FastAPI)
- [ ] Install FastAPI + uvicorn: `pip install fastapi uvicorn[standard] websockets python-multipart`
- [ ] Create minimal `backend/main.py` with "Hello World" endpoint
- [ ] Test: `uvicorn main:app --reload` → visit http://localhost:8000
- [ ] **Learn**: How FastAPI routes work, what `async def` means, how to return JSON

#### 0.3 Frontend setup (Vue 3 + Vite)
- [ ] Install Node.js (if not present): `brew install node`
- [ ] Create Vue project: `npm create vue@latest frontend`
- [ ] Install dependencies: Leaflet, Tailwind CSS, Pinia, Vue Router
- [ ] Create minimal page that shows "Hello World"
- [ ] Test: `npm run dev` → visit http://localhost:5173
- [ ] **Learn**: Vue single-file components (.vue files), how `<script setup>` works

#### 0.4 Connect frontend ↔ backend
- [ ] Configure Vite proxy to forward `/api/*` to FastAPI backend
- [ ] Create a test API endpoint: `GET /api/health` → returns `{"status": "ok"}`
- [ ] Call it from Vue using `fetch()` and display the result
- [ ] **Learn**: How the frontend talks to the backend, what CORS is

---

### Phase 1: Core Backend (port Python logic)
> Goal: Get the FastAPI backend serving data, independent of any frontend

#### 1.1 Copy reusable modules
- [ ] Copy all "fully reusable" modules listed above into `backend/core/`
- [ ] Copy `resources/` directory (cfg, legends, mask, shapefiles)
- [ ] Update imports to match new package structure
- [ ] Verify imports work: `python -c "from core.config.config import get_config; print(get_config())"`

#### 1.2 Config API
- [ ] `GET /api/config` → returns available models, thresholds, environment info
- [ ] `GET /api/config/models` → returns model list
- [ ] Test with `curl http://localhost:8000/api/config`

#### 1.3 Data API (groundtruth & predictions)
- [ ] `GET /api/data/latest-sri` → returns latest SRI filename
- [ ] `GET /api/data/groundtruth/{timestamp}` → returns groundtruth data as JSON or binary
- [ ] `GET /api/predictions/{model}/{timestamp}` → returns prediction array (numpy → JSON)
- [ ] `GET /api/predictions/{model}/check?start={}&end={}` → returns which predictions exist/missing
- [ ] `GET /api/predictions/{model}/status` → returns prediction file status
- [ ] Reuse: `data/checking.py`, `data/predictions.py`, `data/groundtruth.py`

#### 1.4 Job API
- [ ] `POST /api/jobs/submit` → submit PBS or mock prediction job
- [ ] `GET /api/jobs/{job_id}/status` → check job status
- [ ] `GET /api/jobs/model/{model_name}/status` → check model job status
- [ ] Reuse: `hpc/pbs.py`, `hpc/jobs.py`, `mock/predictions.py`

#### 1.5 Rendering API (figures & GIFs)
- [ ] `GET /api/render/figure/{model}/{timestamp}?lead_time=30` → returns PNG image
- [ ] `POST /api/gifs/create` → trigger GIF creation, return task ID
- [ ] `GET /api/gifs/{model}/{start}_{end}` → return GIF file
- [ ] `GET /api/gifs/check?model={}&start={}&end={}` → check if GIFs exist
- [ ] Reuse: `rendering/figures.py`, `rendering/gifs.py` (remove st.progress)

#### 1.6 Metrics API
- [ ] `POST /api/metrics/compute` → compute CSI/POD/FAR/FSS for models in date range
- [ ] `GET /api/metrics/results/{task_id}` → get computation results
- [ ] Reuse: `evaluation/metrics.py`, `pages/csi_helpers.py`

#### 1.7 Background task system
- [ ] Implement `BackgroundTasks` for long-running operations (GIF creation, metrics)
- [ ] Simple in-memory task store: `{task_id: {status, progress, result}}`
- [ ] `GET /api/tasks/{task_id}` → check task progress
- [ ] **No Redis/Celery needed** - FastAPI's built-in background tasks are enough

#### 1.8 WebSocket for real-time updates
- [ ] `WS /ws/realtime` → pushes new SRI file notifications to connected clients
- [ ] `WS /ws/jobs/{job_id}` → pushes job progress updates
- [ ] Rewrite `core/workers.py` file monitoring as async background task
- [ ] **Learn**: What WebSocket is, how it differs from HTTP polling

---

### Phase 2: Frontend Foundation
> Goal: Build the basic UI shell with navigation and layout

#### 2.1 App layout
- [ ] Create main layout: top nav bar + content area
- [ ] Style with Tailwind CSS: dark header, clean white content
- [ ] Add Leonardo logo to header
- [ ] Navigation tabs: Real Time | Nowcasting | Prediction by Date | Model Comparison | Metrics

#### 2.2 Vue Router setup
- [ ] Set up routes for each tab/page
- [ ] Tabs become routes: `/realtime`, `/nowcasting`, `/prediction`, `/comparison`, `/metrics`
- [ ] Active tab highlighting in nav bar

#### 2.3 Pinia stores
- [ ] `appStore`: global config, models list, environment info
- [ ] `realtimeStore`: latest file, model statuses, paused state
- [ ] `predictionsStore`: prediction data, GIF paths
- [ ] `jobsStore`: active jobs, progress
- [ ] **Learn**: What a "store" is and why it replaces st.session_state

#### 2.4 API composable
- [ ] Create `useApi.js` composable with `fetch` wrappers
- [ ] Base URL configuration (points to FastAPI)
- [ ] Error handling and loading states
- [ ] On app startup: fetch config, models list

---

### Phase 3: Real-Time Prediction Page
> Goal: Replicate the real-time prediction tab with smooth map

#### 3.1 Leaflet map component (`RadarMap.vue`)
- [ ] Initialize Leaflet map centered on Italy (42.0, 12.5)
- [ ] Add tile layers: OSM, Satellite, Gray Canvas
- [ ] Add radar markers from `radar_positions.txt`
- [ ] Add precipitation colorbar/legend
- [ ] **Learn**: How Leaflet.js works (it's the same as Folium but in JS)

#### 3.2 Radar overlay on map
- [ ] Backend: `GET /api/render/overlay/{timestamp}` → returns RGBA PNG for map overlay
- [ ] Frontend: Use Leaflet `L.imageOverlay()` to display radar data on map
- [ ] Same bounds as current: `[[35.0623, 4.51987], [47.5730, 20.4801]]`
- [ ] Smooth opacity transitions between frames

#### 3.3 Animation controls
- [ ] Play/Pause button
- [ ] Timeline slider (19 timesteps: -30min to +60min)
- [ ] Speed control (Slow/Normal/Fast/Very Fast)
- [ ] Current timestamp display with formatted time
- [ ] All client-side - no server calls during animation

#### 3.4 Model selector
- [ ] Dropdown to select active model
- [ ] When model changes, fetch new prediction overlay

#### 3.5 Status panel
- [ ] Show latest SRI file, model statuses, system info
- [ ] WebSocket connection for real-time updates
- [ ] Animated status indicators (computing, queue, ready)
- [ ] Pause/Resume button for real-time monitoring

#### 3.6 WebSocket integration
- [ ] Connect to `WS /ws/realtime` on page mount
- [ ] When backend detects new SRI file → push to all connected clients
- [ ] Client fetches new overlay data and updates map
- [ ] No polling, no page reloads

---

### Phase 4: Nowcasting Page
> Goal: Date range prediction with GIF creation and display

#### 4.1 Sidebar form (becomes in-page form)
- [ ] Model selector dropdown
- [ ] Start date/time picker
- [ ] End date/time picker
- [ ] Submit button
- [ ] Form validation (end > start, dates not in future)

#### 4.2 Prediction checking
- [ ] On submit: call `GET /api/predictions/{model}/check?start={}&end={}`
- [ ] Display status: all exist / partially exist / none exist
- [ ] Show which timestamps are missing

#### 4.3 Job submission and monitoring
- [ ] "Compute Predictions" button → `POST /api/jobs/submit`
- [ ] Progress indicator with WebSocket updates
- [ ] Job status: queued → running → completed/failed
- [ ] Error display with PBS log content if available

#### 4.4 GIF creation and display
- [ ] "Create GIFs" button → `POST /api/gifs/create`
- [ ] Progress bar during GIF creation
- [ ] Display 7 GIFs in grid layout:
  - Row 1: Groundtruth, Target+30, Target+60
  - Row 2: Prediction+30, Prediction+60
  - Row 3: Difference+30, Difference+60
- [ ] Back button to return to form

#### 4.5 Missing target data warning
- [ ] Check target data availability before GIF creation
- [ ] Show warning dialog with proceed/cancel options
- [ ] Empty frames for missing data

---

### Phase 5: Prediction by Date Page
> Goal: Single-timestamp prediction view

#### 5.1 Date/time selection
- [ ] Date picker, time picker (5-min intervals), model selector
- [ ] "Check/Compute Prediction" button

#### 5.2 Prediction workflow
- [ ] Check if prediction exists → show Display/Recompute options
- [ ] If missing → submit job, monitor, display when ready
- [ ] Display: groundtruth grid (12 frames) + target + prediction + difference columns

#### 5.3 Image grid display
- [ ] Server renders matplotlib figures, returns as PNG
- [ ] Client displays in responsive grid
- [ ] Colorbar alongside images

---

### Phase 6: Model Comparison Page
> Goal: Side-by-side model comparison at single timestamp

#### 6.1 Timestamp selection + ground truth loading
- [ ] Date/time picker
- [ ] "Load Ground Truth" button

#### 6.2 Model management
- [ ] "Add Model" button → check if prediction exists → display/compute
- [ ] Remove model button
- [ ] Dynamic column layout based on number of models

#### 6.3 Comparison grid
- [ ] For each lead time (5-60 min):
  - Ground truth image + model prediction images + CSI table
- [ ] Synchronized zoom across images (keep current JS logic)
- [ ] CSI metrics table per lead time

---

### Phase 7: Metrics Analysis Page
> Goal: CSI/POD/FAR/FSS computation and visualization

#### 7.1 Date range and model selection
- [ ] Start/end date-time pickers
- [ ] Model checkboxes with prediction status
- [ ] "Select All with predictions" / "Deselect All" buttons

#### 7.2 Computation workflow
- [ ] "Compute Predictions" → submit jobs for selected models
- [ ] "Compute Metrics" → background task for CSI/FSS computation
- [ ] Progress tracking via WebSocket

#### 7.3 Results display
- [ ] CSI tables with color gradients (use a JS table library)
- [ ] CSI vs Lead Time line plots (use Chart.js or similar)
- [ ] Performance Fit Diagrams
- [ ] FSS by window size tables and charts
- [ ] Overall model comparison bar chart
- [ ] Summary metrics cards (Best/Worst model, Overall average)

---

### Phase 8: Polish & Integration
> Goal: Make it production-ready and beautiful

#### 8.1 Static file serving
- [ ] Build frontend: `npm run build` → outputs to `frontend/dist/`
- [ ] Configure FastAPI to serve `frontend/dist/` as static files
- [ ] Single `uvicorn main:app` serves everything (API + frontend)

#### 8.2 Error handling
- [ ] Global error handler in FastAPI (returns JSON errors)
- [ ] Frontend error display (toast notifications)
- [ ] Graceful degradation when backend is unreachable

#### 8.3 Loading states
- [ ] Skeleton loaders while data is fetching
- [ ] Progress bars for long operations
- [ ] Smooth transitions between states

#### 8.4 Responsive design
- [ ] Works well on different screen sizes
- [ ] Sidebar collapses on small screens
- [ ] Map takes full width when possible

#### 8.5 Performance optimization
- [ ] Cache frequently accessed data (config, model list)
- [ ] Lazy-load heavy components (maps, charts)
- [ ] Compress API responses (gzip)

#### 8.6 Dark mode (optional)
- [ ] Tailwind dark mode toggle
- [ ] Dark-friendly colormaps

---

### Phase 9: HPC Deployment
> Goal: Run on HPC via SSH tunnel

#### 9.1 HPC preparation
- [ ] Install Python deps: `pip install fastapi uvicorn websockets`
- [ ] Build frontend locally, commit `static/` folder
- [ ] No Node.js needed on HPC (just static files)
- [ ] Verify all data paths work with HPC config

#### 9.2 Launch script
- [ ] Create `start_webapp.sh` script:
  ```bash
  #!/bin/bash
  conda activate nwc_webapp
  cd /path/to/nwc_webapp_v2/backend
  uvicorn main:app --host 0.0.0.0 --port 8000
  ```
- [ ] Access via SSH tunnel: `ssh -L 8000:localhost:8000 hpc-node`

#### 9.3 Testing on HPC
- [ ] Test all tabs with real data
- [ ] Test PBS job submission
- [ ] Test real-time file monitoring with real SRI files
- [ ] Verify performance with large datasets

---

## Technology Quick Reference

### Backend (Python - you already know this!)

**FastAPI** - Modern Python web framework
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/api/models")           # GET endpoint - returns data
async def get_models():
    config = get_config()
    return {"models": config.models}

@app.post("/api/jobs/submit")     # POST endpoint - receives data, does something
async def submit_job(model: str, start: str, end: str):
    job_id = submit_date_range_prediction_job(model, start, end)
    return {"job_id": job_id}
```

**uvicorn** - The server that runs FastAPI (like `streamlit run` but for FastAPI)
```bash
uvicorn main:app --reload --port 8000
# --reload: auto-restart when code changes (like streamlit does)
# --port: which port to listen on
```

**WebSocket** - Persistent connection for real-time updates
```python
from fastapi import WebSocket

@app.websocket("/ws/realtime")
async def realtime_ws(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Send updates when new data arrives
        await websocket.send_json({"new_file": "22-11-2025-20-00.hdf"})
        await asyncio.sleep(5)
```

### Frontend (JavaScript - new territory!)

**Vue 3** - UI framework (similar concept to Streamlit widgets, but client-side)
```vue
<template>
  <!-- HTML template - what you see -->
  <div>
    <select v-model="selectedModel">
      <option v-for="model in models" :key="model">{{ model }}</option>
    </select>
    <button @click="loadPrediction">Load</button>
  </div>
</template>

<script setup>
// JavaScript logic - what happens
import { ref } from 'vue'

const selectedModel = ref('ConvLSTM')   // Like st.session_state["model"]
const models = ref([])                   // List that updates the UI automatically

async function loadPrediction() {
  const response = await fetch(`/api/predictions/${selectedModel.value}`)
  const data = await response.json()
  // Do something with data...
}
</script>
```

**Leaflet.js** - Map library (same as Folium, but directly in JavaScript)
```javascript
// Create map (equivalent to folium.Map())
const map = L.map('map').setView([42.0, 12.5], 6)

// Add tile layer (equivalent to folium.TileLayer())
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map)

// Add image overlay (equivalent to folium.ImageOverlay())
L.imageOverlay(imageUrl, [[35.06, 4.52], [47.57, 20.48]]).addTo(map)
```

**Pinia** - State management (replaces st.session_state)
```javascript
// Define a store
export const useRealtimeStore = defineStore('realtime', {
  state: () => ({
    latestFile: null,        // Like st.session_state["latest_file"]
    selectedModel: null,     // Like st.session_state["selected_model"]
    isPaused: true,          // Like st.session_state["realtime_paused"]
  }),
  actions: {
    setLatestFile(file) { this.latestFile = file },
    togglePause() { this.isPaused = !this.isPaused },
  }
})
```

**Tailwind CSS** - Utility-first CSS framework (makes things look good with classes)
```html
<!-- Instead of writing CSS, you add classes directly -->
<div class="bg-white rounded-lg shadow-md p-6 flex gap-4">
  <button class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
    Click me
  </button>
</div>
```

---

## Key Concepts: Streamlit vs New Architecture

| Streamlit Concept | New Architecture Equivalent | Why it's better |
|---|---|---|
| `st.session_state["key"]` | Pinia store | State lives in browser, survives navigation, no reruns |
| `st.rerun()` | Vue reactivity (automatic) | UI updates instantly when data changes, no manual trigger |
| `st.spinner("Loading...")` | Loading component / skeleton | Non-blocking, doesn't freeze the entire page |
| `st.tabs(["Tab1", "Tab2"])` | Vue Router routes | URL changes, back button works, bookmarkable |
| `st.sidebar` | Sidebar component | Always responsive, doesn't rerun the app |
| `st.fragment(run_every=5)` | WebSocket push | Instant updates, no polling overhead |
| `streamlit-autorefresh` | WebSocket + event listener | More efficient, instant, no page reload |
| `st_folium(map)` | Leaflet.js directly | No iframe, instant pan/zoom, no server round-trip |
| `st.image(fig)` | `<img>` tag with API URL | Browser caches, lazy loads, instant display |
| Background threads | FastAPI BackgroundTasks + async | Designed for this, no hacky context injection |
| `add_script_run_ctx()` | Not needed | No Streamlit script context to manage |
| `st.progress(0.5)` | Progress component + WebSocket | Updates without blocking, works across pages |

---

## Priority Order

Work in this order. Each phase produces something testable:

1. **Phase 0** → You can run both FastAPI and Vue dev servers
2. **Phase 1** → Backend serves all data via API (test with curl/browser)
3. **Phase 2** → Frontend shows app layout with navigation
4. **Phase 3** → Real-time tab works with smooth map ← **first "wow" moment**
5. **Phase 4** → Nowcasting tab works with GIF display
6. **Phase 5** → Prediction by date works
7. **Phase 6** → Model comparison works
8. **Phase 7** → Metrics analysis works
9. **Phase 8** → Everything polished and production-ready
10. **Phase 9** → Deploy to HPC

---

## Estimated Effort

| Phase | Effort | What you get |
|-------|--------|-------------|
| Phase 0: Setup | 1 day | Project structure, tools installed, "Hello World" working |
| Phase 1: Backend | 3-4 days | Complete API serving all data |
| Phase 2: Frontend Foundation | 2 days | App shell with navigation |
| Phase 3: Real-time | 3-4 days | Smooth real-time map (biggest visual impact) |
| Phase 4: Nowcasting | 2-3 days | Date range predictions + GIFs |
| Phase 5: Prediction by Date | 1-2 days | Single timestamp predictions |
| Phase 6: Model Comparison | 2-3 days | Side-by-side comparison |
| Phase 7: Metrics | 2-3 days | CSI/FSS analysis |
| Phase 8: Polish | 2-3 days | Error handling, loading states, responsive design |
| Phase 9: HPC Deploy | 1 day | Running on HPC |
| **Total** | **~3-5 weeks** | **Complete migration** |

---

## Notes

- The Streamlit app continues to work during migration (we're building alongside, not replacing)
- Each phase is independently testable
- Frontend can be built locally with hot reload, then committed as static files
- No Docker needed anywhere in this stack
- No external services (Redis, databases) needed
- Same `pip install` workflow as current project