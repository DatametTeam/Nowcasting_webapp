/**
 * API Service - centralized HTTP calls to the FastAPI backend.
 *
 * WHY THIS FILE EXISTS:
 * Instead of writing fetch() calls everywhere, we define them ONCE here.
 * Each Vue component just calls api.getModels() or api.submitJob(...).
 *
 * This is like having a "client library" for your backend API.
 * If the API URL changes, you only update it here.
 *
 * HOW FETCH WORKS:
 * fetch() is the browser's built-in way to make HTTP requests.
 *   const response = await fetch('/api/something')  // GET request
 *   const data = await response.json()               // parse JSON response
 *
 * For POST requests:
 *   await fetch('/api/something', {
 *     method: 'POST',
 *     headers: { 'Content-Type': 'application/json' },
 *     body: JSON.stringify({ key: 'value' })
 *   })
 */

const API_BASE = '/api'

/**
 * Helper: make a GET request and return JSON.
 * Throws an error if the response is not OK (status 4xx or 5xx).
 */
async function get(path) {
  const response = await fetch(`${API_BASE}${path}`)
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
  return response.json()
}

/**
 * Helper: make a POST request with JSON body and return JSON.
 */
async function post(path, data) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
  return response.json()
}

// ============================================================================
// API methods — one function per backend endpoint
// ============================================================================

export default {
  // --- Health ---
  health: () => get('/health'),

  // --- Config ---
  getConfig: () => get('/config/'),
  getModels: () => get('/config/models'),
  getEnvironment: () => get('/config/environment'),

  // --- Data ---
  getLatestSRI: () => get('/data/sri/latest'),
  generateNextMockData: () => post('/data/mock/generate-next'),

  checkPredictions: (model, start, end) =>
    get(`/data/predictions/check?model=${model}&start=${start}&end=${end}`),

  checkSinglePrediction: (model, timestamp) =>
    get(`/data/predictions/check-single?model=${model}&timestamp=${timestamp}`),

  checkTargetRange: (start, end) =>
    get(`/data/target/check-range?start=${start}&end=${end}`),

  checkSingleTarget: (timestamp) =>
    get(`/data/target/check-single?timestamp=${timestamp}`),

  // --- Jobs ---
  submitJob: (model, start, end) =>
    post('/jobs/submit', { model, start, end }),

  getJobStatus: (model) =>
    get(`/jobs/status?model=${model}`),

  // --- Rendering ---
  /**
   * Returns the URL for an overlay image (not fetched, used as <img src>).
   * The browser will request the image directly from the backend.
   */
  /**
   * Returns the URL for a groundtruth (SRI) overlay image.
   * Used for the past portion of the timeline (-60 to 0 minutes).
   */
  groundtruthOverlayUrl: (timestamp) =>
    `${API_BASE}/render/overlay/groundtruth/${timestamp}`,

  overlayUrl: (model, timestamp, leadTime = 0, frameType = 'prediction') =>
    `${API_BASE}/render/overlay/${model}/${timestamp}?lead_time=${leadTime}&frame_type=${frameType}`,

  figureUrl: (model, timestamp, leadTime = 0, figureType = 'prediction') =>
    `${API_BASE}/render/figure/${model}/${timestamp}?lead_time=${leadTime}&figure_type=${figureType}`,

  checkGifs: (model, start, end) =>
    get(`/render/gifs/check?model=${model}&start=${start}&end=${end}`),

  gifFileUrl: (path) =>
    `${API_BASE}/render/gifs/file?path=${encodeURIComponent(path)}`,

  // --- Real-time ---
  startRealTime: () => post('/realtime/start'),
  stopRealTime: () => post('/realtime/stop'),
  getRealTimeStatus: () => get('/realtime/status'),

  // --- Metrics ---
  computeMetrics: (models, start, end) =>
    post('/metrics/compute', { models, start, end }),
}