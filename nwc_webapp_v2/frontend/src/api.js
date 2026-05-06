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

/**
 * Helper: make a POST request with JSON body and return a Blob (binary data).
 * Used for endpoints that return images (e.g. fit diagrams).
 */
async function postBlob(path, data) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
  return response.blob()
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

  /**
   * Sample radar product values (and optionally a model prediction) at a
   * single (lat, lon) for the popup. Returns:
   *   { lat, lon, x, y, in_bounds, timestamp, values: { SRI_adj, VMI, ..., __model__<name> } }
   */
  samplePixel: ({ lat, lon, timestamp, products = [], model = '', models = [], leadTime = 0 }) => {
    const params = new URLSearchParams({
      lat: String(lat),
      lon: String(lon),
      timestamp,
      products: products.join(','),
      lead_time: String(leadTime),
    })
    if (model) params.set('model', model)
    if (models.length) params.set('models', models.join(','))
    return get(`/data/sample?${params.toString()}`)
  },

  /** Get which dates in a month have predictions (for calendar highlighting). */
  getCalendarAvailability: (models, year, month) =>
    get(`/data/predictions/calendar?models=${models.join(',')}&year=${year}&month=${month}`),

  /** Get per-timestamp model availability for a specific date. */
  getDayDetail: (models, date) =>
    get(`/data/predictions/day-detail?models=${models.join(',')}&date=${date}`),

  // --- Jobs ---
  submitJob: (model, start, end) =>
    post('/jobs/submit', { model, start, end }),

  getJobStatus: (model, jobId = null) =>
    get(`/jobs/status?model=${model}${jobId ? `&job_id=${jobId}` : ''}`),

  getJobErrorLog: (model, jobId) =>
    get(`/jobs/error-log?model=${model}&job_id=${jobId}`),

  // --- Data Explorer ---
  /**
   * Get available HDF5 timestamps for a radar product in a date range (max 48h).
   */
  explorerTimestamps: (start, end, product = 'SRI_adj') =>
    get(`/data/explorer/timestamps?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}&product=${product}`),

  /**
   * Returns the URL for a product overlay at a given timestamp.
   * Used as an <img src> by RadarMap's ImageOverlay.
   */
  explorerOverlayUrl: (product, timestamp) =>
    `${API_BASE}/render/overlay/groundtruth/${encodeURIComponent(timestamp)}?product=${product}`,

  /**
   * Batch-render all frames for one product and return a ZIP blob.
   * timestamps: array of ISO strings (only the ones that exist on disk).
   * ZIP contains NNNN.png files indexed by position in the timestamps array.
   */
  explorerBatchOverlay: (product, timestamps) =>
    postBlob('/render/overlay/batch', { product, timestamps }),

  // --- Rendering ---
  /**
   * Returns the URL for a groundtruth overlay image.
   * Used for the past portion of the timeline (-60 to 0 minutes).
   * product defaults to 'SRI_adj'. Pass 'IR_108' for the satellite overlay.
   */
  groundtruthOverlayUrl: (timestamp, product = 'SRI_adj') =>
    `${API_BASE}/render/overlay/groundtruth/${encodeURIComponent(timestamp)}?product=${product}`,

  overlayUrl: (model, timestamp, leadTime = 0, frameType = 'prediction') =>
    `${API_BASE}/render/overlay/${model}/${timestamp}?lead_time=${leadTime}&frame_type=${frameType}`,

  ensembleOverlayUrl: (timestamp, leadTime, threshold, modelList, contours = false) =>
    `${API_BASE}/render/overlay/ensemble/${encodeURIComponent(timestamp)}?lead_time=${leadTime}&threshold=${threshold}&models=${modelList.join(',')}&contours=${contours ? 1 : 0}`,

  figureUrl: (model, timestamp, leadTime = 0, figureType = 'prediction') =>
    `${API_BASE}/render/figure/${model}/${timestamp}?lead_time=${leadTime}&figure_type=${figureType}`,

  // --- Real-time ---
  startRealTime: () => post('/realtime/start'),
  stopRealTime: () => post('/realtime/stop'),
  getRealTimeStatus: () => get('/realtime/status'),

  // --- Metrics ---
  computeMetrics: (models, start, end) =>
    post('/metrics/compute', { models, start, end }),

  computeComparison: (models, timestamp) =>
    post('/metrics/comparison', { models, timestamp }),

  /**
   * Generate a performance fit diagram (POD vs FAR scatter plot).
   * Returns a PNG Blob.
   */
  fitDiagram: (models, pod_values, far_values, csi_values, threshold) =>
    postBlob('/metrics/fit-diagram', { models, pod_values, far_values, csi_values, threshold }),

  // --- Wind / AMV ---
  windTimestamps: () => get('/wind/timestamps'),
  windData: (timestamp) => get(`/wind/data?timestamp=${encodeURIComponent(timestamp)}`),

  // --- WR10 small radar ---
  wr10Config: () => get('/wr10/config'),
  wr10Timestamps: (product, lookbackMinutes) =>
    get(`/wr10/timestamps?product=${product}&lookback_minutes=${lookbackMinutes}`),
  wr10OverlayUrl: (timestamp, product) =>
    `/api/render/overlay/wr10/${encodeURIComponent(timestamp)}?product=${product}`,
  wr10SamplePixel: ({ lat, lon, timestamp, products }) =>
    get(`/wr10/sample?lat=${lat}&lon=${lon}&timestamp=${encodeURIComponent(timestamp)}&products=${products.join(',')}`),

  // --- FSS real-time assessment ---
  fssRecent: (scale = 5, hours = 24) =>
    get(`/fss/recent?scale=${scale}&hours=${hours}`),
  fssDaily: (scale = 5, days = 90) =>
    get(`/fss/daily?scale=${scale}&days=${days}`),
}