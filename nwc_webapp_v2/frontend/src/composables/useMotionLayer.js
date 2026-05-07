/**
 * useMotionLayer — shared composable for the AMV / LK wind layer.
 *
 * Handles fetch, cache, map-update, and WebSocket-driven refresh so that
 * RealTimeView, NowcastingView and WR10View share identical behaviour.
 *
 * Usage:
 *   const { motionMode, motionLoading, activeMotionTs, lkDisplayMode,
 *           lkArrowOpacity, updateMotionLayer, fetchTimestamps }
 *     = useMotionLayer(radarMap, currentTs)
 *
 * @param {Ref<object>}  radarMap   — ref to the RadarMap component instance
 * @param {Ref<string>}  currentTs  — current radar timestamp "YYYY-MM-DDTHH:MM" (or "")
 */

import { ref, watch, onUnmounted } from 'vue'
import api from '../api.js'

const BASE_DELAY_MS = 1_000
const MAX_DELAY_MS  = 30_000

export function useMotionLayer(radarMap, currentTs) {
  // 'none' | 'amv' | 'lk'
  const motionMode     = ref('none')
  const motionLoading  = ref(false)
  const activeMotionTs = ref('')

  // LK display sub-mode — only relevant when motionMode === 'lk'
  // 'particles' = animated leaflet-velocity only
  // 'arrows'    = static PNG quiver overlay only
  // 'both'      = both layers simultaneously
  const lkDisplayMode  = ref('particles')
  const lkArrowOpacity = ref(0.8)

  const _state = {
    amv: { timestamps: ref([]), cache: {} },
    lk:  { timestamps: ref([]), cache: {} },
  }

  // ── API wrappers ────────────────────────────────────────────────────────────

  function _fetchTs(source)       { return source === 'amv' ? api.windTimestamps() : api.lkTimestamps() }
  function _fetchData(source, ts) { return source === 'amv' ? api.windData(ts)     : api.lkData(ts) }

  // ── Nearest timestamp lookup ────────────────────────────────────────────────

  function _nearestTs(source, radarTs) {
    if (!radarTs) return ''
    const { timestamps, cache } = _state[source]
    if (timestamps.value.length === 0) return ''
    let best = ''
    for (const ts of timestamps.value) {
      if (ts > radarTs) break
      if (cache[ts] !== null) best = ts
    }
    return best
  }

  // ── Apply current display mode to the map ──────────────────────────────────
  // Called whenever activeMotionTs, lkDisplayMode, or lkArrowOpacity changes.

  function _applyDisplayMode(target) {
    if (!radarMap.value || !target) return
    const source = motionMode.value
    const data   = _state[source]?.cache[target]

    const showParticles = source === 'amv' || lkDisplayMode.value !== 'arrows'
    const showArrows    = source === 'lk'  && lkDisplayMode.value !== 'particles'

    if (showParticles && data) {
      radarMap.value.setWindLayer(data)
    } else {
      radarMap.value.clearWindLayer()
    }

    if (showArrows) {
      radarMap.value.setLkImage(api.lkImageUrl(target), lkArrowOpacity.value)
    } else {
      radarMap.value.clearLkImage()
    }
  }

  // ── Layer update ────────────────────────────────────────────────────────────

  async function updateMotionLayer() {
    if (motionMode.value === 'none' || !radarMap.value) return

    const source      = motionMode.value
    const radarTsShort = (currentTs.value || '').slice(0, 16)
    const target      = _nearestTs(source, radarTsShort)

    if (!target) {
      radarMap.value.clearWindLayer()
      radarMap.value.clearLkImage()
      activeMotionTs.value = ''
      return
    }

    // Target unchanged — re-apply display mode in case opacity/mode changed
    if (target === activeMotionTs.value) {
      _applyDisplayMode(target)
      return
    }

    motionLoading.value = true
    try {
      const { cache } = _state[source]
      if (!(target in cache)) {
        cache[target] = await _fetchData(source, target).catch(() => null)
      }
      if (!cache[target]) {
        radarMap.value.clearWindLayer()
        radarMap.value.clearLkImage()
        activeMotionTs.value = ''
        return
      }
      _applyDisplayMode(target)
      activeMotionTs.value = target
    } catch (e) {
      console.warn('Motion layer fetch failed:', e)
      activeMotionTs.value = ''
    } finally {
      motionLoading.value = false
    }
  }

  // ── Timestamp fetch ─────────────────────────────────────────────────────────

  async function fetchTimestamps(source) {
    if (!source || source === 'none') return
    try {
      const { timestamps } = _state[source]
      const result = await _fetchTs(source)
      timestamps.value = result.timestamps ?? []
      prefetchData(source)
      if (source === 'lk') _prefetchImages()
    } catch (e) {
      console.warn(`Could not fetch ${source} timestamps:`, e)
    }
  }

  // Fetch JSON data newest-first and in batches of 4 so that playback frames
  // (always the most recent) are cached before the old history is fetched,
  // and the browser's 6-conn-per-origin pool isn't saturated all at once.
  function prefetchData(source, windowStartTs = '') {
    if (!source || source === 'none') return
    const { timestamps, cache } = _state[source]
    // Newest-first: playback always starts from recent frames, so cache those first.
    // Without this, fetching 200+ old timestamps fills the connection pool before
    // the frames being played are even queued.
    const toFetch = timestamps.value
      .filter(ts => ts >= windowStartTs && !(ts in cache))
      .slice()
      .reverse()

    // Batch to 4 concurrent requests — stays within Chrome's 6-conn-per-origin limit
    // while leaving headroom for other API calls (radar images, etc.).
    const CONCURRENCY = 4
    let idx = 0
    function runBatch() {
      const batch = toFetch.slice(idx, idx + CONCURRENCY)
      if (!batch.length) return
      idx += CONCURRENCY
      Promise.all(batch.map(ts =>
        _fetchData(source, ts)
          .then(data  => { cache[ts] = data })
          .catch(() => { cache[ts] = null })
      )).then(runBatch)
    }
    runBatch()
  }

  // Pre-warm the browser image cache for all LK PNGs (newest-first).
  // By the time the user plays the timeline, most PNGs are already cached
  // so setUrl() calls in setLkImage() are served from cache instantly.
  function _prefetchImages() {
    const { timestamps } = _state.lk
    const toFetch = [...timestamps.value].reverse()
    for (const ts of toFetch) {
      const img = new Image()
      img.src = api.lkImageUrl(ts)
    }
  }

  // ── Watchers ────────────────────────────────────────────────────────────────

  watch(motionMode, async (mode) => {
    radarMap.value?.clearWindLayer()
    radarMap.value?.clearLkImage()
    activeMotionTs.value = ''

    if (mode === 'none') return

    // Show spinner immediately — before any async work
    motionLoading.value = true
    try {
      if (_state[mode].timestamps.value.length === 0) {
        await fetchTimestamps(mode)
      }
      await updateMotionLayer()
    } finally {
      motionLoading.value = false
    }
  })

  watch(currentTs, () => {
    if (motionMode.value !== 'none') updateMotionLayer()
  })

  // Re-apply layers whenever the LK display sub-mode or arrow opacity changes
  watch(lkDisplayMode, () => {
    if (motionMode.value === 'lk' && activeMotionTs.value) {
      _applyDisplayMode(activeMotionTs.value)
    }
  })

  watch(lkArrowOpacity, () => {
    if (motionMode.value === 'lk' && activeMotionTs.value && lkDisplayMode.value !== 'particles') {
      radarMap.value?.setLkImage(api.lkImageUrl(activeMotionTs.value), lkArrowOpacity.value)
    }
  })

  // ── WebSocket: LK live updates ──────────────────────────────────────────────
  // Connects to /api/lk/ws and forces a timestamp refresh + layer re-render
  // whenever the cron script notifies that a new flow field has been saved.
  // (AMV has no WS yet — it updates every 20 min so polling is fine.)

  let _ws        = null
  let _retryDelay = BASE_DELAY_MS
  let _retryTimer = null
  let _stopped    = false

  function _wsUrl() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    return `${proto}://${location.host}/api/lk/ws`
  }

  function _connectWs() {
    if (_stopped) return
    _ws = new WebSocket(_wsUrl())

    _ws.onmessage = async (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg.type === 'lk_updated') {
          // Clear LK cache so the next fetch reads the new file from disk
          _state.lk.cache = {}
          await fetchTimestamps('lk')
          if (motionMode.value === 'lk') {
            // Force re-render even if target timestamp hasn't changed
            activeMotionTs.value = ''
            await updateMotionLayer()
          }
        }
      } catch { /* malformed frame */ }
    }

    _ws.onopen  = () => { _retryDelay = BASE_DELAY_MS }
    _ws.onclose = () => {
      _ws = null
      if (!_stopped) {
        _retryTimer = setTimeout(() => { _retryTimer = null; _connectWs() }, _retryDelay)
        _retryDelay = Math.min(_retryDelay * 2, MAX_DELAY_MS)
      }
    }
    _ws.onerror = () => { /* onclose fires next */ }
  }

  _connectWs()

  onUnmounted(() => {
    _stopped = true
    if (_retryTimer) { clearTimeout(_retryTimer); _retryTimer = null }
    if (_ws) { _ws.onclose = null; _ws.close(); _ws = null }
  })

  // ── Motion sampling (for map-click popups) ──────────────────────────────────
  // Returns { speed_ms, speed_kmh, direction_deg, source } for the grid cell
  // nearest to (lat, lng), or null if no motion layer is active / out of grid.

  function sampleMotionAt(lat, lng) {
    const source = motionMode.value
    if (source === 'none') return null
    const ts = activeMotionTs.value
    if (!ts) return null
    const data = _state[source]?.cache[ts]
    if (!data || !Array.isArray(data) || data.length < 2) return null

    const hdr = data[0].header
    const { lo1, la1, nx, ny, dx, dy } = hdr
    if (!dx || !dy) return null

    const ci = Math.round((lng - lo1) / dx)
    const ri = Math.round((la1 - lat) / dy)   // la1 = northernmost row → ri increases southward

    if (ci < 0 || ci >= nx || ri < 0 || ri >= ny) return null

    const idx = ri * nx + ci
    const u = data[0].data[idx]
    const v = data[1].data[idx]

    if (!Number.isFinite(u) || !Number.isFinite(v)) return null

    const speed_ms  = Math.sqrt(u * u + v * v)
    // Direction toward: 0°=N, 90°=E (clockwise from north)
    const direction = (Math.atan2(u, v) * 180 / Math.PI + 360) % 360

    return { speed_ms, speed_kmh: speed_ms * 3.6, direction, source }
  }

  // ── Exposed ─────────────────────────────────────────────────────────────────

  return {
    motionMode,
    motionLoading,
    activeMotionTs,
    lkDisplayMode,
    lkArrowOpacity,
    updateMotionLayer,
    fetchTimestamps,
    prefetchData,
    sampleMotionAt,
  }
}
