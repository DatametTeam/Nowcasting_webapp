/**
 * useMotionLayer — shared composable for the AMV / LK wind layer.
 *
 * Handles fetch, cache, map-update, and WebSocket-driven refresh so that
 * RealTimeView, NowcastingView and WR10View share identical behaviour.
 *
 * Usage:
 *   const { motionMode, motionLoading, activeMotionTs,
 *           updateMotionLayer, fetchTimestamps } = useMotionLayer(radarMap, currentTs)
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

  const _state = {
    amv: { timestamps: ref([]), cache: {} },
    lk:  { timestamps: ref([]), cache: {} },
  }

  // ── API wrappers ────────────────────────────────────────────────────────────

  function _fetchTs(source)     { return source === 'amv' ? api.windTimestamps() : api.lkTimestamps() }
  function _fetchData(source, ts) { return source === 'amv' ? api.windData(ts) : api.lkData(ts) }

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

  // ── Layer update ────────────────────────────────────────────────────────────

  async function updateMotionLayer() {
    if (motionMode.value === 'none' || !radarMap.value) return

    const source = motionMode.value
    const radarTsShort = (currentTs.value || '').slice(0, 16)
    const target = _nearestTs(source, radarTsShort)

    if (!target) {
      radarMap.value.clearWindLayer()
      activeMotionTs.value = ''
      return
    }
    if (target === activeMotionTs.value) return

    motionLoading.value = true
    try {
      const { cache } = _state[source]
      if (!(target in cache)) {
        cache[target] = await _fetchData(source, target).catch(() => null)
      }
      if (!cache[target]) {
        radarMap.value.clearWindLayer()
        activeMotionTs.value = ''
        return
      }
      radarMap.value.setWindLayer(cache[target])
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
    } catch (e) {
      console.warn(`Could not fetch ${source} timestamps:`, e)
    }
  }

  function prefetchData(source, windowStartTs = '') {
    if (!source || source === 'none') return
    const { timestamps, cache } = _state[source]
    const toFetch = timestamps.value.filter(ts => ts >= windowStartTs && !(ts in cache))
    for (const ts of toFetch) {
      _fetchData(source, ts)
        .then(data  => { cache[ts] = data })
        .catch(() => { cache[ts] = null })
    }
  }

  // ── Watchers ────────────────────────────────────────────────────────────────

  watch(motionMode, async (mode) => {
    radarMap.value?.clearWindLayer()
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

  // ── Exposed ─────────────────────────────────────────────────────────────────

  return {
    motionMode,
    motionLoading,
    activeMotionTs,
    updateMotionLayer,
    fetchTimestamps,
    prefetchData,
  }
}
