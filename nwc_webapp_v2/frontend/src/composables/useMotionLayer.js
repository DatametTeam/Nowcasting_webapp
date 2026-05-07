/**
 * useMotionLayer — shared composable for the AMV / LK wind layer.
 *
 * Encapsulates all fetch, cache, and map-update logic so that RealTimeView,
 * NowcastingView and WR10View can all share the same behaviour with a single
 * import.
 *
 * Usage:
 *   const { motionMode, motionLoading, activeMotionTs, updateMotionLayer,
 *           fetchTimestamps, prefetchData } = useMotionLayer(radarMap, currentTs)
 *
 * @param {Ref<object>}  radarMap   — ref to the RadarMap component instance
 * @param {Ref<string>}  currentTs  — ref/computed returning the current radar
 *                                    timestamp as "YYYY-MM-DDTHH:MM" (or "")
 */

import { ref, watch } from 'vue'
import api from '../api.js'

export function useMotionLayer(radarMap, currentTs) {
  // 'none' | 'amv' | 'lk'
  const motionMode    = ref('none')
  const motionLoading = ref(false)
  const activeMotionTs = ref('')

  // Per-source caches and timestamp lists
  const _state = {
    amv: { timestamps: ref([]), cache: {} },
    lk:  { timestamps: ref([]), cache: {} },
  }

  // ── API wrappers ────────────────────────────────────────────────────────────

  function _fetchTimestampsFn(source) {
    return source === 'amv' ? api.windTimestamps() : api.lkTimestamps()
  }

  function _fetchDataFn(source, ts) {
    return source === 'amv' ? api.windData(ts) : api.lkData(ts)
  }

  // ── Nearest timestamp lookup ────────────────────────────────────────────────

  /**
   * Return the most recent timestamp ≤ radarTs that loaded successfully,
   * or '' if none is available.
   */
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

    const source = motionMode.value   // 'amv' | 'lk'
    const radarTsShort = (currentTs.value || '').slice(0, 16)
    const target = _nearestTs(source, radarTsShort)

    if (!target) {
      radarMap.value.clearWindLayer()
      activeMotionTs.value = ''
      return
    }

    if (target === activeMotionTs.value) return   // same snapshot, nothing to do

    motionLoading.value = true
    try {
      const { cache } = _state[source]
      if (!(target in cache)) {
        cache[target] = await _fetchDataFn(source, target).catch(() => null)
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

  // ── Timestamp fetch + prefetch ──────────────────────────────────────────────

  /**
   * Fetch (or refresh) the timestamp list for the given source.
   * Call on mount and periodically.
   */
  async function fetchTimestamps(source) {
    if (!source || source === 'none') return
    try {
      const key = source === 'amv' ? 'timestamps' : 'timestamps'
      const { timestamps } = _state[source]
      const result = await _fetchTimestampsFn(source)
      timestamps.value = result.timestamps ?? []
      prefetchData(source)
      if (motionMode.value === source) updateMotionLayer()
    } catch (e) {
      console.warn(`Could not fetch ${source} timestamps:`, e)
    }
  }

  /**
   * Background-prefetch all uncached timestamps that fall within the
   * radar lookback window so the slider is instant.
   */
  function prefetchData(source, windowStartTs = '') {
    if (!source || source === 'none') return
    const { timestamps, cache } = _state[source]
    const toFetch = timestamps.value.filter(
      ts => ts >= windowStartTs && !(ts in cache)
    )
    for (const ts of toFetch) {
      _fetchDataFn(source, ts)
        .then(data  => { cache[ts] = data })
        .catch(() => { cache[ts] = null  })
    }
  }

  // ── Watchers ────────────────────────────────────────────────────────────────

  watch(motionMode, async (mode) => {
    // Clear whatever is currently on the map
    radarMap.value?.clearWindLayer()
    activeMotionTs.value = ''

    if (mode === 'none') return

    // Fetch timestamps for the newly selected source if not yet loaded
    if (_state[mode].timestamps.value.length === 0) {
      await fetchTimestamps(mode)
    }
    await updateMotionLayer()
  })

  watch(currentTs, () => {
    if (motionMode.value !== 'none') updateMotionLayer()
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
