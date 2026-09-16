import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import { resolveBaseLayerName } from '../basemaps.js'

const STORAGE_KEY = 'nwc_settings'

export const useSettingsStore = defineStore('settings', () => {
  const saved = (() => {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}') }
    catch { return {} }
  })()

  // resolveBaseLayerName also migrates retired base map names to the default
  const baseLayer       = ref(resolveBaseLayerName(saved.baseLayer))
  const showColorbars   = ref(saved.showColorbars   ?? true)
  const timeZone        = ref(saved.timeZone        ?? 'local')  // 'local' | 'utc'
  const defaultLookback = ref(saved.defaultLookback ?? 1)        // hours
  const defaultSpeed    = ref(saved.defaultSpeed    ?? 1)        // playback speed

  function persist() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      baseLayer:       baseLayer.value,
      showColorbars:   showColorbars.value,
      timeZone:        timeZone.value,
      defaultLookback: defaultLookback.value,
      defaultSpeed:    defaultSpeed.value,
    }))
  }

  watch([baseLayer, showColorbars, timeZone, defaultLookback, defaultSpeed], persist)

  return { baseLayer, showColorbars, timeZone, defaultLookback, defaultSpeed }
})
