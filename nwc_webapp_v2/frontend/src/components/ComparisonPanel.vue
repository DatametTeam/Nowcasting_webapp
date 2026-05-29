<!--
  ComparisonPanel.vue — Lightweight Leaflet map panel for the comparison tab.

  API (imperative, same pattern as RadarMap.vue):
    preloadFrames(urls, onProgress?)  — download all N frames in parallel,
                                        create hidden ImageOverlay layers.
                                        onProgress(1) is called per loaded frame.
    showFrame(index)                  — instantly show frame N (opacity toggle).
    getMap()                          — returns the L.Map instance for Leaflet.Sync.
-->
<template>
  <div class="flex flex-col h-full min-h-0 relative overflow-hidden">
    <!-- Panel label -->
    <div class="absolute top-2 left-1/2 -translate-x-1/2 z-[1000] px-3 py-1 rounded-full
                text-xs font-bold shadow-lg pointer-events-none"
      :class="isGroundtruth
        ? 'bg-emerald-800/80 text-emerald-200 ring-1 ring-emerald-500/50'
        : 'bg-blue-900/80 text-blue-200 ring-1 ring-blue-500/50'">
      {{ label }}
    </div>
    <!-- Map container -->
    <div ref="mapEl" class="flex-1 min-h-0 w-full" />
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import 'leaflet.sync'

const RADAR_BOUNDS = [[35.0623, 4.51987], [47.5730, 20.4801]]
const OVERLAY_OPACITY = 0.75

const props = defineProps({
  label:         { type: String,  required: true },
  isGroundtruth: { type: Boolean, default: false },
  showZoom:      { type: Boolean, default: false },
  center:        { type: Array,   default: () => [42.0, 12.5] },
  zoom:          { type: Number,  default: 6 },
})

const mapEl = ref(null)
let map = null

// Array of Leaflet ImageOverlay layers — one per preloaded frame.
// Null slots mean the URL was missing or failed to load.
let frameLayers = []
let activeFrameIndex = -1
// Generation counter: bumped on each preloadFrames call so stale loads
// discard their results instead of populating the wrong frame set.
let preloadGen = 0

onMounted(() => {
  map = L.map(mapEl.value, {
    center: props.center,
    zoom: props.zoom,
    zoomControl: props.showZoom,
    attributionControl: false,
    zoomAnimation: true,
    fadeAnimation: false,
  })

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '',
    subdomains: 'abcd',
    maxZoom: 20,
  }).addTo(map)
})

onUnmounted(() => {
  clearFrames()
  if (map) { map.remove(); map = null }
})

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function clearFrames() {
  for (const layer of frameLayers) {
    if (map && layer) map.removeLayer(layer)
  }
  frameLayers = []
  activeFrameIndex = -1
}

// ---------------------------------------------------------------------------
// Public API (exposed to parent)
// ---------------------------------------------------------------------------

/**
 * Download all frame URLs in parallel, create hidden ImageOverlay layers.
 *
 * @param {(string|null)[]} urls    - One URL per frame; null = missing frame.
 * @param {function}        onProgress - Called with (1) each time a frame
 *                                       finishes loading (success or error).
 */
async function preloadFrames(urls, onProgress) {
  if (!map) return

  const thisGen = ++preloadGen
  clearFrames()

  const promises = urls.map((url, index) => {
    if (!url) {
      onProgress?.(1)
      return Promise.resolve({ index, url: null, success: false })
    }
    return new Promise((resolve) => {
      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.onload = () => { onProgress?.(1); resolve({ index, url, success: true }) }
      img.onerror = () => { onProgress?.(1); resolve({ index, url, success: false }) }
      img.src = url
    })
  })

  const results = await Promise.all(promises)

  // Discard if a newer preloadFrames call was made while we were loading
  if (thisGen !== preloadGen || !map) return

  frameLayers = new Array(urls.length).fill(null)
  for (const r of results) {
    if (r.success) {
      frameLayers[r.index] = L.imageOverlay(r.url, RADAR_BOUNDS, {
        opacity: 0,
        interactive: false,
      }).addTo(map)
    }
  }

  // Show the first frame immediately after preloading
  showFrame(0)
}

/**
 * Instantly show frame at `index`, hide the previously active frame.
 * Uses opacity toggle — no network request.
 */
function showFrame(index) {
  if (index < 0 || index >= frameLayers.length) return

  if (activeFrameIndex >= 0 && frameLayers[activeFrameIndex]) {
    frameLayers[activeFrameIndex].setOpacity(0)
  }
  if (frameLayers[index]) {
    frameLayers[index].setOpacity(OVERLAY_OPACITY)
  }
  activeFrameIndex = index
}

defineExpose({ preloadFrames, showFrame, getMap: () => map })
</script>

<style scoped>
:deep(.leaflet-container)   { background: #1a1a2e; }
:deep(.leaflet-image-layer) { image-rendering: pixelated; }
</style>
