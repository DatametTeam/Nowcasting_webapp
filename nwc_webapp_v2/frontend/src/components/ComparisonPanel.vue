<!--
  ComparisonPanel.vue — A minimal Leaflet map panel for the comparison tab.

  Intentionally lightweight: no geosearch, no radar markers, no wind layer.
  Just a base tile layer + one ImageOverlay that updates as the parent changes
  the overlay URL.

  The parent collects map instances (via getMap()) and wires up Leaflet.Sync
  so that pan/zoom on any panel mirrors to all others.
-->
<template>
  <div class="flex flex-col h-full min-h-0 relative overflow-hidden">
    <!-- Panel label -->
    <div class="absolute top-2 left-1/2 -translate-x-1/2 z-[1000] px-3 py-1 rounded-full text-xs font-bold shadow-lg pointer-events-none"
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
import { ref, watch, onMounted, onUnmounted } from 'vue'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import 'leaflet.sync'

const RADAR_BOUNDS = [[35.0623, 4.51987], [47.5730, 20.4801]]

const props = defineProps({
  label: { type: String, required: true },
  isGroundtruth: { type: Boolean, default: false },
  overlayUrl: { type: String, default: null },
  opacity: { type: Number, default: 0.75 },
  showZoom: { type: Boolean, default: false },
  center: { type: Array, default: () => [42.0, 12.5] },
  zoom: { type: Number, default: 6 },
})

const mapEl = ref(null)
let map = null
let tileLayer = null
let overlay = null

onMounted(() => {
  map = L.map(mapEl.value, {
    center: props.center,
    zoom: props.zoom,
    zoomControl: props.showZoom,
    attributionControl: false,
    zoomAnimation: true,
    fadeAnimation: false,   // disable to avoid flicker on overlay swap
  })

  tileLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '',
    subdomains: 'abcd',
    maxZoom: 20,
  }).addTo(map)

  if (props.overlayUrl) {
    overlay = L.imageOverlay(props.overlayUrl, RADAR_BOUNDS, {
      opacity: props.opacity,
      interactive: false,
    }).addTo(map)
  }
})

watch(() => props.overlayUrl, (url) => {
  if (!map) return
  if (overlay) {
    if (url) {
      overlay.setUrl(url)
    } else {
      map.removeLayer(overlay)
      overlay = null
    }
  } else if (url) {
    overlay = L.imageOverlay(url, RADAR_BOUNDS, {
      opacity: props.opacity,
      interactive: false,
    }).addTo(map)
  }
})

watch(() => props.opacity, (op) => {
  if (overlay) overlay.setOpacity(op)
})

onUnmounted(() => {
  if (map) {
    map.remove()
    map = null
  }
  tileLayer = null
  overlay = null
})

defineExpose({ getMap: () => map })
</script>

<style scoped>
/* Leaflet CSS is globally scoped — styles here are just for the wrapper */
:deep(.leaflet-container) {
  background: #1a1a2e;
}
:deep(.leaflet-image-layer) {
  image-rendering: pixelated;
}
</style>
