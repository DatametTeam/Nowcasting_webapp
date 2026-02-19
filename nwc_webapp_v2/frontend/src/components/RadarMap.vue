<!--
  RadarMap.vue — Leaflet map with preloaded radar overlay frames.

  PRELOADING STRATEGY:
  Instead of loading one image at a time (which causes flickering when
  dragging the slider), we preload ALL 12 lead-time frames at once.

  Each frame becomes a Leaflet ImageOverlay layer, but only the active
  one is visible. Switching lead times just toggles CSS visibility —
  no network request, no loading delay. This is what makes it feel instant
  like Windy, where dragging the timeline slider shows frames immediately.

  The parent component calls:
    preloadFrames(urls)  → downloads all 12 PNGs in parallel
    showFrame(index)     → instantly shows frame N, hides the rest
-->
<template>
  <div class="relative w-full h-full">
    <div ref="mapContainer" class="w-full h-full rounded-lg" />
    <!-- Loading indicator while frames are being preloaded -->
    <div
      v-if="loading"
      class="absolute top-3 left-1/2 -translate-x-1/2 z-[1001]
             bg-black/70 backdrop-blur-sm text-white text-xs font-medium
             px-4 py-2 rounded-full flex items-center gap-2"
    >
      <svg class="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
      Loading frames... {{ loadedCount }}/{{ totalCount }}
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { GeoSearchControl, OpenStreetMapProvider } from 'leaflet-geosearch'
import 'leaflet-geosearch/dist/geosearch.css'

const props = defineProps({
  center: { type: Array, default: () => [42.0, 12.5] },
  zoom: { type: Number, default: 6 },
  overlayOpacity: { type: Number, default: 0.7 },
})

const mapContainer = ref(null)
const loading = ref(false)
const loadedCount = ref(0)
const totalCount = ref(0)

let map = null
// Array of Leaflet ImageOverlay layers (one per frame)
let frameLayers = []
let activeFrameIndex = -1
// Generation counter: incremented on each preloadFrames call so that
// stale (superseded) preloads discard their results instead of leaking layers.
let preloadGeneration = 0

// Radar overlay bounds (from maps.py in the Streamlit app)
const RADAR_BOUNDS = [
  [35.0623, 4.51987],   // Southwest corner
  [47.5730, 20.4801],   // Northeast corner
]

// Italian radar station positions [name, lat, lon]
const RADARS = [
  ['BRIC', 45.034, 7.734],
  ['CAPOCACCIA', 40.5671, 8.1588],
  ['CAPOFIUME', 44.65, 11.62],
  ['CROCIONE', 43.9615, 10.6103],
  ['FIUMICINO', 41.9105, 12.2344],
  ['FOSSALON', 45.72, 13.47],
  ['GATTATICO', 44.79, 10.51],
  ['GRANDE', 45.36, 11.67],
  ['IL MONTE', 41.9394, 14.6208],
  ['LAURO', 37.1126, 14.8357],
  ['LINATE', 45.3437, 9.28837],
  ['MACAION', 46.49, 11.21],
  ['MIDIA', 42.0578, 13.1772],
  ['PETTINASCURA', 39.3698, 16.6183],
  ['RASU', 40.42, 9.01],
  ['SAGITTARIA', 45.69, 12.79],
  ['SERANO', 42.8659, 12.8002],
  ['SETTEPANI', 44.247, 8.199],
  ['XBAND1', 40.883, 14.286],
  ['XBAND2', 38.07, 15.65],
  ['XBAND3', 41.139, 16.76],
  ['XBAND4', 37.462, 15.05],
  ['ZOUFPLAN', 46.5625, 12.9703],
  ['ARMIDDA', 39.8822, 9.4937],
  ['DESIO', 45.6273, 9.1963],
  ['FLERO', 45.4814, 10.1768],
]

// Base map tile layers
const BASE_MAPS = {
  'Dark': L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 20,
  }),
  'OpenStreetMap': L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    maxZoom: 19,
  }),
  'Satellite': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: '&copy; <a href="https://www.esri.com/">Esri</a>',
    maxZoom: 18,
  }),
  'Terrain': L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://opentopomap.org">OpenTopoMap</a>',
    maxZoom: 17,
  }),
}

onMounted(() => {
  map = L.map(mapContainer.value, {
    center: props.center,
    zoom: props.zoom,
    zoomControl: true,
    attributionControl: true,
    zoomAnimation: true,
  })

  BASE_MAPS['Dark'].addTo(map)

  L.control.layers(BASE_MAPS, null, {
    position: 'topright',
    collapsed: true,
  }).addTo(map)

  // Place search bar — uses OpenStreetMap's free Nominatim geocoder
  const searchControl = new GeoSearchControl({
    provider: new OpenStreetMapProvider(),
    style: 'bar',
    position: 'topright',
    showMarker: true,
    showPopup: false,
    autoClose: true,
    retainZoomLevel: false,
    animateZoom: true,
    searchLabel: 'Search place...',
  })
  map.addControl(searchControl)

  // Add radar station markers with custom icon and tooltip on hover
  const radarIcon = L.icon({
    iconUrl: '/radar.png',
    iconSize: [22, 22],
    iconAnchor: [11, 11],
    tooltipAnchor: [0, -14],
  })

  for (const [name, lat, lon] of RADARS) {
    L.marker([lat, lon], { icon: radarIcon, interactive: true })
      .bindTooltip(name, {
        permanent: false,
        direction: 'top',
        className: 'radar-tooltip',
      })
      .addTo(map)
  }
})

onUnmounted(() => {
  clearFrames()
  if (map) {
    map.remove()
    map = null
  }
})

/**
 * Remove all existing overlay layers from the map.
 */
function clearFrames() {
  for (const layer of frameLayers) {
    if (map && layer) map.removeLayer(layer)
  }
  frameLayers = []
  activeFrameIndex = -1
}

/**
 * Preload all frame URLs in parallel.
 * Creates a hidden ImageOverlay for each frame, then shows the first one.
 *
 * HOW IT WORKS:
 * 1. For each URL, we create a browser Image() to download it in the background
 * 2. Once downloaded, we create a Leaflet ImageOverlay (invisible initially)
 * 3. When all are loaded, we show frame 0
 *
 * This means: one burst of 12 HTTP requests, then instant switching forever.
 */
async function preloadFrames(urls) {
  if (!map) return

  // Bump generation so any in-flight preload knows it's been superseded
  const thisGeneration = ++preloadGeneration

  clearFrames()
  loading.value = true
  loadedCount.value = 0
  totalCount.value = urls.length

  // Download all images in parallel using browser Image objects
  const promises = urls.map((url, index) => {
    return new Promise((resolve) => {
      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.onload = () => {
        loadedCount.value++
        resolve({ index, url, success: true })
      }
      img.onerror = () => {
        loadedCount.value++
        resolve({ index, url, success: false })
      }
      img.src = url
    })
  })

  const results = await Promise.all(promises)

  // If a newer preloadFrames call was made while we were downloading,
  // discard these results — the newer call owns the map now.
  if (thisGeneration !== preloadGeneration) return

  // Create Leaflet ImageOverlay layers for each successfully loaded frame
  // All layers are added to the map but set to opacity 0 (invisible)
  frameLayers = new Array(urls.length).fill(null)

  for (const result of results) {
    if (result.success) {
      const layer = L.imageOverlay(result.url, RADAR_BOUNDS, {
        opacity: 0,  // Start invisible
        interactive: false,
      }).addTo(map)
      frameLayers[result.index] = layer
    }
  }

  loading.value = false

  // Show the first frame
  if (frameLayers.length > 0) {
    showFrame(0)
  }
}

/**
 * Show a specific frame by index. Hides all others.
 * Uses the overlayOpacity prop for the visible frame's opacity.
 * The CSS transition on .leaflet-image-layer creates a smooth cross-fade.
 */
function showFrame(index) {
  if (index < 0 || index >= frameLayers.length) return

  // Hide the previously active frame
  if (activeFrameIndex >= 0 && frameLayers[activeFrameIndex]) {
    frameLayers[activeFrameIndex].setOpacity(0)
  }

  // Show the requested frame at the current overlay opacity
  if (frameLayers[index]) {
    frameLayers[index].setOpacity(props.overlayOpacity)
  }

  activeFrameIndex = index
}

/**
 * Update the opacity of the currently visible frame.
 * Called when the user drags the opacity slider.
 */
function setOverlayOpacity(opacity) {
  if (activeFrameIndex >= 0 && frameLayers[activeFrameIndex]) {
    frameLayers[activeFrameIndex].setOpacity(opacity)
  }
}

function invalidateSize() {
  if (map) {
    nextTick(() => map.invalidateSize())
  }
}

// Expose methods for the parent component to call
defineExpose({ preloadFrames, showFrame, setOverlayOpacity, invalidateSize })
</script>

<style>
.leaflet-container {
  background: #1a1a2e;
}

/*
  image-rendering: pixelated — tells the browser NOT to smooth/blur
  the radar image when zoomed in. Instead you see crisp grid cells.
  This preserves the true 1400x1200 resolution of the radar data.

  Without this: zooming in shows blurry interpolated colors
  With this: zooming in shows sharp individual pixels (1km grid cells)
*/
.leaflet-image-layer {
  image-rendering: pixelated;
  /* No transition — instant frame switching for snappy slider/animation */
}

/* Style the geosearch bar to match the dark map theme */
.leaflet-control-geosearch {
  width: 260px;
}
.leaflet-control-geosearch form {
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 8px;
  padding: 0;
}
.leaflet-control-geosearch form input {
  color: white;
  background: transparent;
  font-size: 13px;
  padding: 8px 12px;
  outline: none;
  min-height: unset;
  height: auto;
}
.leaflet-control-geosearch form input::placeholder {
  color: rgba(255, 255, 255, 0.5);
}
.leaflet-control-geosearch .results {
  background: rgba(0, 0, 0, 0.85);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-top: none;
  border-radius: 0 0 8px 8px;
}
.leaflet-control-geosearch .results > * {
  color: white;
  font-size: 12px;
  padding: 6px 12px;
  border: none;
}
.leaflet-control-geosearch .results > *:hover {
  background: rgba(255, 255, 255, 0.1);
}

/* Radar station tooltip — dark style matching the map theme */
.radar-tooltip {
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(4px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 6px;
  color: white;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.5px;
  padding: 4px 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
}
.radar-tooltip::before {
  border-top-color: rgba(0, 0, 0, 0.8) !important;
}
</style>