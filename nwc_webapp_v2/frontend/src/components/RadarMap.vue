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
import { ref, watch, onMounted, onUnmounted, nextTick } from 'vue'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import 'leaflet-velocity/dist/leaflet-velocity.css'
import 'leaflet-velocity'
import { GeoSearchControl, OpenStreetMapProvider } from 'leaflet-geosearch'
import 'leaflet-geosearch/dist/geosearch.css'
import { useSettingsStore } from '../stores/settings.js'

const props = defineProps({
  center: { type: Array, default: () => [42.0, 12.5] },
  zoom: { type: Number, default: 6 },
  overlayOpacity: { type: Number, default: 0.7 },
  // Optional: override the default Italian-radar bounds for the image overlays.
  // Pass [[lat_sw, lon_sw], [lat_ne, lon_ne]].  When null, RADAR_BOUNDS is used.
  overlayBounds: { type: Array, default: null },
})

// Emit click events with the clicked lat/lon so parents can show pixel-value popups.
const emit = defineEmits(['mapclick'])

const mapContainer = ref(null)
const loading = ref(false)
const loadedCount = ref(0)
const totalCount = ref(0)

const settings = useSettingsStore()

let map = null
let activeBaseLayer = null
// Array of Leaflet ImageOverlay layers (one per frame) — used by single-product API
let frameLayers = []
let activeFrameIndex = -1
// Generation counter: incremented on each preloadFrames/loadProductFrames call so that
// stale (superseded) preloads discard their results instead of leaking layers.
let preloadGeneration = 0

/** Return the bounds to use for ImageOverlay creation. */
function effectiveBounds() {
  return props.overlayBounds || RADAR_BOUNDS
}

// Radar station markers keyed by site name so icons can be updated per-frame
let radarMarkers = {}

/**
 * Build a DivIcon for a radar site using status-specific PNG assets.
 * status: 'active' → radar_avail.png | 'inactive' → radar_not_avail.png | 'unknown' → radar.png (white)
 */
function makeRadarDivIcon(status) {
  const src = status === 'active'   ? '/radar_avail.png'
            : status === 'inactive' ? '/radar_not_avail.png'
            : '/radar.png'
  const style = status === 'unknown'
    ? 'width:22px;height:22px;display:block;filter:brightness(0) invert(1)'
    : 'width:22px;height:22px;display:block'
  return L.divIcon({
    html: `<img src="${src}" style="${style}">`,
    className: 'radar-status-icon',
    iconSize: [22, 22],
    iconAnchor: [11, 11],
    tooltipAnchor: [0, -14],
  })
}

// Normalize site name for comparison: remove spaces, uppercase.
// Needed because the FTP file uses "ILMONTE" while RADARS has "IL MONTE".
function _normalizeRadarName(name) {
  return name.replace(/\s+/g, '').toUpperCase()
}

// Multi-product layer map for DataExplorer:
// { productKey: { layers: [...ImageOverlay|null], activeIndex: number, opacity: number } }
let productLayerMap = {}

// Per-product generation counters — independent of the single-product counter.
// Allows loading 4 products in parallel without cancelling each other.
let productGenerations = {}

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
  // Position read from the Torchiarolo HDF5 (where/projdef: +lat_0 / +lon_0)
  ['TORCHIAROLO', 40.5064, 18.0598],
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

  // Use the saved setting (falls back to Dark if key not found)
  const initialLayer = BASE_MAPS[settings.baseLayer] ?? BASE_MAPS['Dark']
  initialLayer.addTo(map)
  activeBaseLayer = initialLayer

  L.control.layers(BASE_MAPS, null, {
    position: 'topleft',
    collapsed: true,
  }).addTo(map)

  // Sync when user switches base layer in Settings modal
  watch(() => settings.baseLayer, (name) => {
    const next = BASE_MAPS[name]
    if (!next || !map) return
    if (activeBaseLayer) map.removeLayer(activeBaseLayer)
    next.addTo(map)
    activeBaseLayer = next
    // dark-basemap class drives colorbar/overlay CSS
    if (name === 'Dark') {
      mapContainer.value?.classList.add('dark-basemap')
    } else {
      mapContainer.value?.classList.remove('dark-basemap')
    }
  })

  // Place search bar — uses OpenStreetMap's free Nominatim geocoder
  const searchControl = new GeoSearchControl({
    provider: new OpenStreetMapProvider(),
    style: 'bar',
    position: 'topleft',
    showMarker: true,
    showPopup: false,
    autoClose: true,
    retainZoomLevel: false,
    animateZoom: true,
    searchLabel: 'Search place...',
  })
  map.addControl(searchControl)

  // Add radar station markers — stored by name so status colors can be updated later
  for (const [name, lat, lon] of RADARS) {
    radarMarkers[name] = L.marker([lat, lon], { icon: makeRadarDivIcon('unknown'), interactive: true })
      .bindTooltip(name, {
        permanent: false,
        direction: 'top',
        className: 'radar-tooltip',
      })
      .addTo(map)
  }

  const DARK_LAYERS = new Set(['Dark', 'Satellite'])
  if (DARK_LAYERS.has(settings.baseLayer)) {
    mapContainer.value.classList.add('dark-basemap')
  }

  // Click → emit (lat, lon) so the parent view can fetch pixel values and
  // display a popup. Skip clicks on existing popups, controls, or markers.
  map.on('click', (e) => {
    emit('mapclick', { lat: e.latlng.lat, lng: e.latlng.lng })
  })

  map.on('baselayerchange', (e) => {
    if (DARK_LAYERS.has(e.name)) {
      mapContainer.value.classList.add('dark-basemap')
    } else {
      mapContainer.value.classList.remove('dark-basemap')
    }
  })
})

onUnmounted(() => {
  clearFrames()
  radarMarkers = {}
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
      const layer = L.imageOverlay(result.url, effectiveBounds(), {
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

/**
 * Bring all single-product (frameLayers) overlays to the front.
 * Call this after loading any multi-product layers to ensure SRI stays on top.
 */
function bringFramesToFront() {
  for (const layer of frameLayers) {
    if (layer) layer.bringToFront()
  }
}

function invalidateSize() {
  if (map) {
    nextTick(() => map.invalidateSize())
  }
}

// ==========================================================================
// Click-to-inspect popup
// ==========================================================================

let activePopup = null

/**
 * Open a Leaflet popup at (lat, lng) showing arbitrary HTML content.
 * Closes any previously-open popup. Pass html='' to just close.
 */
function showPopup(latlng, html, options = {}) {
  if (!map) return
  if (activePopup) {
    map.closePopup(activePopup)
    activePopup = null
  }
  if (!html) return
  activePopup = L.popup({
    closeButton: true,
    autoClose: true,
    className: 'pixel-inspect-popup',
    maxWidth: 280,
    ...options,
  })
    .setLatLng(latlng)
    .setContent(html)
    .openOn(map)
}

function closePopup() {
  if (map && activePopup) {
    map.closePopup(activePopup)
    activePopup = null
  }
}

// ==========================================================================
// Multi-product API (used by DataExplorerView)
// ==========================================================================

/**
 * Remove all existing overlay layers for a specific product.
 */
function removeProduct(product) {
  if (!productLayerMap[product]) return
  for (const layer of productLayerMap[product].layers) {
    if (map && layer) map.removeLayer(layer)
  }
  delete productLayerMap[product]
}

/**
 * Remove all product layers from the map (full reset for multi-product view).
 */
function clearAllProducts() {
  for (const product of Object.keys(productLayerMap)) {
    removeProduct(product)
  }
  // Increment every known generation so any in-flight loadProductFrames call
  // sees myGen !== current and discards its results.  Resetting to {} instead
  // would give both the old and new load the same generation (1), making the
  // stale-load check useless and leaving removed layers in productLayerMap.
  for (const key of Object.keys(productGenerations)) {
    productGenerations[key]++
  }
}

/**
 * Preload all frames for a single product in parallel.
 * Creates hidden ImageOverlay layers — call showAllAtFrame() to display them.
 *
 * @param {string} product - Product key (e.g. 'SRI_adj', 'VMI')
 * @param {string[]} urls - Array of image URLs, one per unified timestamp
 * @param {number} opacity - Initial opacity for visible frames
 * @param {Array|null} bounds - Optional [[lat_sw,lon_sw],[lat_ne,lon_ne]].
 *   When null, falls back to props.overlayBounds or the default Italian radar bounds.
 *   Pass explicit bounds when a product covers a different area than the default
 *   (e.g. WR10 local radar vs. the full national mosaic).
 */
async function loadProductFrames(product, urls, opacity = 0.7, bounds = null) {
  if (!map) return

  // Use per-product generation so parallel loads of different products
  // don't cancel each other (bug: shared preloadGeneration did exactly that).
  if (!productGenerations[product]) productGenerations[product] = 0
  const myGen = ++productGenerations[product]

  removeProduct(product)

  const layers = new Array(urls.length).fill(null)
  const productBounds = bounds || effectiveBounds()

  const promises = urls.map((url, index) => {
    if (!url) return Promise.resolve({ index, success: false })
    return new Promise((resolve) => {
      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.onload = () => resolve({ index, url, success: true })
      img.onerror = () => resolve({ index, url, success: false })
      img.src = url
    })
  })

  const results = await Promise.all(promises)
  // Only cancel if THIS product's load was superseded — not other products
  if (myGen !== productGenerations[product]) return

  for (const result of results) {
    if (result.success) {
      layers[result.index] = L.imageOverlay(result.url, productBounds, {
        opacity: 0,
        interactive: false,
      }).addTo(map)
    }
  }

  productLayerMap[product] = { layers, activeIndex: -1, opacity, bounds: productBounds }
}

/**
 * Show a specific frame for all loaded products simultaneously.
 * Hides the previous frame for each product.
 *
 * @param {number} frameIndex - Index into the unified timestamp array
 * @param {Object|null} opacities - Optional { product: opacity } map.
 *   If provided, each product uses its entry from this map instead of
 *   entry.opacity. Pass 0 to hide a disabled product without removing
 *   its layers.
 */
function showAllAtFrame(frameIndex, opacities = null) {
  for (const [product, entry] of Object.entries(productLayerMap)) {
    // Hide current frame
    if (entry.activeIndex >= 0 && entry.layers[entry.activeIndex]) {
      entry.layers[entry.activeIndex].setOpacity(0)
    }
    // Determine target opacity
    const targetOpacity = opacities != null
      ? (opacities[product] ?? entry.opacity)
      : entry.opacity
    // Show new frame
    if (frameIndex >= 0 && frameIndex < entry.layers.length && entry.layers[frameIndex]) {
      entry.layers[frameIndex].setOpacity(targetOpacity)
    }
    entry.activeIndex = frameIndex
  }
}

/**
 * Update the opacity of the currently visible frame for a product.
 *
 * @param {string} product - Product key
 * @param {number} opacity - New opacity value (0–1)
 */
function setProductOpacity(product, opacity) {
  if (!productLayerMap[product]) return
  const entry = productLayerMap[product]
  entry.opacity = opacity
  if (entry.activeIndex >= 0 && entry.layers[entry.activeIndex]) {
    entry.layers[entry.activeIndex].setOpacity(opacity)
  }
}

/**
 * Remove the oldest N frames for a product (sliding window).
 * Removes layers from the front of the array and adjusts the active index.
 * Called after appending new frames so the window stays fixed-size.
 *
 * @param {string} product - Product key
 * @param {number} count   - Number of frames to drop from the front
 */
function trimProductFrames(product, count) {
  if (!productLayerMap[product] || count <= 0) return
  const entry = productLayerMap[product]
  if (count >= entry.layers.length) return  // safety: don't trim everything

  const removed = entry.layers.splice(0, count)
  for (const layer of removed) {
    if (map && layer) map.removeLayer(layer)
  }
  // Shift the active-frame pointer; -1 means "no frame currently visible"
  entry.activeIndex = Math.max(-1, entry.activeIndex - count)
}

/**
 * Append new frames to an already-loaded product without clearing existing layers.
 * Used by LiveView for efficient polling: only the newly-arrived frames are fetched,
 * existing frames remain on the map untouched.
 *
 * @param {string} product - Product key (e.g. 'SRI_adj')
 * @param {(string|null)[]} urls - URLs for the new frames to append (null = missing frame)
 */
async function appendProductFrames(product, urls) {
  if (!map || !productLayerMap[product] || urls.length === 0) return

  const entry = productLayerMap[product]
  const startIndex = entry.layers.length

  // Reserve slots in the array immediately so indices are stable
  for (let i = 0; i < urls.length; i++) {
    entry.layers.push(null)
  }

  const promises = urls.map((url, i) => {
    if (!url) return Promise.resolve()
    const globalIndex = startIndex + i
    return new Promise((resolve) => {
      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.onload = () => {
        if (productLayerMap[product]) {
          entry.layers[globalIndex] = L.imageOverlay(url, entry.bounds || effectiveBounds(), {
            opacity: 0,
            interactive: false,
          }).addTo(map)
        }
        resolve()
      }
      img.onerror = () => resolve()
      img.src = url
    })
  })

  await Promise.all(promises)
}

/**
 * Resolve a previously-missing frame: load a real image into an existing null slot.
 * Called when a file that was initially absent is later found by the poll.
 *
 * @param {string} product - Product key
 * @param {number} index   - Frame index in the product's layer array
 * @param {string} url     - Image URL to load into that slot
 */
async function resolveProductFrame(product, index, url) {
  if (!map || !productLayerMap[product] || !url) return false
  const entry = productLayerMap[product]
  if (index < 0 || index >= entry.layers.length) return false
  if (entry.layers[index]) {
    map.removeLayer(entry.layers[index])
    entry.layers[index] = null
  }
  // Retry up to 3 times — the render endpoint can transiently fail when a file
  // is still being written to disk when the first request arrives.
  for (let attempt = 0; attempt < 3; attempt++) {
    if (attempt > 0) await new Promise(r => setTimeout(r, 800))
    // Cache-bust on retries so the browser doesn't serve a cached error response
    const src = attempt > 0 ? `${url}&_r=${attempt}` : url
    const ok = await new Promise((resolve) => {
      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.onload = () => {
        if (productLayerMap[product]) {
          entry.layers[index] = L.imageOverlay(src, entry.bounds || effectiveBounds(), {
            opacity: 0,
            interactive: false,
          }).addTo(map)
        }
        resolve(true)
      }
      img.onerror = () => resolve(false)
      img.src = src
    })
    if (ok) return true
  }
  return false
}

/**
 * Set the visual stacking order of products on the map.
 *
 * @param {string[]} topToBottom - Products ordered from topmost (index 0) to bottommost (last).
 *
 * Leaflet stacks layers by DOM insertion order (last child = rendered on top).
 * bringToFront() moves a layer's DOM elements to the end of the container pane,
 * making it appear above all others at that point.
 *
 * Strategy: iterate from BOTTOM to TOP, calling bringToFront on every frame of each
 * product in that order. The topmost product is called last, so it wins the "front"
 * position. All frames are reordered (not just the active one) so that navigating the
 * timeline always respects the chosen stacking order.
 */
function setProductOrder(topToBottom) {
  if (!map) return
  // Iterate from last (bottommost) to first (topmost)
  for (let i = topToBottom.length - 1; i >= 0; i--) {
    const entry = productLayerMap[topToBottom[i]]
    if (!entry) continue
    for (const layer of entry.layers) {
      if (layer) layer.bringToFront()
    }
  }
}

// ---- Wind / AMV velocity layer ----
let velocityLayer = null

// ---- LK arrow image overlay ----
// Bounds match lk_config.yaml output_grid exactly: [[lat_sw, lon_sw], [lat_ne, lon_ne]]
const LK_ARROW_BOUNDS = [[35.5, 4.5], [47.75, 19.5]]
let lkImageLayer     = null
let _lkTargetOpacity = 0.8   // tracks the desired opacity so load/error handlers stay in sync

/**
 * Render (or update) the leaflet-velocity wind layer with new data.
 * `velocityData` is the two-element array [U, V] from /api/wind/data.
 */
function setWindLayer(velocityData) {
  if (!map) return
  if (velocityLayer) {
    // Update in-place: setData clears the canvas and rebuilds the Windy grid
    // without removing/re-adding the DOM element, so the blank is much shorter
    // than a full destroy+recreate cycle.
    velocityLayer.setData(velocityData)
    return
  }
  velocityLayer = L.velocityLayer({
    displayValues: true,
    displayOptions: {
      velocityType: 'Wind',
      position: 'bottomleft',
      emptyString: 'No wind data',
      angleConvention: 'bearingCW',
      speedUnit: 'm/s',
    },
    data: velocityData,
    maxVelocity: 25,
    velocityScale: 0.005,
  })
  velocityLayer.addTo(map)
}

/** Remove the wind layer (called when the user toggles it off). */
function clearWindLayer() {
  if (!map || !velocityLayer) return
  map.removeLayer(velocityLayer)
  velocityLayer = null
}

/**
 * Show (or update) the LK quiver-arrow PNG as an ImageOverlay.
 *
 * The layer starts invisible (opacity 0) and becomes visible only once the
 * image actually loads. If the PNG is missing (404), the 'error' event fires
 * and we keep it hidden — no broken-image placeholder appears on the map.
 */
function setLkImage(url, opacity = 0.8) {
  if (!map) return
  _lkTargetOpacity = opacity

  if (lkImageLayer) {
    // Hide while the new URL is loading; load/error handlers will restore opacity.
    lkImageLayer.setOpacity(0)
    lkImageLayer.setUrl(url)
    return
  }

  // Create hidden; attach load/error before adding to map so no frame is missed.
  lkImageLayer = L.imageOverlay(url, LK_ARROW_BOUNDS, {
    opacity: 0,
    interactive: false,
  })
  lkImageLayer.on('load',  () => lkImageLayer?.setOpacity(_lkTargetOpacity))
  lkImageLayer.on('error', () => lkImageLayer?.setOpacity(0))   // 404 or network error → stay invisible
  lkImageLayer.addTo(map)
}

/** Remove the LK arrow overlay (called when toggling off arrows or changing mode). */
function clearLkImage() {
  if (!map || !lkImageLayer) return
  map.removeLayer(lkImageLayer)
  lkImageLayer = null
}

/**
 * Update radar marker icon colors to reflect availability at the current frame.
 *
 * @param {string[]|null} activeNames - List of active site names from the status file,
 *   or null when no status data is available (all markers revert to gray).
 *
 * Name comparison is normalized (spaces stripped, uppercase) so that "IL MONTE"
 * in RADARS matches "ILMONTE" in the FTP status files.
 */
function updateRadarStatus(activeNames) {
  if (!activeNames) {
    for (const marker of Object.values(radarMarkers)) {
      marker.setIcon(makeRadarDivIcon('unknown'))
    }
    return
  }
  const activeSet = new Set(activeNames.map(_normalizeRadarName))
  for (const [name, marker] of Object.entries(radarMarkers)) {
    const status = activeSet.has(_normalizeRadarName(name)) ? 'active' : 'inactive'
    marker.setIcon(makeRadarDivIcon(status))
  }
}

/**
 * Add a one-off marker that is NOT tracked in radarMarkers, so it is unaffected
 * by updateRadarStatus() calls. Returns the Leaflet marker instance.
 */
function addFixedMarker(lat, lon, status = 'active', tooltip = '') {
  if (!map) return null
  const marker = L.marker([lat, lon], { icon: makeRadarDivIcon(status), interactive: !!tooltip })
  if (tooltip) {
    marker.bindTooltip(tooltip, { permanent: false, direction: 'top', className: 'radar-tooltip', offset: [0, -14] })
  }
  marker.addTo(map)
  return marker
}

// Expose methods for the parent component to call
defineExpose({
  // Single-product (backward compat — used by RealTimeView)
  preloadFrames, showFrame, setOverlayOpacity, bringFramesToFront,
  // Multi-product (used by DataExplorerView and LiveView)
  loadProductFrames, appendProductFrames, trimProductFrames, resolveProductFrame,
  showAllAtFrame, setProductOpacity, removeProduct, clearAllProducts, setProductOrder,
  // Utility
  invalidateSize, showPopup, closePopup,
  // Wind / AMV
  setWindLayer, clearWindLayer,
  // LK arrow image overlay
  setLkImage, clearLkImage,
  // Radar status coloring
  updateRadarStatus,
  // Fixed markers (not affected by updateRadarStatus)
  addFixedMarker,
})
</script>

<style>
.leaflet-container {
  background: #1a1a2e;
}

/* Mobile: hide zoom +/- buttons (pinch-zoom is more natural on touch screens
   and the buttons collide with the sidebar toggle in the top-left corner). */
@media (max-width: 1023px) {
  .leaflet-control-zoom {
    display: none !important;
  }
}

/* The layers control and the geosearch bar share the top-left corner.
   Without explicit z-index they stack by DOM order and the geosearch form
   (which has a backdrop-filter, creating its own stacking context) wins
   over the expanded layers panel. Force layers above, geosearch below. */
.leaflet-control-layers {
  position: relative;
  z-index: 1100 !important;
}
.leaflet-control-geosearch {
  position: relative;
  z-index: 900;
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

/* Invert radar icons to white on dark/satellite base maps (legacy class, kept for safety) */
.dark-basemap .radar-icon {
  filter: invert(1);
}

/* Strip Leaflet's default white box around divIcon wrappers used for radar status markers */
.radar-status-icon {
  background: none !important;
  border: none !important;
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

/* Pixel-inspect popup — dark theme matching the rest of the UI. */
.pixel-inspect-popup .leaflet-popup-content-wrapper {
  background: rgba(15, 23, 42, 0.95);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 8px;
  color: white;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
}
.pixel-inspect-popup .leaflet-popup-content {
  margin: 10px 12px;
  font-size: 12px;
  line-height: 1.45;
}
.pixel-inspect-popup .leaflet-popup-tip {
  background: rgba(15, 23, 42, 0.95);
}
.pixel-inspect-popup .leaflet-popup-close-button {
  color: rgba(255, 255, 255, 0.7) !important;
  padding: 6px 8px 0 0 !important;
}
.pixel-inspect-popup .pi-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}
.pixel-inspect-popup .pi-label {
  color: rgba(255, 255, 255, 0.6);
}
.pixel-inspect-popup .pi-value {
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.pixel-inspect-popup .pi-header {
  margin-bottom: 6px;
  padding-bottom: 6px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  font-size: 11px;
  color: rgba(255, 255, 255, 0.55);
}
</style>