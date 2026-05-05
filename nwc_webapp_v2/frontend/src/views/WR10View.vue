<!--
  WR10View.vue — Real-time monitor for the WR10 small X-band radar.

  Shows SRI and VMI products from the local WR10 radar for a rolling lookback
  window.  Auto-loads on mount, polls every 5 minutes for new data, and also
  receives instant push notifications via WebSocket (/api/wr10/ws).

  Architecture mirrors RealTimeView but is simpler: no model predictions,
  no complex search-window logic — just watch the folder and show what's there.
-->
<template>
  <div class="h-[calc(100dvh-3rem)] sm:h-[calc(100vh-3.5rem)] flex overflow-hidden">

    <!-- ================================================================ -->
    <!-- LEFT: Map area                                                    -->
    <!-- ================================================================ -->
    <div class="flex-1 flex flex-col relative min-w-0">
      <RadarMap
        ref="radarMap"
        :center="radarCenter"
        :zoom="radarZoom"
        :overlay-bounds="overlayBounds"
        class="flex-1"
        @mapclick="onMapClick"
      />

      <!-- Mobile sidebar toggle -->
      <button
        v-if="!sidebarOpen"
        @click="sidebarOpen = true"
        class="absolute top-3 right-3 z-[1001]
               flex items-center gap-1.5 px-3 h-9 rounded-full
               bg-white shadow-lg border border-gray-200 text-gray-600
               hover:bg-gray-50 transition-colors"
        title="Open panel"
      >
        <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path d="M4 6h16M4 12h16M4 18h16" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
        <span class="text-sm font-medium hidden sm:inline">Menu</span>
      </button>

      <!-- Loading indicator — visible on the map when sidebar is closed -->
      <div
        v-if="isLoading && !sidebarOpen"
        class="absolute top-14 right-3 z-[1000] flex items-center gap-2 px-3 h-9 rounded-full
               bg-white/90 shadow-lg border border-gray-200 text-gray-600 text-sm"
      >
        <svg class="animate-spin h-4 w-4 text-blue-500 flex-shrink-0" viewBox="0 0 24 24" fill="none">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <span class="text-xs font-medium">Loading…</span>
      </div>

      <!-- Colorbars -->
      <div
        v-if="settings.showColorbars"
        class="absolute right-[10px] bottom-24 z-[1001] flex flex-col gap-1.5 items-end"
      >
        <ColorBar
          v-for="product in colorbarsToShow"
          :key="product"
          :legend="productMeta[product]"
          :product-name="PRODUCT_LABELS[product] || product"
        />
      </div>

      <!-- ============================================================ -->
      <!-- BOTTOM: Timeline controls                                     -->
      <!-- ============================================================ -->
      <div
        class="absolute bottom-0 left-0 right-0 z-[1000]
               bg-gradient-to-t from-black/80 via-black/60 to-transparent
               px-3 sm:px-6 pt-6 sm:pt-10
               pb-[calc(1rem+env(safe-area-inset-bottom))] sm:pb-4"
        :class="{ 'pointer-events-none opacity-40': !isLoaded }"
      >
        <div class="flex items-center justify-between text-white mb-2">
          <div class="text-xs font-medium text-gray-300 hidden sm:block truncate max-w-[160px]">
            {{ visibleProducts.map(p => PRODUCT_LABELS[p]).join(' + ') || '—' }}
          </div>
          <div class="text-center">
            <span class="text-xs sm:text-xl font-bold tabular-nums tracking-tight">
              {{ currentTimestampDisplay }}
            </span>
          </div>
          <div class="hidden sm:block max-w-[160px] flex-1" />
        </div>

        <div class="flex items-start gap-3">
          <button
            @click="togglePlay"
            :disabled="!isLoaded || timestamps.length === 0"
            class="w-9 h-9 flex items-center justify-center rounded-full flex-shrink-0
                   bg-white/10 hover:bg-white/20 border border-white/20 text-white transition-colors
                   disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <svg v-if="!isPlaying" class="w-4 h-4 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
            <svg v-else class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
            </svg>
          </button>

          <div class="flex-1 min-w-0">
            <div class="h-9 flex items-center">
              <input
                type="range"
                :min="0"
                :max="Math.max(0, timestamps.length - 1)"
                :value="frameIndex"
                @input="onSliderInput"
                :disabled="!isLoaded || timestamps.length === 0"
                class="w-full h-1.5 rounded-full appearance-none cursor-pointer timeline-slider
                       disabled:opacity-40 disabled:cursor-not-allowed"
              />
            </div>
            <div v-if="hourTicks.length" class="relative h-4">
              <span
                v-for="tick in hourTicks"
                :key="tick.pct"
                class="absolute text-[9px] text-gray-400 -translate-x-1/2"
                :style="{ left: tick.pct + '%' }"
              >{{ tick.label }}</span>
            </div>
          </div>

          <div class="h-9 flex items-center flex-shrink-0">
            <button
              @click="cycleSpeed"
              class="px-3 py-1.5 rounded-full bg-white/20 hover:bg-white/30
                     text-white text-xs font-medium transition-colors backdrop-blur-sm
                     tabular-nums min-w-[44px]"
              title="Animation speed"
            >
              {{ playSpeed }}×
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- ================================================================ -->
    <!-- RIGHT: Sidebar                                                    -->
    <!-- ================================================================ -->
    <div
      v-if="sidebarOpen"
      class="fixed inset-0 bg-black/40 z-[1100] lg:hidden"
      @click="sidebarOpen = false"
    />

    <div
      class="bg-gray-900 flex-shrink-0 overflow-hidden
             fixed right-0 top-12 sm:top-14 bottom-0 z-[1101] w-72
             lg:relative lg:top-auto lg:right-auto lg:bottom-auto lg:z-auto
             transition-all duration-200 ease-out"
      :class="sidebarOpen
        ? 'translate-x-0 border-l border-gray-700 lg:w-72'
        : 'translate-x-full lg:translate-x-0 lg:w-0 border-l-0'"
    >
      <div class="w-72 h-full flex flex-col overflow-y-auto">
        <!-- Close button -->
        <button
          @click="sidebarOpen = false"
          class="absolute top-2 right-2 w-8 h-8 flex items-center justify-center
                 rounded-full text-gray-400 hover:text-gray-200 hover:bg-white/10 transition-colors"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path d="M6 18L18 6M6 6l12 12" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </button>

        <div class="p-4 space-y-5">

          <!-- Title -->
          <div class="pt-1">
            <h2 class="text-white font-bold text-base">WR10 Radar</h2>
            <p class="text-gray-400 text-xs mt-0.5">Local X-band radar monitor</p>
          </div>

          <!-- Live status card -->
          <div class="bg-gray-800 rounded-lg p-3 space-y-2.5">
            <div class="flex items-center gap-2">
              <div
                class="w-2 h-2 rounded-full flex-shrink-0"
                :class="isLoaded ? 'bg-green-400' : isLoading ? 'bg-yellow-400 animate-pulse' : 'bg-gray-500'"
              />
              <span class="text-xs text-gray-300">{{ statusText }}</span>
              <span
                :title="wsConnected ? 'WebSocket connected — instant updates' : 'WebSocket offline — using 5-min poll'"
                class="ml-auto flex items-center gap-0.5"
              >
                <span class="w-1.5 h-1.5 rounded-full" :class="wsConnected ? 'bg-green-400' : 'bg-gray-600'" />
                <span class="text-[10px]" :class="wsConnected ? 'text-green-400' : 'text-gray-600'">WS</span>
              </span>
            </div>

            <!-- Follow Live toggle -->
            <button
              @click="followLive = !followLive"
              class="w-full py-1.5 rounded text-xs font-semibold transition-colors border"
              :class="followLive
                ? 'bg-green-600/20 border-green-500/50 text-green-400 hover:bg-green-600/30'
                : 'bg-gray-700 border-gray-600 text-gray-400 hover:bg-gray-600'"
            >
              {{ followLive ? '● Following Live' : '○ Follow Live' }}
            </button>

            <button
              @click="goToLatest"
              :disabled="!isLoaded || timestamps.length === 0"
              class="w-full py-1.5 rounded text-xs font-semibold transition-colors
                     border border-white/10 bg-white/5 hover:bg-white/10 text-gray-300
                     disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Jump to Latest
            </button>
          </div>

          <!-- Lookback selector -->
          <div class="space-y-2">
            <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Lookback Window</h3>
            <div class="grid grid-cols-5 gap-1">
              <button
                v-for="h in [1, 2, 4, 6, 12]"
                :key="h"
                @click="setLookback(h)"
                :disabled="isLoading"
                class="py-1.5 rounded text-xs font-semibold transition-colors border disabled:cursor-not-allowed"
                :class="lookbackHours === h
                  ? 'bg-blue-600 border-blue-500 text-white'
                  : 'bg-gray-800 border-gray-700 text-gray-400 hover:bg-gray-700'"
              >
                {{ h }}h
              </button>
            </div>
          </div>

          <!-- Error + retry -->
          <div v-if="loadError && !isLoading" class="bg-red-900/30 border border-red-700/50 rounded-lg p-3 space-y-2">
            <p class="text-red-400 text-xs leading-snug">{{ loadError }}</p>
            <button
              @click="loadData()"
              class="w-full py-1.5 rounded text-xs font-semibold
                     bg-red-700/30 hover:bg-red-700/50 border border-red-600/50 text-red-300"
            >Retry</button>
          </div>

          <!-- Loading indicator -->
          <div v-if="isLoading" class="flex items-center gap-2 text-xs text-gray-400">
            <svg class="animate-spin h-4 w-4 text-blue-400" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <span>Loading frames…</span>
          </div>

          <!-- Layers -->
          <div class="space-y-2">
            <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider">WR10 Layers</h3>
            <div
              v-for="product in WR10_PRODUCTS"
              :key="product"
              class="bg-gray-800 rounded-lg p-3 space-y-2"
            >
              <div class="flex items-center gap-2">
                <input
                  type="checkbox"
                  :id="`layer-${product}`"
                  v-model="layerConfig[product].enabled"
                  class="w-4 h-4 rounded accent-blue-500 cursor-pointer flex-shrink-0"
                />
                <label :for="`layer-${product}`" class="text-white text-sm font-bold cursor-pointer flex-1">
                  {{ PRODUCT_LABELS[product] }}
                </label>
                <span class="text-gray-400 text-xs">
                  {{ productMeta[product]?.unit || 'dBZ' }}
                </span>
              </div>
              <div v-if="productFrameCount[product] > 0" class="text-xs text-green-400">
                {{ productFrameCount[product] }} frames
              </div>
              <div class="flex items-center gap-2">
                <span class="text-gray-400 text-xs w-12 flex-shrink-0">Opacity</span>
                <input
                  type="range" min="0" max="1" step="0.05"
                  v-model.number="layerConfig[product].opacity"
                  class="flex-1 h-1 accent-blue-400 cursor-pointer"
                />
                <span class="text-gray-400 text-xs w-8 text-right tabular-nums">
                  {{ Math.round(layerConfig[product].opacity * 100) }}%
                </span>
              </div>
            </div>

            <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider pt-1">Mosaic Layers</h3>
            <p class="text-gray-500 text-xs -mt-1">National HDF composite (full Italy)</p>
            <div
              v-for="product in MOSAIC_PRODUCTS"
              :key="product"
              class="bg-gray-800 rounded-lg p-3 space-y-2"
            >
              <div class="flex items-center gap-2">
                <input
                  type="checkbox"
                  :id="`layer-${product}`"
                  v-model="layerConfig[product].enabled"
                  class="w-4 h-4 rounded accent-blue-500 cursor-pointer flex-shrink-0"
                />
                <label :for="`layer-${product}`" class="text-white text-sm font-bold cursor-pointer flex-1">
                  {{ PRODUCT_LABELS[product] }}
                </label>
                <span class="text-gray-400 text-xs">
                  {{ productMeta[product]?.unit || 'dBZ' }}
                </span>
              </div>
              <div v-if="productFrameCount[product] > 0" class="text-xs text-green-400">
                {{ productFrameCount[product] }} frames
              </div>
              <div class="flex items-center gap-2">
                <span class="text-gray-400 text-xs w-12 flex-shrink-0">Opacity</span>
                <input
                  type="range" min="0" max="1" step="0.05"
                  v-model.number="layerConfig[product].opacity"
                  class="flex-1 h-1 accent-blue-400 cursor-pointer"
                />
                <span class="text-gray-400 text-xs w-8 text-right tabular-nums">
                  {{ Math.round(layerConfig[product].opacity * 100) }}%
                </span>
              </div>
            </div>
          </div>

          <!-- Radar info -->
          <div class="bg-gray-800 rounded-lg p-3 space-y-1 text-xs text-gray-400">
            <p class="font-semibold text-gray-300">Radar Info</p>
            <p>Type: X-band (WR10)</p>
            <p>Range: 72 km</p>
            <p v-if="overlayBounds">
              Lat {{ overlayBounds[0][0].toFixed(2) }}° – {{ overlayBounds[1][0].toFixed(2) }}°
            </p>
            <p v-if="overlayBounds">
              Lon {{ overlayBounds[0][1].toFixed(2) }}° – {{ overlayBounds[1][1].toFixed(2) }}°
            </p>
          </div>

        </div>
      </div>
    </div>

  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import RadarMap from '../components/RadarMap.vue'
import ColorBar from '../components/ColorBar.vue'
import { useSettingsStore } from '../stores/settings.js'
import { useConfigStore } from '../stores/config.js'
import { useWr10Ws } from '../composables/useWr10Ws.js'
import api from '../api.js'

const settings  = useSettingsStore()
const configStore = useConfigStore()

// ---- Products ----
// WR10_PRODUCTS: local X-band radar (WR10 bounds)
// MOSAIC_PRODUCTS: national mosaic from HDF files (Italy bounds)
const WR10_PRODUCTS    = ['SRI', 'VMI']
const MOSAIC_PRODUCTS  = ['SRI_MOSAIC', 'VMI_MOSAIC']
const PRODUCTS         = [...WR10_PRODUCTS, ...MOSAIC_PRODUCTS]

// Maps each mosaic layer key to the product name used by the groundtruth endpoint
const MOSAIC_API_PRODUCT = { SRI_MOSAIC: 'SRI_adj', VMI_MOSAIC: 'VMI' }

// Display labels for the timeline and layer headings
const PRODUCT_LABELS = { SRI: 'SRI', VMI: 'VMI', SRI_MOSAIC: 'SRI Mosaic', VMI_MOSAIC: 'VMI Mosaic' }

// National mosaic overlay bounds (same as the full Italian radar composite)
const ITALY_BOUNDS = [[35.0623, 4.51987], [47.5730, 20.4801]]

const layerConfig = ref({
  SRI:       { enabled: false, opacity: 0.8 },
  VMI:       { enabled: true,  opacity: 1.0 },
  SRI_MOSAIC: { enabled: false, opacity: 0.7 },
  VMI_MOSAIC: { enabled: false, opacity: 0.7 },
})

const visibleProducts = computed(() =>
  PRODUCTS.filter(p => layerConfig.value[p].enabled)
)

// SRI and SRI_MOSAIC share the same colormap; same for VMI/VMI_MOSAIC.
// Show at most one colorbar per colormap type.
const COLORBAR_PRODUCT = { SRI: 'SRI', VMI: 'VMI', SRI_MOSAIC: 'SRI', VMI_MOSAIC: 'VMI' }
const colorbarsToShow = computed(() => {
  const seen = new Set()
  const result = []
  for (const p of visibleProducts.value) {
    const key = COLORBAR_PRODUCT[p] || p
    if (!seen.has(key)) { seen.add(key); result.push(key) }
  }
  return result
})

const productMeta     = ref({})    // { SRI: { legend, label, unit }, … }
const productFrameCount = ref({ SRI: 0, VMI: 0, SRI_MOSAIC: 0, VMI_MOSAIC: 0 })

// ---- Map ----
const radarMap     = ref(null)
const radarCenter  = ref([41.842, 12.647])
const radarZoom    = ref(10)
const overlayBounds = ref(null)  // [[lat_sw, lon_sw], [lat_ne, lon_ne]]

// ---- Timeline ----
const timestamps  = ref([])
const frameIndex  = ref(0)
const isLoaded    = ref(false)
const isLoading   = ref(false)
const loadError   = ref('')
const followLive  = ref(true)

// ---- Playback ----
const SPEEDS    = [0.5, 1, 2, 4]
const playSpeed = ref(settings.defaultSpeed ?? 1)
const isPlaying = ref(false)
let   _animTimer = null

// ---- Sidebar ----
const sidebarOpen = ref(false)

// ---- Lookback ----
const lookbackHours = ref(settings.defaultLookback ?? 1)

// ---- Status ----
const statusText = computed(() => {
  if (isLoading.value) return 'Loading…'
  if (isLoaded.value && timestamps.value.length > 0) return `${timestamps.value.length} frames loaded`
  if (isLoaded.value) return 'No data found'
  return 'Idle'
})

// ---- Timestamp display ----
const currentTimestampDisplay = computed(() => {
  const ts = timestamps.value[frameIndex.value]
  if (!ts) return '—'
  try {
    const dt = new Date(ts + 'Z')
    if (settings.timeZone === 'utc') {
      return dt.toLocaleString('it-IT', { timeZone: 'UTC',
        day: '2-digit', month: '2-digit', year: 'numeric',
        hour: '2-digit', minute: '2-digit' }) + ' UTC'
    }
    return dt.toLocaleString('it-IT', { timeZone: 'Europe/Rome',
      day: '2-digit', month: '2-digit', year: 'numeric',
      hour: '2-digit', minute: '2-digit' })
  } catch {
    return ts
  }
})

// ---- Hour ticks for slider ----
const hourTicks = computed(() => {
  const n = timestamps.value.length
  if (n < 2) return []
  const ticks = []
  const first = new Date(timestamps.value[0] + 'Z')
  const last  = new Date(timestamps.value[n - 1] + 'Z')
  const totalMs = last - first
  if (totalMs <= 0) return []
  const current = new Date(first)
  current.setUTCMinutes(0, 0, 0)
  current.setUTCHours(current.getUTCHours() + 1)
  while (current <= last) {
    const pct = ((current - first) / totalMs) * 100
    ticks.push({
      pct: Math.min(100, Math.max(0, pct)),
      label: current.toLocaleString('it-IT', {
        timeZone: settings.timeZone === 'utc' ? 'UTC' : 'Europe/Rome',
        hour: '2-digit', minute: '2-digit',
      }),
    })
    current.setUTCHours(current.getUTCHours() + 1)
  }
  return ticks
})

// ---- WebSocket ----
const { connected: wsConnected } = useWr10Ws({
  onWr10Update: (data) => {
    // New file arrived — reload immediately instead of waiting for the next 5-min tick
    if (!isLoading.value) loadData()
  },
})

// ---- 5-minute clock-aligned poll ----
let _pollTimer = null

function _scheduleNextPoll() {
  clearTimeout(_pollTimer)
  const now = new Date()
  const msToNextMark = (5 * 60 * 1000) - ((now.getMinutes() % 5) * 60 + now.getSeconds()) * 1000 - now.getMilliseconds()
  // Add a small grace delay so the file has time to finish writing
  const delay = msToNextMark + 15_000
  _pollTimer = setTimeout(async () => {
    await loadData()
    _scheduleNextPoll()
  }, delay)
}

// ---- Helpers ----

/**
 * Given a sorted array of ISO timestamps and a target ISO timestamp,
 * return the largest element that is <= target, or null if none exists.
 * Used to snap each WR10 slot to the closest previous mosaic frame.
 */
function findPreviousOrEqualTs(sortedTs, target) {
  let lo = 0, hi = sortedTs.length - 1, result = null
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (sortedTs[mid] <= target) { result = sortedTs[mid]; lo = mid + 1 }
    else hi = mid - 1
  }
  return result
}

// ---- Core load ----
async function loadData() {
  isLoading.value = true
  loadError.value = ''

  try {
    const lookbackMinutes = lookbackHours.value * 60

    // WR10 timestamps — these drive the timeline slider
    const wr10Results = await Promise.all(
      WR10_PRODUCTS.map(product =>
        api.wr10Timestamps(product, lookbackMinutes).catch(err => {
          console.error(`[WR10View] WR10 timestamps failed for ${product}:`, err)
          return { timestamps: [], total: 0 }
        })
      )
    )

    // Mosaic timestamps — fetch a slightly extended window (+15 min) so the oldest
    // WR10 slot always has a previous mosaic frame to snap to.
    const now = new Date()
    const MOSAIC_EXTRA_MIN = 15
    const mosaicStart = new Date(now - (lookbackMinutes + MOSAIC_EXTRA_MIN) * 60 * 1000)
    const startISO = mosaicStart.toISOString().slice(0, 16)
    const endISO   = now.toISOString().slice(0, 16)
    const mosaicResults = await Promise.all(
      MOSAIC_PRODUCTS.map(product =>
        api.explorerTimestamps(startISO, endISO, MOSAIC_API_PRODUCT[product]).catch(err => {
          console.error(`[WR10View] mosaic timestamps failed for ${product}:`, err)
          return { timestamps: [], total_found: 0 }
        })
      )
    )

    // Timeline is driven by WR10 only — union of SRI + VMI WR10 timestamps
    const tsSet = new Set()
    wr10Results.forEach(r => r.timestamps.forEach(ts => tsSet.add(ts)))
    const sortedTs = Array.from(tsSet).sort()

    if (sortedTs.length === 0) {
      loadError.value = 'No WR10 data found in the selected window.'
      isLoaded.value = false
      return
    }

    loadError.value = ''
    const prevLen      = timestamps.value.length
    const prevFraction = prevLen > 1 ? frameIndex.value / (prevLen - 1) : 1

    timestamps.value = sortedTs

    // Track frame counts per product
    wr10Results.forEach((r, i) => {
      productFrameCount.value[WR10_PRODUCTS[i]] = r.total
    })
    mosaicResults.forEach((r, i) => {
      productFrameCount.value[MOSAIC_PRODUCTS[i]] = r.total_found ?? r.total ?? 0
    })

    // Load overlays — WR10 layers with WR10 bounds, mosaic layers with Italy bounds.
    // For each WR10 slot, the mosaic layer shows the closest previous mosaic frame
    // (same frame repeats until the next mosaic timestamp arrives).
    radarMap.value?.clearAllProducts()
    const wr10Bounds = overlayBounds.value || undefined

    await Promise.all([
      // WR10 products — one URL per WR10 timestamp (null if that product lacks the slot)
      ...WR10_PRODUCTS.map(async (product, i) => {
        const found = new Set(wr10Results[i].timestamps)
        const urls  = sortedTs.map(ts => found.has(ts) ? api.wr10OverlayUrl(ts, product) : null)
        await radarMap.value?.loadProductFrames(product, urls, layerConfig.value[product].opacity, wr10Bounds)
      }),
      // Mosaic products — snap each WR10 slot to the closest previous mosaic timestamp
      ...MOSAIC_PRODUCTS.map(async (product, i) => {
        const apiProduct  = MOSAIC_API_PRODUCT[product]
        const mosaicSorted = [...mosaicResults[i].timestamps].sort()
        const urls = sortedTs.map(ts => {
          const closest = findPreviousOrEqualTs(mosaicSorted, ts)
          return closest ? api.explorerOverlayUrl(apiProduct, closest) : null
        })
        await radarMap.value?.loadProductFrames(product, urls, layerConfig.value[product].opacity, ITALY_BOUNDS)
      }),
    ])

    isLoaded.value = true

    if (followLive.value) {
      goToFrame(sortedTs.length - 1)
    } else if (prevLen > 0) {
      goToFrame(Math.min(Math.round(prevFraction * (sortedTs.length - 1)), sortedTs.length - 1))
    } else {
      goToFrame(sortedTs.length - 1)
    }

  } catch (e) {
    console.error('[WR10View] loadData error:', e)
    loadError.value = e.message || 'Unknown error'
  } finally {
    isLoading.value = false
  }
}

// ---- Frame navigation ----
function goToFrame(idx) {
  frameIndex.value = idx
  if (!radarMap.value || !isLoaded.value) return
  const opacities = {}
  for (const product of PRODUCTS) {
    opacities[product] = layerConfig.value[product]?.enabled
      ? layerConfig.value[product].opacity
      : 0
  }
  radarMap.value.showAllAtFrame(idx, opacities)
}

function goToLatest() {
  if (timestamps.value.length > 0) goToFrame(timestamps.value.length - 1)
}

// Re-render current frame when layer visibility/opacity changes
watch(layerConfig, () => {
  if (!isLoaded.value || timestamps.value.length === 0) return
  goToFrame(frameIndex.value)
}, { deep: true })

// ---- Lookback change ----
async function setLookback(hours) {
  if (isLoading.value) return
  if (hours === lookbackHours.value && isLoaded.value) return
  lookbackHours.value = hours
  stopAnimation()
  await loadData()
}

// ---- Slider ----
function onSliderInput(e) {
  goToFrame(parseInt(e.target.value))
  followLive.value = false
}

// ---- Animation ----
function togglePlay() {
  isPlaying.value ? stopAnimation() : startAnimation()
}

function startAnimation() {
  if (timestamps.value.length === 0) return
  isPlaying.value = true
  _animTimer = setInterval(() => {
    const next = (frameIndex.value + 1) % timestamps.value.length
    goToFrame(next)
    if (next === timestamps.value.length - 1) followLive.value = false
  }, 500 / playSpeed.value)
}

function stopAnimation() {
  isPlaying.value = false
  clearInterval(_animTimer)
  _animTimer = null
}

function cycleSpeed() {
  const idx = SPEEDS.indexOf(playSpeed.value)
  playSpeed.value = SPEEDS[(idx + 1) % SPEEDS.length]
  if (isPlaying.value) { stopAnimation(); startAnimation() }
}

// ---- Click-to-inspect popup ----
function fmtValue(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return 'N/A'
  return Math.abs(v) < 10 ? v.toFixed(2) : v.toFixed(1)
}

async function onMapClick(latlng) {
  if (!radarMap.value || !timestamps.value.length) return
  const ts = timestamps.value[frameIndex.value]
  if (!ts) return

  const enabledWr10    = WR10_PRODUCTS.filter(p => layerConfig.value[p].enabled)
  const enabledMosaic  = MOSAIC_PRODUCTS.filter(p => layerConfig.value[p].enabled)
  if (enabledWr10.length === 0 && enabledMosaic.length === 0) return

  const tzLabel = settings.timeZone === 'utc' ? 'UTC' : 'Local'
  radarMap.value.showPopup(latlng, `
    <div class="pi-header">${ts.replace('T', ' ')} (UTC)</div>
    <div class="pi-row"><span class="pi-label">Loading…</span></div>
  `)

  try {
    // Run both sampling calls in parallel (skipping if no enabled products)
    const [wr10Data, mosaicData] = await Promise.all([
      enabledWr10.length > 0
        ? api.wr10SamplePixel({ lat: latlng.lat, lon: latlng.lng, timestamp: ts, products: enabledWr10 })
        : Promise.resolve(null),
      enabledMosaic.length > 0
        ? api.samplePixel({
            lat: latlng.lat,
            lon: latlng.lng,
            timestamp: ts,
            products: enabledMosaic.map(p => MOSAIC_API_PRODUCT[p]),
          }).catch(() => null)
        : Promise.resolve(null),
    ])

    let body = ''

    // WR10 section
    if (wr10Data) {
      if (!wr10Data.in_bounds) {
        body += `<div class="pi-row"><span class="pi-label">WR10: outside coverage</span></div>`
      } else {
        body += `
          <div class="pi-row" style="margin-bottom:4px;">
            <span class="pi-label">range / az</span>
            <span class="pi-value">${wr10Data.range_km} km / ${wr10Data.azimuth_deg}°</span>
          </div>`
        for (const p of enabledWr10) {
          const v = wr10Data.values?.[p]
          const u = productMeta.value[p]?.unit || ''
          body += `
            <div class="pi-row">
              <span class="pi-label">${PRODUCT_LABELS[p]}</span>
              <span class="pi-value">${fmtValue(v)}${v != null && u ? ' ' + u : ''}</span>
            </div>`
        }
      }
    }

    // Mosaic section
    if (mosaicData) {
      if (!mosaicData.in_bounds) {
        body += `<div class="pi-row"><span class="pi-label">Mosaic: outside coverage</span></div>`
      } else {
        for (const p of enabledMosaic) {
          const apiKey = MOSAIC_API_PRODUCT[p]
          const v = mosaicData.values?.[apiKey]
          const u = productMeta.value[p]?.unit || ''
          body += `
            <div class="pi-row">
              <span class="pi-label">${PRODUCT_LABELS[p]}</span>
              <span class="pi-value">${fmtValue(v)}${v != null && u ? ' ' + u : ''}</span>
            </div>`
        }
      }
    }

    radarMap.value.showPopup(latlng, `
      <div class="pi-header">${ts.replace('T', ' ')} (${tzLabel})</div>
      ${body || '<div class="pi-row"><span class="pi-label">No data</span></div>'}
    `)
  } catch (e) {
    radarMap.value.showPopup(latlng,
      `<div class="pi-row"><span class="pi-label">Error: ${e.message || e}</span></div>`)
  }
}

// ---- Lifecycle ----
onMounted(async () => {
  // Block the WebSocket handler from triggering a loadData() while we are
  // still fetching the WR10 config.  Without this, a WS notification arriving
  // during the config fetch starts a concurrent load (isLoading is still false
  // at that point), and the two calls race on clearAllProducts/productGenerations,
  // leaving stale removed-from-map layers in productLayerMap.
  isLoading.value = true

  try {
    const cfg = await api.wr10Config()
    radarCenter.value   = cfg.center
    radarZoom.value     = cfg.zoom ?? 10
    overlayBounds.value = cfg.overlay_bounds
    productMeta.value   = cfg.products ?? {}
  } catch (e) {
    console.warn('[WR10View] Failed to fetch config, using defaults:', e)
  }

  // Add metadata for mosaic layers (reuse WR10 legends since they share the same products)
  productMeta.value['SRI_MOSAIC'] = { unit: 'mm/h', legend: 'R',  label: 'SRI Mosaic', thresholds: [], colors: [] }
  productMeta.value['VMI_MOSAIC'] = { unit: 'dBZ',  legend: 'CZ', label: 'VMI Mosaic', thresholds: [], colors: [] }

  isLoading.value = false
  await loadData()
  _scheduleNextPoll()
})

onUnmounted(() => {
  clearTimeout(_pollTimer)
  stopAnimation()
})
</script>

<style scoped>
/* Timeline slider — identical to RealTimeView */
.timeline-slider {
  background: rgba(255, 255, 255, 0.2);
}
.timeline-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: white;
  cursor: pointer;
  box-shadow: 0 0 4px rgba(0,0,0,0.5);
}
.timeline-slider::-moz-range-thumb {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: white;
  cursor: pointer;
  border: none;
  box-shadow: 0 0 4px rgba(0,0,0,0.5);
}
</style>
