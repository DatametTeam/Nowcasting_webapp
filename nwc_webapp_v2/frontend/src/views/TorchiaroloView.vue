<!--
  TorchiaroloView.vue — Real-time monitor for the Torchiarolo (Puglia) radar.

  Shows the four composite products delivered by the provider — SRI (rain rate),
  VMI (max reflectivity), VIL (vertically integrated liquid) and ETM (echo top)
  — for a rolling lookback window. Auto-loads on mount, polls every 5 minutes
  for new data, and also receives instant push notifications via WebSocket
  (/api/torchiarolo/ws).

  Structure mirrors CagliariView: no model predictions, just folder-watching and
  map display. Unlike Cagliari there are no PPI elevations — every product is a
  single Cartesian field on the same 400x400 tmerc grid.
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

      <!-- Loading indicator -->
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
            <h2 class="text-white font-bold text-base">Torchiarolo</h2>
            <p class="text-gray-400 text-xs mt-0.5">Puglia radar composite monitor</p>
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
            <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Torchiarolo Layers</h3>
            <div
              v-for="product in TORCHIAROLO_PRODUCTS"
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
                  {{ productMeta[product]?.unit || '' }}
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

          <!-- Motion field layer (AMV / LK) -->
          <div class="space-y-2">
            <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Motion Field</h3>
            <div class="bg-gray-800 rounded-lg p-3 space-y-2">
              <div class="flex items-center gap-1">
                <button
                  v-for="mode in ['none', 'amv', 'lk']"
                  :key="mode"
                  @click="motionMode = mode"
                  :class="['flex-1 py-1 text-xs font-semibold rounded transition-colors',
                           motionMode === mode
                             ? 'bg-blue-500 text-white'
                             : 'bg-gray-700 text-gray-400 hover:text-white']"
                >{{ mode === 'none' ? 'None' : mode.toUpperCase() }}</button>
                <svg v-if="motionLoading" class="animate-spin h-3 w-3 text-blue-400 flex-shrink-0 ml-1" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              </div>
              <template v-if="motionMode === 'lk'">
                <div class="flex items-center gap-1">
                  <button
                    v-for="dm in ['particles', 'arrows', 'both']"
                    :key="dm"
                    @click="lkDisplayMode = dm"
                    :class="['flex-1 py-1 text-[10px] font-semibold rounded transition-colors capitalize',
                             lkDisplayMode === dm
                               ? 'bg-teal-600 text-white'
                               : 'bg-gray-700 text-gray-400 hover:text-white']"
                  >{{ dm }}</button>
                </div>
                <div v-if="lkDisplayMode !== 'particles'" class="flex items-center gap-2">
                  <span class="text-gray-400 text-[10px] w-12 flex-shrink-0">Arrows</span>
                  <input
                    type="range" min="0" max="1" step="0.05"
                    v-model.number="lkArrowOpacity"
                    class="flex-1 h-1 accent-teal-400 cursor-pointer"
                  />
                  <span class="text-gray-400 text-[10px] w-8 text-right tabular-nums">
                    {{ Math.round(lkArrowOpacity * 100) }}%
                  </span>
                </div>
              </template>
              <div v-if="motionMode !== 'none' && activeMotionTs" class="text-gray-500 text-[10px]">
                {{ motionMode.toUpperCase() }}: {{ activeMotionTs.replace('T', ' ') }} UTC
                <span v-if="motionMode === 'amv'" class="text-gray-600 ml-1">(20 min cadence)</span>
              </div>
              <div v-if="motionMode !== 'none' && !activeMotionTs && !motionLoading" class="text-amber-400 text-[10px]">
                No {{ motionMode.toUpperCase() }} data for current time
              </div>
            </div>
          </div>

          <!-- Radar info -->
          <div class="bg-gray-800 rounded-lg p-3 space-y-1 text-xs text-gray-400">
            <p class="font-semibold text-gray-300">Radar Info</p>
            <p>Type: composite (Torchiarolo)</p>
            <p v-if="gridInfo">Range: {{ gridInfo.range_km }} km</p>
            <p v-if="gridInfo">
              Grid: {{ gridInfo.ncols }}×{{ gridInfo.nlines }} @ {{ (gridInfo.xscale / 1000).toFixed(1) }} km
            </p>
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
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import RadarMap from '../components/RadarMap.vue'
import ColorBar from '../components/ColorBar.vue'
import { useSettingsStore } from '../stores/settings.js'
import { useConfigStore } from '../stores/config.js'
import { useTorchiaroloWs } from '../composables/useTorchiaroloWs.js'
import { useRealtimeWs } from '../composables/useRealtimeWs.js'
import { useMotionLayer } from '../composables/useMotionLayer.js'
import api from '../api.js'

const settings    = useSettingsStore()
const configStore = useConfigStore()

// ---- Products ----
const TORCHIAROLO_PRODUCTS = ['SRI', 'VMI', 'VIL', 'ETM']
const MOSAIC_PRODUCTS      = ['SRI_MOSAIC', 'VMI_MOSAIC']
const PRODUCTS             = [...TORCHIAROLO_PRODUCTS, ...MOSAIC_PRODUCTS]

const MOSAIC_API_PRODUCT = { SRI_MOSAIC: 'SRI_adj', VMI_MOSAIC: 'VMI' }

// Layers are labelled with the provider's product codes rather than expanded
// names, matching how the products are referred to operationally.
const PRODUCT_LABELS = {
  SRI: 'SRI', VMI: 'VMI', VIL: 'VIL', ETM: 'ETM',
  SRI_MOSAIC: 'SRI Mosaic', VMI_MOSAIC: 'VMI Mosaic',
}

const ITALY_BOUNDS = [[35.0623, 4.51987], [47.5730, 20.4801]]

const layerConfig = ref({
  SRI:        { enabled: false, opacity: 0.8 },
  VMI:        { enabled: true,  opacity: 1.0 },
  VIL:        { enabled: false, opacity: 1.0 },
  ETM:        { enabled: false, opacity: 1.0 },
  SRI_MOSAIC: { enabled: false, opacity: 0.7 },
  VMI_MOSAIC: { enabled: false, opacity: 0.7 },
})

const visibleProducts = computed(() =>
  PRODUCTS.filter(p => layerConfig.value[p].enabled)
)

// Each Torchiarolo product has its own legend; the mosaics reuse the matching one.
const COLORBAR_PRODUCT = {
  SRI: 'SRI', VMI: 'VMI', VIL: 'VIL', ETM: 'ETM',
  SRI_MOSAIC: 'SRI', VMI_MOSAIC: 'VMI',
}
const colorbarsToShow = computed(() => {
  const seen = new Set()
  const result = []
  for (const p of visibleProducts.value) {
    const key = COLORBAR_PRODUCT[p] || p
    if (!seen.has(key)) { seen.add(key); result.push(key) }
  }
  return result
})

const productMeta       = ref({})
const productFrameCount = ref({ SRI: 0, VMI: 0, VIL: 0, ETM: 0, SRI_MOSAIC: 0, VMI_MOSAIC: 0 })

// ---- Map ----
const radarMap      = ref(null)
const radarCenter   = ref([40.5064, 18.0598])
const radarZoom     = ref(8)
const overlayBounds = ref(null)
const gridInfo      = ref(null)

// ---- Timeline ----
const timestamps = ref([])
const frameIndex = ref(0)
const isLoaded   = ref(false)

// ---- Lookback ----
const lookbackHours = ref(settings.defaultLookback ?? 1)

// ---- Motion field layer (AMV / LK) ----
const _motionCurrentTs = computed(() => (timestamps.value[frameIndex.value] ?? '').slice(0, 16))
const { motionMode, motionLoading, activeMotionTs, lkDisplayMode, lkArrowOpacity,
        sampleMotionAt } =
  useMotionLayer(radarMap, _motionCurrentTs, lookbackHours)
const isLoading  = ref(false)
const loadError  = ref('')
const followLive = ref(true)

// ---- Playback ----
const SPEEDS    = [0.5, 1, 2, 4]
const playSpeed = ref(settings.defaultSpeed ?? 1)
const isPlaying = ref(false)
let   _animTimer = null

// ---- Sidebar ----
const sidebarOpen = ref(false)

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

// ---- WebSockets ----
// Torchiarolo drives the timeline: a new file there rebuilds everything.
const { connected: wsConnected } = useTorchiaroloWs({
  onTorchiaroloUpdate: () => {
    if (!isLoading.value) loadData()
  },
})

// The national products arrive on their own schedule (about a minute after
// Torchiarolo), so top up just the mosaic layers when they land instead of
// waiting for the next full reload.
const MOSAIC_WATCHED = new Set(Object.values(MOSAIC_API_PRODUCT))

// Note: useRealtimeWs calls this as onProductReady(product, timestamp) — the
// product is a positional string, not the nested {data:{...}} payload the
// Torchiarolo and Cagliari sockets send.
useRealtimeWs({
  onProductReady: (product) => {
    if (!MOSAIC_WATCHED.has(product)) return
    if (isLoading.value) return
    reloadMosaics()
  },
})

// ---- 5-minute clock-aligned poll ----
let _pollTimer = null

function _scheduleNextPoll() {
  clearTimeout(_pollTimer)
  const now = new Date()
  const msToNextMark = (5 * 60 * 1000) - ((now.getMinutes() % 5) * 60 + now.getSeconds()) * 1000 - now.getMilliseconds()
  const delay = msToNextMark + 15_000
  _pollTimer = setTimeout(async () => {
    await loadData()
    _scheduleNextPoll()
  }, delay)
}

// ---- Mosaic layers ----
//
// The national mosaic lands about a minute after the Torchiarolo file for the
// same slot, so at the moment a Torchiarolo push triggers a reload the newest
// mosaic frame usually does not exist yet. We match strictly on equal
// timestamps and leave the frame empty rather than substituting an older
// mosaic, which would silently show a 5-minute-stale field next to fresh data.
// The gap fills itself in: ProductWatcherService broadcasts 'product_ready'
// when the mosaic arrives, and we reload just those layers then.
const MOSAIC_EXTRA_MIN = 15

async function fetchMosaicUrls(sortedTs) {
  const lookbackMinutes = lookbackHours.value * 60
  const now = new Date()
  const mosaicStart = new Date(now - (lookbackMinutes + MOSAIC_EXTRA_MIN) * 60 * 1000)
  mosaicStart.setMinutes(Math.floor(mosaicStart.getMinutes() / 5) * 5, 0, 0)
  const startISO = mosaicStart.toISOString().slice(0, 16)
  const endISO   = now.toISOString().slice(0, 16)

  const results = await Promise.all(
    MOSAIC_PRODUCTS.map(product =>
      api.explorerTimestamps(startISO, endISO, MOSAIC_API_PRODUCT[product], 'torchiarolo-mosaic').catch(err => {
        console.error(`[TorchiaroloView] mosaic timestamps failed for ${product}:`, err)
        return { timestamps: [], total_found: 0 }
      })
    )
  )

  return MOSAIC_PRODUCTS.map((product, i) => {
    const found = new Set(results[i].timestamps)
    return sortedTs.map(ts => found.has(ts) ? api.explorerOverlayUrl(MOSAIC_API_PRODUCT[product], ts) : null)
  })
}

async function loadMosaicLayers(sortedTs) {
  const urlsPerProduct = await fetchMosaicUrls(sortedTs)
  await Promise.all(MOSAIC_PRODUCTS.map(async (product, i) => {
    const urls = urlsPerProduct[i]
    productFrameCount.value[product] = urls.filter(Boolean).length
    await radarMap.value?.loadProductFrames(product, urls, layerConfig.value[product].opacity, ITALY_BOUNDS)
  }))
}

/** Refresh only the mosaic layers, leaving the Torchiarolo layers untouched. */
async function reloadMosaics() {
  if (!isLoaded.value || timestamps.value.length === 0) return
  try {
    await loadMosaicLayers(timestamps.value)
    goToFrame(frameIndex.value)
  } catch (e) {
    console.error('[TorchiaroloView] mosaic reload failed:', e)
  }
}

// ---- Core load ----
async function loadData() {
  isLoading.value = true
  loadError.value = ''

  try {
    const lookbackMinutes = lookbackHours.value * 60

    const torchiaroloResults = await Promise.all(
      TORCHIAROLO_PRODUCTS.map(product =>
        api.torchiaroloTimestamps(product, lookbackMinutes).catch(err => {
          console.error(`[TorchiaroloView] timestamps failed for ${product}:`, err)
          return { timestamps: [], total: 0 }
        })
      )
    )

    const tsSet = new Set()
    torchiaroloResults.forEach(r => r.timestamps.forEach(ts => tsSet.add(ts)))
    const sortedTs = Array.from(tsSet).sort()

    if (sortedTs.length === 0) {
      loadError.value = 'No Torchiarolo data found in the selected window.'
      isLoaded.value = false
      return
    }

    loadError.value = ''
    const prevLen      = timestamps.value.length
    const prevFraction = prevLen > 1 ? frameIndex.value / (prevLen - 1) : 1

    timestamps.value = sortedTs

    torchiaroloResults.forEach((r, i) => {
      productFrameCount.value[TORCHIAROLO_PRODUCTS[i]] = r.total
    })

    radarMap.value?.clearAllProducts()
    const torBounds = overlayBounds.value || undefined

    await Promise.all([
      ...TORCHIAROLO_PRODUCTS.map(async (product, i) => {
        const found = new Set(torchiaroloResults[i].timestamps)
        const urls  = sortedTs.map(ts => found.has(ts) ? api.torchiaroloOverlayUrl(ts, product) : null)
        await radarMap.value?.loadProductFrames(product, urls, layerConfig.value[product].opacity, torBounds)
      }),
      loadMosaicLayers(sortedTs),
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
    console.error('[TorchiaroloView] loadData error:', e)
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

watch(layerConfig, () => {
  if (!isLoaded.value || timestamps.value.length === 0) return
  goToFrame(frameIndex.value)
}, { deep: true })

watch(sidebarOpen, async () => {
  await nextTick()
  radarMap.value?.invalidateSize()
})

async function setLookback(hours) {
  if (isLoading.value) return
  if (hours === lookbackHours.value && isLoaded.value) return
  lookbackHours.value = hours
  stopAnimation()
  await loadData()
}

function onSliderInput(e) {
  goToFrame(parseInt(e.target.value))
  followLive.value = false
}

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

  const enabledTorchiarolo = TORCHIAROLO_PRODUCTS.filter(p => layerConfig.value[p].enabled)
  const enabledMosaic      = MOSAIC_PRODUCTS.filter(p => layerConfig.value[p].enabled)
  if (enabledTorchiarolo.length === 0 && enabledMosaic.length === 0) return

  const tzLabel = settings.timeZone === 'utc' ? 'UTC' : 'Local'
  radarMap.value.showPopup(latlng, `
    <div class="pi-header">${ts.replace('T', ' ')} (UTC)</div>
    <div class="pi-row"><span class="pi-label">Loading…</span></div>
  `)

  try {
    const [torData, mosaicData] = await Promise.all([
      enabledTorchiarolo.length > 0
        ? api.torchiaroloSamplePixel({
            lat: latlng.lat, lon: latlng.lng, timestamp: ts, products: enabledTorchiarolo,
          })
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

    if (torData) {
      if (!torData.in_bounds) {
        body += `<div class="pi-row"><span class="pi-label">Torchiarolo: outside coverage</span></div>`
      } else {
        body += `
          <div class="pi-row" style="margin-bottom:4px;">
            <span class="pi-label">range / az</span>
            <span class="pi-value">${torData.range_km} km / ${torData.azimuth_deg}°</span>
          </div>`
        for (const p of enabledTorchiarolo) {
          const v = torData.values?.[p]
          const u = productMeta.value[p]?.unit || ''
          body += `
            <div class="pi-row">
              <span class="pi-label">${PRODUCT_LABELS[p]}</span>
              <span class="pi-value">${fmtValue(v)}${v != null && u ? ' ' + u : ''}</span>
            </div>`
        }
      }
    }

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

    const motion = sampleMotionAt(latlng.lat, latlng.lng)
    if (motion) {
      const label    = motion.source === 'amv' ? 'AMV' : 'LK'
      const cardinals = ['N','NE','E','SE','S','SW','W','NW']
      const cardinal  = cardinals[Math.round(motion.direction / 45) % 8]
      body += `
        <div class="pi-row" style="margin-top:6px;border-top:1px solid rgba(255,255,255,0.12);padding-top:4px;">
          <span class="pi-label">${label} speed</span>
          <span class="pi-value">${motion.speed_kmh.toFixed(1)} km/h</span>
        </div>
        <div class="pi-row">
          <span class="pi-label">${label} dir</span>
          <span class="pi-value">${Math.round(motion.direction)}° ${cardinal}</span>
        </div>`
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
  isLoading.value = true

  try {
    const cfg = await api.torchiaroloConfig()
    radarCenter.value   = cfg.center
    radarZoom.value     = cfg.zoom ?? 8
    overlayBounds.value = cfg.overlay_bounds
    productMeta.value   = cfg.products ?? {}
    gridInfo.value      = cfg.grid ?? null
  } catch (e) {
    console.warn('[TorchiaroloView] Failed to fetch config, using defaults:', e)
  }

  productMeta.value['SRI_MOSAIC'] = { unit: 'mm/h', legend: 'R',  label: 'SRI Mosaic', thresholds: [], colors: [] }
  productMeta.value['VMI_MOSAIC'] = { unit: 'dBZ',  legend: 'CZ', label: 'VMI Mosaic', thresholds: [], colors: [] }

  isLoading.value = false
  await loadData()
  // No addFixedMarker here: TORCHIAROLO is one of the national sites in
  // RadarMap's RADARS list, so the marker is already drawn. Adding a fixed one
  // would stack a second icon on the same spot.
  _scheduleNextPoll()
})

onUnmounted(() => {
  clearTimeout(_pollTimer)
  stopAnimation()
})
</script>

<style scoped>
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
