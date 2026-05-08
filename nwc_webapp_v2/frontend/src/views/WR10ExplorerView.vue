<!--
  WR10ExplorerView.vue — Date-range browser for the WR10 X-band radar.

  Shows VMI, SRI, and PPI (corrected + uncorrected) for a user-selected
  time window. PPI has a shared elevation selector (1.5° – 6.0°). Corrected
  and uncorrected PPI are kept as independent map layers so they can be
  viewed side-by-side for quality assessment.
-->
<template>
  <div class="h-[calc(100vh-3.5rem)] flex overflow-hidden">

    <!-- ================================================================ -->
    <!-- LEFT: Map                                                          -->
    <!-- ================================================================ -->
    <div class="flex-1 flex flex-col relative min-w-0">
      <RadarMap
        ref="radarMap"
        :center="radarCenter"
        :zoom="radarZoom"
        :overlay-bounds="overlayBounds"
        class="flex-1"
      />

      <!-- Mobile sidebar toggle -->
      <button
        v-if="!sidebarOpen"
        @click="sidebarOpen = true"
        class="absolute top-3 right-3 z-[1001] lg:hidden
               w-10 h-10 flex items-center justify-center rounded-full
               bg-white shadow-lg border border-gray-200 text-gray-600"
        title="Open panel"
      >
        <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path d="M4 6h16M4 12h16M4 18h16" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
      </button>

      <!-- Colorbars — bottom right, above timeline -->
      <div
        v-if="isLoaded"
        class="absolute bottom-[110px] right-[10px] z-[1001]
               flex flex-col gap-1.5 items-end
               max-h-[calc(100vh-18rem)] overflow-y-auto"
      >
        <ColorBar
          v-for="cb in colorbarsToShow"
          :key="cb.key"
          :legend="productMeta[cb.metaKey]"
          :product-name="cb.label"
        />
      </div>

      <!-- ============================================================ -->
      <!-- BOTTOM: Timeline                                               -->
      <!-- ============================================================ -->
      <div
        class="absolute bottom-0 left-0 right-0 z-[1000]
               bg-gradient-to-t from-black/80 via-black/60 to-transparent
               px-3 sm:px-6 pt-10 pb-4"
        :class="{ 'pointer-events-none opacity-40': !isLoaded }"
      >
        <div class="flex items-center justify-between text-white mb-2">
          <div class="text-xs font-medium text-gray-300 hidden sm:block truncate max-w-[200px]">
            {{ visibleLayerLabels || '—' }}
          </div>
          <div class="text-center">
            <span class="text-base sm:text-xl font-bold tabular-nums tracking-tight">
              {{ currentTimestampDisplay }}
            </span>
          </div>
          <button
            @click="cycleSpeed"
            class="text-sm bg-white/15 hover:bg-white/25 border border-white/25 text-white
                   rounded-md px-3 py-1 transition-colors font-semibold tabular-nums min-w-[48px]"
          >
            {{ playSpeed }}×
          </button>
        </div>

        <div class="flex items-center gap-3">
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

          <div class="flex-1 relative">
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
            <div v-if="hourTicks.length" class="relative mt-1 h-4">
              <span
                v-for="tick in hourTicks"
                :key="tick.label"
                class="absolute text-[9px] text-gray-400 -translate-x-1/2"
                :style="{ left: tick.pct + '%' }"
              >{{ tick.label }}</span>
            </div>
          </div>

          <div class="text-xs text-gray-400 flex-shrink-0 tabular-nums">
            {{ timestamps.length ? `${frameIndex + 1}/${timestamps.length}` : '0/0' }}
          </div>
        </div>
      </div>
    </div>

    <!-- ================================================================ -->
    <!-- RIGHT: Sidebar                                                     -->
    <!-- ================================================================ -->
    <div
      v-if="sidebarOpen"
      class="fixed inset-0 bg-black/40 z-[1100] lg:hidden"
      @click="sidebarOpen = false"
    />

    <div
      class="bg-gray-900 border-l border-gray-700 flex flex-col
             fixed right-0 top-14 bottom-0 z-[1101] w-72
             transform transition-transform duration-200 ease-out overflow-y-auto
             lg:static lg:translate-x-0 lg:z-auto"
      :class="sidebarOpen ? 'translate-x-0' : 'translate-x-full'"
    >
      <!-- Close (mobile) -->
      <button
        @click="sidebarOpen = false"
        class="lg:hidden absolute top-2 right-2 w-8 h-8 flex items-center justify-center
               rounded-full text-gray-400 hover:text-gray-200 hover:bg-white/10 transition-colors"
      >
        <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path d="M6 18L18 6M6 6l12 12" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
      </button>

      <div class="p-4 space-y-5">

        <!-- Title -->
        <div class="pt-1">
          <h2 class="text-white font-bold text-base">WR10 Explorer</h2>
          <p class="text-gray-400 text-xs mt-0.5">Browse historical WR10 radar data</p>
        </div>

        <!-- Date Range -->
        <div class="space-y-3">
          <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Date Range</h3>

          <div>
            <label class="block text-xs text-gray-500 mb-1.5">Start</label>
            <div class="flex gap-2">
              <VueDatePicker
                :model-value="startDate"
                @update:model-value="onStartDateChange"
                :time-config="{ enableTimePicker: false }"
                auto-apply :dark="true" :formats="dateFormats"
                model-type="yyyy-MM-dd"
                input-class-name="dp-dark-input dp-wr10ex-date"
              />
              <VueDatePicker
                :model-value="startTimeObj"
                @update:model-value="onStartTimeChange"
                time-picker :dark="true" :is-24="true"
                :time-config="{ minutesIncrement: 5, minutesGridIncrement: 5 }"
                input-class-name="dp-dark-input dp-wr10ex-time"
              />
            </div>
          </div>

          <div>
            <label class="block text-xs text-gray-500 mb-1.5">End</label>
            <div class="flex gap-2">
              <VueDatePicker
                :model-value="endDate"
                @update:model-value="onEndDateChange"
                :time-config="{ enableTimePicker: false }"
                auto-apply :dark="true" :formats="dateFormats"
                model-type="yyyy-MM-dd"
                input-class-name="dp-dark-input dp-wr10ex-date"
              />
              <VueDatePicker
                :model-value="endTimeObj"
                @update:model-value="onEndTimeChange"
                time-picker :dark="true" :is-24="true"
                :time-config="{ minutesIncrement: 5, minutesGridIncrement: 5 }"
                input-class-name="dp-dark-input dp-wr10ex-time"
              />
            </div>
          </div>

          <p v-if="rangeWarning" class="text-amber-400 text-xs flex items-start gap-1">
            <svg class="w-3.5 h-3.5 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
            </svg>
            {{ rangeWarning }}
          </p>
        </div>

        <!-- Load button -->
        <button
          @click="loadData"
          :disabled="isLoading || !!rangeWarning"
          class="w-full py-2.5 rounded-lg text-sm font-semibold transition-colors
                 bg-blue-600 hover:bg-blue-500 text-white
                 disabled:opacity-50 disabled:cursor-not-allowed
                 flex items-center justify-center gap-2"
        >
          <svg v-if="isLoading" class="animate-spin h-4 w-4" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <span>{{ isLoading ? 'Loading…' : 'Load Data' }}</span>
        </button>

        <!-- No data -->
        <p v-if="noDataMsg" class="text-amber-400 text-xs text-center">{{ noDataMsg }}</p>

        <!-- ---- VMI / SRI / CPI layers ---- -->
        <div class="space-y-2">
          <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Layers</h3>

          <div
            v-for="item in STANDARD_LAYERS"
            :key="item.key"
            class="bg-gray-800 rounded-lg p-3 space-y-2"
          >
            <div class="flex items-center gap-2">
              <input
                type="checkbox"
                :id="`layer-${item.key}`"
                v-model="layerConfig[item.key].enabled"
                class="w-4 h-4 rounded accent-blue-500 cursor-pointer flex-shrink-0"
              />
              <label :for="`layer-${item.key}`" class="text-white text-sm font-bold cursor-pointer flex-1">
                {{ item.label }}
              </label>
              <span class="text-gray-400 text-xs">{{ item.unit }}</span>
            </div>
            <div class="flex items-center gap-2">
              <span class="text-gray-400 text-xs w-12 flex-shrink-0">Opacity</span>
              <input
                type="range" min="0" max="1" step="0.05"
                v-model.number="layerConfig[item.key].opacity"
                class="flex-1 h-1 accent-blue-400 cursor-pointer"
              />
              <span class="text-gray-400 text-xs w-8 text-right tabular-nums">
                {{ Math.round(layerConfig[item.key].opacity * 100) }}%
              </span>
            </div>
          </div>
        </div>

        <!-- ---- PPI section ---- -->
        <div class="space-y-2">
          <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider">PPI</h3>

          <!-- Elevation selector -->
          <div class="bg-gray-800 rounded-lg p-3 space-y-3">
            <div>
              <p class="text-xs text-gray-400 mb-2">Elevation angle</p>
              <div v-if="availableElevations.length" class="flex flex-wrap gap-1">
                <button
                  v-for="elev in availableElevations"
                  :key="elev"
                  @click="changeElevation(elev)"
                  :disabled="ppiLoading"
                  class="px-2 py-1 rounded text-xs font-semibold transition-colors border
                         disabled:cursor-not-allowed"
                  :class="selectedElevation === elev
                    ? 'bg-blue-600 border-blue-500 text-white'
                    : 'bg-gray-700 border-gray-600 text-gray-300 hover:bg-gray-600'"
                >
                  {{ elevLabel(elev) }}°
                </button>
              </div>
              <p v-else class="text-gray-500 text-xs italic">Load data first</p>

              <!-- PPI loading spinner -->
              <div v-if="ppiLoading" class="flex items-center gap-1.5 mt-2">
                <svg class="animate-spin h-3 w-3 text-blue-400" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span class="text-xs text-blue-400">Switching elevation…</span>
              </div>
            </div>

            <!-- PPI-C (corrected) -->
            <div class="space-y-1.5">
              <div class="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="layer-PPI_C"
                  v-model="layerConfig.PPI_C.enabled"
                  class="w-4 h-4 rounded accent-blue-500 cursor-pointer flex-shrink-0"
                />
                <label for="layer-PPI_C" class="text-white text-sm font-bold cursor-pointer flex-1">
                  Corrected
                </label>
                <span class="text-gray-400 text-xs">dBZ</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-gray-400 text-xs w-12 flex-shrink-0">Opacity</span>
                <input
                  type="range" min="0" max="1" step="0.05"
                  v-model.number="layerConfig.PPI_C.opacity"
                  class="flex-1 h-1 accent-blue-400 cursor-pointer"
                />
                <span class="text-gray-400 text-xs w-8 text-right tabular-nums">
                  {{ Math.round(layerConfig.PPI_C.opacity * 100) }}%
                </span>
              </div>
            </div>

            <!-- PPI-U (uncorrected) -->
            <div class="space-y-1.5">
              <div class="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="layer-PPI_U"
                  v-model="layerConfig.PPI_U.enabled"
                  class="w-4 h-4 rounded accent-blue-500 cursor-pointer flex-shrink-0"
                />
                <label for="layer-PPI_U" class="text-white text-sm font-bold cursor-pointer flex-1">
                  Uncorrected
                </label>
                <span class="text-gray-400 text-xs">dBZ</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-gray-400 text-xs w-12 flex-shrink-0">Opacity</span>
                <input
                  type="range" min="0" max="1" step="0.05"
                  v-model.number="layerConfig.PPI_U.opacity"
                  class="flex-1 h-1 accent-blue-400 cursor-pointer"
                />
                <span class="text-gray-400 text-xs w-8 text-right tabular-nums">
                  {{ Math.round(layerConfig.PPI_U.opacity * 100) }}%
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- Summary after load -->
        <div v-if="isLoaded" class="bg-gray-800 rounded-lg p-3 text-xs text-gray-400 space-y-1">
          <div class="flex justify-between">
            <span>Frames</span>
            <span class="text-white font-medium">{{ timestamps.length }}</span>
          </div>
          <div class="flex justify-between">
            <span>Duration</span>
            <span class="text-white font-medium">{{ durationDisplay }}</span>
          </div>
          <div v-for="product in ['VMI', 'SRI', 'CPI', 'PPI']" :key="product" class="flex justify-between">
            <span>{{ product }}</span>
            <span class="text-green-400 font-medium">{{ perProductCount[product] ?? 0 }} frames</span>
          </div>
        </div>

      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { VueDatePicker } from '@vuepic/vue-datepicker'
import '@vuepic/vue-datepicker/dist/main.css'
import RadarMap from '../components/RadarMap.vue'
import ColorBar from '../components/ColorBar.vue'
import api from '../api.js'

// ---- Map ref + config ----
const radarMap      = ref(null)
const radarCenter   = ref([41.842, 12.647])
const radarZoom     = ref(10)
const overlayBounds = ref(null)
const productMeta   = ref({})   // { VMI: { legend, unit, ... }, SRI: ..., PPI: ... }

// ---- Sidebar ----
const sidebarOpen = ref(false)

// ---- Date/time state ----
function todayStr() {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`
}
const startDateTime = ref(`${todayStr()}T00:00`)
const endDateTime   = ref(`${todayStr()}T02:00`)

function parseParts(isoStr) {
  if (!isoStr?.includes('T')) return { date: '', hour: '00', minute: '00' }
  const [date, time] = isoStr.split('T')
  const [hour, minute] = (time || '00:00').split(':')
  return { date, hour: hour || '00', minute: minute || '00' }
}
const startDate    = computed(() => parseParts(startDateTime.value).date || null)
const endDate      = computed(() => parseParts(endDateTime.value).date || null)
const startTimeObj = computed(() => {
  const p = parseParts(startDateTime.value)
  return { hours: parseInt(p.hour) || 0, minutes: parseInt(p.minute) || 0, seconds: 0 }
})
const endTimeObj = computed(() => {
  const p = parseParts(endDateTime.value)
  return { hours: parseInt(p.hour) || 0, minutes: parseInt(p.minute) || 0, seconds: 0 }
})

function buildDT(date, hour, minute) {
  if (!date) return ''
  return `${date}T${String(hour).padStart(2,'0')}:${String(minute).padStart(2,'0')}`
}
function onStartDateChange(val) {
  if (typeof val === 'string' && val) {
    const p = parseParts(startDateTime.value)
    startDateTime.value = buildDT(val, p.hour, p.minute)
  }
}
function onStartTimeChange(val) {
  if (val?.hours !== undefined) {
    const p = parseParts(startDateTime.value)
    startDateTime.value = buildDT(p.date || todayStr(), val.hours, val.minutes)
  }
}
function onEndDateChange(val) {
  if (typeof val === 'string' && val) {
    const p = parseParts(endDateTime.value)
    endDateTime.value = buildDT(val, p.hour, p.minute)
  }
}
function onEndTimeChange(val) {
  if (val?.hours !== undefined) {
    const p = parseParts(endDateTime.value)
    endDateTime.value = buildDT(p.date || todayStr(), val.hours, val.minutes)
  }
}
const dateFormats = { input: 'dd/MM/yyyy' }

// ---- Range validation ----
const MAX_HOURS = 48
const rangeWarning = computed(() => {
  if (!startDateTime.value || !endDateTime.value) return null
  const s = new Date(startDateTime.value), e = new Date(endDateTime.value)
  if (isNaN(s) || isNaN(e)) return 'Invalid date'
  if (e <= s) return 'End must be after start'
  if ((e - s) / 3_600_000 > MAX_HOURS) return `Range cannot exceed ${MAX_HOURS} hours`
  return null
})

// ---- Standard (non-PPI) layers ----
const STANDARD_LAYERS = [
  { key: 'VMI', label: 'VMI', unit: 'dBZ' },
  { key: 'SRI', label: 'SRI', unit: 'mm/h' },
  { key: 'CPI', label: 'CPI', unit: 'dBZ' },
]

// ---- Layer config ----
const layerConfig = ref({
  VMI:   { enabled: true,  opacity: 0.8  },
  SRI:   { enabled: false, opacity: 0.8  },
  CPI:   { enabled: false, opacity: 0.8  },
  PPI_C: { enabled: true,  opacity: 0.85 },
  PPI_U: { enabled: true,  opacity: 0.7  },
})

const ALL_PRODUCTS = ['VMI', 'SRI', 'CPI', 'PPI_C', 'PPI_U']

// ---- PPI elevation state ----
const availableElevations = ref([])
const selectedElevation   = ref('0015')
const ppiLoading          = ref(false)
// Per-elevation timestamp sets — only frames that actually exist at each elevation.
// Using these (instead of the union) prevents 404s for frames present at other elevations.
const ppiPerElevation     = ref({})   // { "0015": Set<isoStr>, "0025": Set<isoStr>, ... }

// ---- Timeline state ----
const timestamps    = ref([])
const frameIndex    = ref(0)
const isPlaying     = ref(false)
const isLoading     = ref(false)
const isLoaded      = ref(false)
const noDataMsg     = ref('')
const perProductCount = ref({})

let playInterval = null

// ---- Speeds ----
const SPEEDS = [0.5, 1, 2, 4]
const playSpeed = ref(1)

// ---- Computed ----

function elevLabel(code) {
  return (parseInt(code, 10) / 10).toFixed(1)
}

const visibleLayerLabels = computed(() => {
  const labels = []
  if (layerConfig.value.VMI.enabled)   labels.push('VMI')
  if (layerConfig.value.SRI.enabled)   labels.push('SRI')
  if (layerConfig.value.CPI.enabled)   labels.push('CPI')
  const elev = elevLabel(selectedElevation.value)
  if (layerConfig.value.PPI_C.enabled) labels.push(`PPI-C ${elev}°`)
  if (layerConfig.value.PPI_U.enabled) labels.push(`PPI-U ${elev}°`)
  return labels.join(' + ')
})

const currentTimestampDisplay = computed(() => {
  if (!timestamps.value.length) return '--/--/---- - --:--'
  const ts = timestamps.value[frameIndex.value]
  if (!ts) return '--/--/---- - --:--'
  const dt = new Date(ts)
  const d = String(dt.getDate()).padStart(2,'0')
  const m = String(dt.getMonth()+1).padStart(2,'0')
  const y = dt.getFullYear()
  const H = String(dt.getHours()).padStart(2,'0')
  const M = String(dt.getMinutes()).padStart(2,'0')
  return `${d}/${m}/${y} - ${H}:${M}`
})

const durationDisplay = computed(() => {
  if (timestamps.value.length < 2) return '--'
  const mins = Math.round(
    (new Date(timestamps.value.at(-1)) - new Date(timestamps.value[0])) / 60_000
  )
  return mins >= 60 ? `${Math.floor(mins/60)}h ${mins%60}m` : `${mins}m`
})

const hourTicks = computed(() => {
  if (timestamps.value.length < 2) return []
  const ticks = []
  const seen = new Set()
  timestamps.value.forEach((ts, i) => {
    const dt = new Date(ts)
    if (dt.getMinutes() === 0) {
      const label = `${String(dt.getHours()).padStart(2,'0')}:00`
      if (!seen.has(label)) {
        seen.add(label)
        ticks.push({ label, pct: (i / (timestamps.value.length - 1)) * 100 })
      }
    }
  })
  return ticks
})

// Colorbars: SRI → R legend, VMI/CPI/PPI_C/PPI_U → one shared CZ colorbar
const colorbarsToShow = computed(() => {
  const bars = []
  // SRI is independent (rain rate, R legend)
  if (layerConfig.value.SRI.enabled && (perProductCount.value.SRI ?? 0) > 0)
    bars.push({ key: 'SRI', metaKey: 'SRI', label: 'SRI' })
  // Reflectivity products share a single CZ colorbar
  const elev = elevLabel(selectedElevation.value)
  const czProducts = [
    layerConfig.value.VMI.enabled   ? 'VMI'            : null,
    layerConfig.value.CPI.enabled   ? 'CPI'            : null,
    layerConfig.value.PPI_C.enabled ? `PPI-C ${elev}°` : null,
    layerConfig.value.PPI_U.enabled ? `PPI-U ${elev}°` : null,
  ].filter(Boolean)
  if (czProducts.length > 0)
    bars.push({ key: 'CZ', metaKey: 'VMI', label: czProducts.join(' / ') })
  return bars
})

// ---- Frame navigation ----
function goToFrame(idx) {
  frameIndex.value = idx
  if (!radarMap.value || !isLoaded.value) return
  const opacities = {}
  for (const p of ALL_PRODUCTS) {
    opacities[p] = layerConfig.value[p].enabled ? layerConfig.value[p].opacity : 0
  }
  radarMap.value.showAllAtFrame(idx, opacities)
}

watch(layerConfig, () => {
  if (!isLoaded.value || timestamps.value.length === 0) return
  goToFrame(frameIndex.value)
}, { deep: true })

// ---- Load data ----
async function loadData() {
  if (rangeWarning.value) return
  isLoading.value = true
  isLoaded.value  = false
  noDataMsg.value = ''
  stopAnimation()
  timestamps.value      = []
  perProductCount.value = {}
  ppiPerElevation.value = {}
  availableElevations.value = []

  radarMap.value?.clearAllProducts()

  try {
    const result = await api.wr10ExplorerTimestamps(startDateTime.value, endDateTime.value, 'VMI,SRI,CPI,PPI')

    timestamps.value = result.timestamps
    availableElevations.value = result.ppi_elevations ?? []

    // Build per-elevation Sets for precise URL array construction
    ppiPerElevation.value = {}
    for (const [elev, tsArr] of Object.entries(result.ppi_per_elevation ?? {})) {
      ppiPerElevation.value[elev] = new Set(tsArr)
    }

    // Default to lowest elevation if current selection is unavailable
    if (availableElevations.value.length && !availableElevations.value.includes(selectedElevation.value)) {
      selectedElevation.value = availableElevations.value[0]
    }

    if (timestamps.value.length === 0) {
      noDataMsg.value = 'No WR10 data found for the selected range.'
      return
    }

    const tsAll = timestamps.value
    const vmiSet = new Set(result.per_product?.VMI ?? [])
    const sriSet = new Set(result.per_product?.SRI ?? [])
    const cpiSet = new Set(result.per_product?.CPI ?? [])
    // Use the per-elevation set (not the union) so frames missing at this elevation get null URLs
    const ppiElevSet = ppiPerElevation.value[selectedElevation.value] ?? new Set()

    perProductCount.value = {
      VMI: vmiSet.size,
      SRI: sriSet.size,
      CPI: cpiSet.size,
      PPI: (result.per_product?.PPI ?? []).length,
    }

    const bounds = overlayBounds.value ?? undefined

    await Promise.all([
      radarMap.value?.loadProductFrames(
        'VMI',
        tsAll.map(ts => vmiSet.has(ts) ? api.wr10OverlayUrl(ts, 'VMI') : null),
        layerConfig.value.VMI.opacity,
        bounds,
      ),
      radarMap.value?.loadProductFrames(
        'SRI',
        tsAll.map(ts => sriSet.has(ts) ? api.wr10OverlayUrl(ts, 'SRI') : null),
        layerConfig.value.SRI.opacity,
        bounds,
      ),
      radarMap.value?.loadProductFrames(
        'CPI',
        tsAll.map(ts => cpiSet.has(ts) ? api.wr10OverlayUrl(ts, 'CPI') : null),
        layerConfig.value.CPI.opacity,
        bounds,
      ),
      radarMap.value?.loadProductFrames(
        'PPI_C',
        tsAll.map(ts => ppiElevSet.has(ts) ? api.wr10PpiOverlayUrl(ts, selectedElevation.value, 'C') : null),
        layerConfig.value.PPI_C.opacity,
        bounds,
      ),
      radarMap.value?.loadProductFrames(
        'PPI_U',
        tsAll.map(ts => ppiElevSet.has(ts) ? api.wr10PpiOverlayUrl(ts, selectedElevation.value, 'U') : null),
        layerConfig.value.PPI_U.opacity,
        bounds,
      ),
    ])

    isLoaded.value = true
    goToFrame(tsAll.length - 1)

  } catch (e) {
    console.error('[WR10ExplorerView] loadData error:', e)
    noDataMsg.value = e.message || 'Failed to load data.'
  } finally {
    isLoading.value = false
  }
}

// ---- Change PPI elevation (incremental update) ----
async function changeElevation(elev) {
  if (elev === selectedElevation.value) return
  selectedElevation.value = elev
  if (!isLoaded.value || timestamps.value.length === 0) return

  ppiLoading.value = true
  const tsAll      = timestamps.value
  const ppiElevSet = ppiPerElevation.value[elev] ?? new Set()
  const bounds     = overlayBounds.value ?? undefined

  try {
    await Promise.all([
      radarMap.value?.loadProductFrames(
        'PPI_C',
        tsAll.map(ts => ppiElevSet.has(ts) ? api.wr10PpiOverlayUrl(ts, elev, 'C') : null),
        layerConfig.value.PPI_C.opacity,
        bounds,
      ),
      radarMap.value?.loadProductFrames(
        'PPI_U',
        tsAll.map(ts => ppiElevSet.has(ts) ? api.wr10PpiOverlayUrl(ts, elev, 'U') : null),
        layerConfig.value.PPI_U.opacity,
        bounds,
      ),
    ])
    goToFrame(frameIndex.value)
  } catch (e) {
    console.error('[WR10ExplorerView] changeElevation error:', e)
  } finally {
    ppiLoading.value = false
  }
}

// ---- Slider + animation ----
function onSliderInput(e) {
  goToFrame(Number(e.target.value))
}

function togglePlay() {
  isPlaying.value ? stopAnimation() : startAnimation()
}
function startAnimation() {
  if (!timestamps.value.length) return
  isPlaying.value = true
  const fps = playSpeed.value * 3
  playInterval = setInterval(() => {
    goToFrame((frameIndex.value + 1) % timestamps.value.length)
  }, 1000 / fps)
}
function stopAnimation() {
  isPlaying.value = false
  if (playInterval) { clearInterval(playInterval); playInterval = null }
}
function cycleSpeed() {
  const idx = SPEEDS.indexOf(playSpeed.value)
  playSpeed.value = SPEEDS[(idx + 1) % SPEEDS.length]
  if (isPlaying.value) { stopAnimation(); startAnimation() }
}

// ---- Lifecycle ----
onMounted(async () => {
  try {
    const cfg = await api.wr10Config()
    radarCenter.value   = cfg.center
    radarZoom.value     = cfg.zoom ?? 10
    overlayBounds.value = cfg.overlay_bounds
    // Build productMeta from wr10 config for colorbars
    productMeta.value = cfg.products ?? {}
    // PPI shares the VMI (reflectivity/CZ) legend
    productMeta.value['PPI'] = { unit: 'dBZ', legend: 'CZ', label: 'PPI', thresholds: [], colors: [] }
  } catch (e) {
    console.warn('[WR10ExplorerView] Failed to fetch WR10 config:', e)
  }
})

onUnmounted(stopAnimation)
</script>

<style scoped>
.timeline-slider {
  background: linear-gradient(
    to right,
    #3b82f6 0%,
    #3b82f6 calc(var(--pct, 0) * 1%),
    #4b5563 calc(var(--pct, 0) * 1%),
    #4b5563 100%
  );
}
.timeline-slider::-webkit-slider-thumb {
  appearance: none;
  width: 16px; height: 16px;
  border-radius: 50%;
  background: white;
  cursor: pointer;
  box-shadow: 0 1px 4px rgba(0,0,0,0.4);
}
.timeline-slider::-moz-range-thumb {
  width: 16px; height: 16px;
  border-radius: 50%;
  background: white;
  cursor: pointer;
  border: none;
  box-shadow: 0 1px 4px rgba(0,0,0,0.4);
}
</style>

<style>
.dp-wr10ex-date {
  width: 118px !important;
  height: 36px !important;
  border-radius: 0.5rem !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  background: rgba(255,255,255,0.06) !important;
  color: white !important;
  font-size: 0.8rem !important;
  padding: 0 0.5rem !important;
}
.dp-wr10ex-time {
  width: 96px !important;
  height: 36px !important;
  border-radius: 0.5rem !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  background: rgba(255,255,255,0.06) !important;
  color: white !important;
  font-size: 0.8rem !important;
  padding: 0 0.4rem !important;
}
.dp-wr10ex-date:focus,
.dp-wr10ex-time:focus {
  border-color: #60a5fa !important;
  outline: none !important;
}
</style>
