<!--
  DataExplorerView.vue — Historical radar data browser.

  Fixes applied (v2):
  1.  Short product names (SRI, VMI, ETM, VIL) in sidebar
  2.  VueDatePicker with 5-minute step (same as MetricsView)
  3.  Always load all 4 products on "Load Data"
  4.  Per-product generation counters → all layers visible simultaneously
  5.  Checkbox toggle immediately shows/hides layer on map
  6.  Speed displayed as cycle button (no white-on-white select)
  7.  Product name on left side of colorbar; colorbars capped to map height
  8.  Sidebar: fixed right-0 top-14 bottom-0 (no white gap)
  9.  overflow-hidden on root → no page scroll; only sidebar scrolls
  10. Full date "DD/MM/YYYY - HH:MM" in timeline bar
  11. Opacity slider goes to 0%
  12. Per-product missing frames list (filenames)
-->
<template>
  <!-- overflow-hidden prevents page-level scroll; sidebar is the only scroll area -->
  <div class="h-[calc(100vh-3.5rem)] flex overflow-hidden">

    <!-- ================================================================ -->
    <!-- LEFT: Map area                                                    -->
    <!-- ================================================================ -->
    <div class="flex-1 flex flex-col relative min-w-0">
      <RadarMap ref="radarMap" class="flex-1" />

      <!-- Sidebar toggle (mobile only) -->
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

      <!-- Stacked colorbars — bottom right, above timeline, max-height capped -->
      <div
        class="absolute bottom-[110px] right-[10px] z-[1001]
               flex flex-col gap-1.5 items-end
               max-h-[calc(100vh-18rem)] overflow-y-auto"
      >
        <ColorBar
          v-for="product in visibleProducts"
          :key="product"
          :legend="radarProducts[product]"
          :product-name="SHORT_NAMES[product]"
        />
      </div>

      <!-- ============================================================ -->
      <!-- BOTTOM BAR: Timeline controls                                 -->
      <!-- ============================================================ -->
      <div
        class="absolute bottom-0 left-0 right-0 z-[1000]
               bg-gradient-to-t from-black/80 via-black/60 to-transparent
               px-3 sm:px-6 pt-10 pb-4"
        :class="{ 'pointer-events-none opacity-40': !isLoaded }"
      >
        <!-- Top row: layer names | full datetime | speed button -->
        <div class="flex items-center justify-between text-white mb-2">
          <div class="text-xs font-medium text-gray-300 hidden sm:block truncate max-w-[160px]">
            {{ visibleProducts.map(p => SHORT_NAMES[p]).join(' + ') || '—' }}
          </div>

          <!-- Full date + time (fix #10) -->
          <div class="text-center">
            <span class="text-base sm:text-xl font-bold tabular-nums tracking-tight">
              {{ currentTimestampDisplay }}
            </span>
          </div>

          <!-- Speed cycle button (fix #6 — no select dropdown) -->
          <button
            @click="cycleSpeed"
            class="text-sm bg-white/15 hover:bg-white/25 border border-white/25 text-white
                   rounded-md px-3 py-1 transition-colors font-semibold tabular-nums min-w-[48px]"
          >
            {{ playSpeed }}×
          </button>
        </div>

        <!-- Slider row -->
        <div class="flex items-center gap-3">
          <!-- Play / Pause -->
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

          <!-- Range slider + hour ticks -->
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

          <!-- Frame counter -->
          <div class="text-xs text-gray-400 flex-shrink-0 tabular-nums">
            {{ timestamps.length ? `${frameIndex + 1}/${timestamps.length}` : '0/0' }}
          </div>
        </div>
      </div>
    </div>

    <!-- ================================================================ -->
    <!-- RIGHT: Sidebar (same pattern as RealTimeView)                   -->
    <!-- ================================================================ -->
    <!-- Mobile backdrop -->
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

      <!-- ---- Sidebar content ---- -->
      <div class="p-4 space-y-5">

        <!-- Title -->
        <div class="pt-1">
          <h2 class="text-white font-bold text-base">Data Explorer</h2>
          <p class="text-gray-400 text-xs mt-0.5">Browse historical radar layers</p>
        </div>

        <!-- ---- Date / Time Range (VueDatePicker, same pattern as MetricsView) ---- -->
        <div class="space-y-3">
          <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Date Range</h3>

          <!-- Start -->
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">Start</label>
            <div class="flex gap-2">
              <VueDatePicker
                :model-value="startDate"
                @update:model-value="onStartDateChange"
                :time-config="{ enableTimePicker: false }"
                auto-apply
                :dark="true"
                :formats="dateFormats"
                model-type="yyyy-MM-dd"
                input-class-name="dp-dark-input dp-explorer-date"
              />
              <VueDatePicker
                :model-value="startTimeObj"
                @update:model-value="onStartTimeChange"
                time-picker
                :dark="true"
                :is-24="true"
                :time-config="{ minutesIncrement: 5, minutesGridIncrement: 5 }"
                input-class-name="dp-dark-input dp-explorer-time"
              />
            </div>
          </div>

          <!-- End -->
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">End</label>
            <div class="flex gap-2">
              <VueDatePicker
                :model-value="endDate"
                @update:model-value="onEndDateChange"
                :time-config="{ enableTimePicker: false }"
                auto-apply
                :dark="true"
                :formats="dateFormats"
                model-type="yyyy-MM-dd"
                input-class-name="dp-dark-input dp-explorer-date"
              />
              <VueDatePicker
                :model-value="endTimeObj"
                @update:model-value="onEndTimeChange"
                time-picker
                :dark="true"
                :is-24="true"
                :time-config="{ minutesIncrement: 5, minutesGridIncrement: 5 }"
                input-class-name="dp-dark-input dp-explorer-time"
              />
            </div>
          </div>

          <!-- Validation -->
          <p v-if="rangeWarning" class="text-amber-400 text-xs flex items-start gap-1">
            <svg class="w-3.5 h-3.5 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
            </svg>
            {{ rangeWarning }}
          </p>
        </div>

        <!-- ---- Load button ---- -->
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
          <span v-if="isLoading">Loading {{ loadProgress.loaded }}/{{ loadProgress.total }}</span>
          <span v-else>Load Data</span>
        </button>

        <!-- ---- Radar Layers ---- -->
        <div class="space-y-2">
          <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Layers</h3>

          <div
            v-for="product in productOrder"
            :key="product"
            class="bg-gray-800 rounded-lg p-3 space-y-2"
          >
            <!-- Header: checkbox + short name + unit + order arrows -->
            <div class="flex items-center gap-2">
              <input
                type="checkbox"
                :id="`layer-${product}`"
                v-model="layerConfig[product].enabled"
                class="w-4 h-4 rounded accent-blue-500 cursor-pointer flex-shrink-0"
              />
              <label :for="`layer-${product}`" class="text-white text-sm font-bold cursor-pointer flex-1">
                {{ SHORT_NAMES[product] }}
              </label>
              <span class="text-gray-400 text-xs mr-1">{{ radarProducts[product]?.unit || '' }}</span>
              <!-- Layer order arrows -->
              <div class="flex flex-col gap-0.5">
                <button
                  @click="moveProductUp(product)"
                  :disabled="productOrder.indexOf(product) === 0"
                  class="w-5 h-4 flex items-center justify-center rounded text-gray-400
                         hover:text-white hover:bg-white/10 disabled:opacity-25 disabled:cursor-not-allowed
                         transition-colors leading-none text-[10px]"
                  title="Move layer up (toward top)"
                >▲</button>
                <button
                  @click="moveProductDown(product)"
                  :disabled="productOrder.indexOf(product) === productOrder.length - 1"
                  class="w-5 h-4 flex items-center justify-center rounded text-gray-400
                         hover:text-white hover:bg-white/10 disabled:opacity-25 disabled:cursor-not-allowed
                         transition-colors leading-none text-[10px]"
                  title="Move layer down (toward bottom)"
                >▼</button>
              </div>
            </div>

            <!-- Opacity slider (fix #11: min=0) -->
            <div class="flex items-center gap-2">
              <span class="text-gray-400 text-xs w-12 flex-shrink-0">Opacity</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                v-model.number="layerConfig[product].opacity"
                class="flex-1 h-1 accent-blue-400 cursor-pointer"
              />
              <span class="text-gray-400 text-xs w-8 text-right tabular-nums">
                {{ Math.round(layerConfig[product].opacity * 100) }}%
              </span>
            </div>

            <!-- Data stats after load -->
            <div v-if="productStats[product]" class="text-xs">
              <span class="text-green-400 font-medium">{{ productStats[product].found }}</span>
              <span class="text-gray-500">/{{ productStats[product].expected }} frames</span>
              <button
                v-if="productStats[product].missingTs.length > 0"
                @click="toggleMissing(product)"
                class="text-amber-400 hover:text-amber-300 ml-1.5 underline underline-offset-2"
              >
                {{ productStats[product].missingTs.length }} missing
                {{ showMissingFor === product ? '▲' : '▼' }}
              </button>
            </div>

            <!-- Missing frames list (fix #12) -->
            <div
              v-if="showMissingFor === product && productStats[product]?.missingTs.length"
              class="mt-1 space-y-0.5 max-h-36 overflow-y-auto rounded bg-black/30 px-2 py-1.5"
            >
              <div
                v-for="ts in productStats[product].missingTs"
                :key="ts"
                class="font-mono text-[10px] text-amber-300/80"
              >
                {{ formatMissingTs(ts) }}
              </div>
            </div>
          </div>
        </div>

        <!-- ---- Summary after load ---- -->
        <div v-if="isLoaded" class="bg-gray-800 rounded-lg p-3 text-xs text-gray-400 space-y-1">
          <div class="flex justify-between">
            <span>Frames</span>
            <span class="text-white font-medium">{{ timestamps.length }}</span>
          </div>
          <div class="flex justify-between">
            <span>Duration</span>
            <span class="text-white font-medium">{{ durationDisplay }}</span>
          </div>
        </div>

      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onUnmounted } from 'vue'
import { VueDatePicker } from '@vuepic/vue-datepicker'
import '@vuepic/vue-datepicker/dist/main.css'
import { useConfigStore } from '../stores/config.js'
import api from '../api.js'
import RadarMap from '../components/RadarMap.vue'
import ColorBar from '../components/ColorBar.vue'

const configStore = useConfigStore()

// ---- Map ref ----
const radarMap = ref(null)
const sidebarOpen = ref(false)

// ---- Short display names (fix #1) ----
const SHORT_NAMES = { SRI_adj: 'SRI', VMI: 'VMI', ETM: 'ETM', VIL: 'VIL', IR_108: 'IR' }
// Ordered top-to-bottom on the map (index 0 = topmost layer). IR_108 is last = bottommost.
const productOrder = ref(['SRI_adj', 'VMI', 'ETM', 'VIL', 'IR_108'])

// ---- Speeds (cycle button, fix #6) ----
const speeds = [0.5, 1, 2, 4]
const playSpeed = ref(1)
function cycleSpeed() {
  const idx = speeds.indexOf(playSpeed.value)
  playSpeed.value = speeds[(idx + 1) % speeds.length]
  if (isPlaying.value) { stopAnimation(); startAnimation() }
}

// ---- Date/time state (VueDatePicker pattern from MetricsView) ----
function todayStr() {
  const now = new Date()
  return `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')}`
}
// Default: today 00:00 → 02:00
const startDateTime = ref(`${todayStr()}T00:00`)
const endDateTime   = ref(`${todayStr()}T02:00`)

function parseParts(isoStr) {
  if (!isoStr || !isoStr.includes('T')) return { date: '', hour: '00', minute: '00' }
  const [date, time] = isoStr.split('T')
  const [hour, minute] = (time || '00:00').split(':')
  return { date, hour: hour || '00', minute: minute || '00' }
}
const startDate = computed(() => parseParts(startDateTime.value).date || null)
const endDate   = computed(() => parseParts(endDateTime.value).date || null)
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

// ---- Layer config: all enabled by default (fix #3) ----
const layerConfig = ref({
  SRI_adj: { enabled: true, opacity: 0.8 },
  VMI:     { enabled: true, opacity: 0.7 },
  ETM:     { enabled: true, opacity: 0.7 },
  VIL:     { enabled: true, opacity: 0.7 },
  IR_108:  { enabled: true, opacity: 0.75 },
})

// ---- Animation state ----
const timestamps    = ref([])
const frameIndex    = ref(0)
const isPlaying     = ref(false)
const isLoading     = ref(false)
const isLoaded      = ref(false)
const loadProgress  = ref({ loaded: 0, total: 0 })
const productStats  = ref({})   // { product: { found, expected, missingTs: string[] } }
const showMissingFor = ref(null)

let playInterval = null

// ---- Computed ----
const radarProducts = computed(() => configStore.radarProducts)

// Products that have been loaded and are currently enabled (for colorbars)
const visibleProducts = computed(() =>
  isLoaded.value ? productOrder.value.filter(p => layerConfig.value[p].enabled) : []
)

const rangeWarning = computed(() => {
  if (!startDateTime.value || !endDateTime.value) return null
  const s = new Date(startDateTime.value)
  const e = new Date(endDateTime.value)
  if (isNaN(s) || isNaN(e)) return 'Invalid date'
  if (e <= s) return 'End must be after start'
  if ((e - s) / 3600000 > 12) return 'Range cannot exceed 12 hours'
  return null
})

// Full date display: "04/03/2026 - 05:40" (fix #10)
const currentTimestampDisplay = computed(() => {
  if (!timestamps.value.length) return '--/--/---- - --:--'
  const ts = timestamps.value[frameIndex.value]
  if (!ts) return '--/--/---- - --:--'
  // ISO timestamp from Python is naive (no tz), browser parses as local time
  const dt = new Date(ts)
  const d = String(dt.getDate()).padStart(2, '0')
  const m = String(dt.getMonth() + 1).padStart(2, '0')
  const y = dt.getFullYear()
  const H = String(dt.getHours()).padStart(2, '0')
  const M = String(dt.getMinutes()).padStart(2, '0')
  return `${d}/${m}/${y} - ${H}:${M}`
})

const durationDisplay = computed(() => {
  if (timestamps.value.length < 2) return '--'
  const mins = Math.round(
    (new Date(timestamps.value.at(-1)) - new Date(timestamps.value[0])) / 60000
  )
  return mins >= 60 ? `${Math.floor(mins / 60)}h ${mins % 60}m` : `${mins}m`
})

// Hour-tick marks for the slider
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

// ---- goToFrame: unified frame navigation with per-product opacities ----
function goToFrame(idx) {
  frameIndex.value = idx
  if (!radarMap.value || !isLoaded.value) return
  // Build opacity map: 0 for disabled products (fix #5)
  const opacities = {}
  for (const product of productOrder.value) {
    opacities[product] = layerConfig.value[product].enabled
      ? layerConfig.value[product].opacity
      : 0
  }
  radarMap.value.showAllAtFrame(idx, opacities)
}

// Watch layerConfig (enabled + opacity) and re-render current frame (fixes #5 and opacity)
watch(layerConfig, () => {
  if (!isLoaded.value || timestamps.value.length === 0) return
  goToFrame(frameIndex.value)
}, { deep: true })

// ---- Load data ----
async function loadData() {
  if (rangeWarning.value) return
  isLoading.value = true
  isLoaded.value = false
  isPlaying.value = false
  stopAnimation()
  timestamps.value = []
  productStats.value = {}
  showMissingFor.value = null
  loadProgress.value = { loaded: 0, total: 0 }

  radarMap.value?.clearAllProducts()

  const start = startDateTime.value
  const end   = endDateTime.value

  try {
    // Fetch timestamps for ALL 4 products in parallel (fix #3)
    const results = await Promise.all(
      productOrder.value.map(product =>
        api.explorerTimestamps(start, end, product).catch(() => ({
          timestamps: [], missing: [], total_expected: 0, total_found: 0, product,
        }))
      )
    )

    // Unified sorted timestamp set (union of all found timestamps)
    const tsSet = new Set()
    results.forEach(r => r.timestamps.forEach(ts => tsSet.add(ts)))
    const sortedTs = Array.from(tsSet).sort()
    timestamps.value = sortedTs

    // Store per-product stats
    results.forEach((r, i) => {
      const product = productOrder.value[i]
      const missingSet = new Set(r.missing)
      productStats.value[product] = {
        found:     r.total_found,
        expected:  r.total_expected,
        missingTs: r.missing,   // array of ISO strings
        missingSet,
      }
    })

    if (sortedTs.length === 0) { isLoading.value = false; return }

    // Load all 4 products in parallel (fix #4: per-product generation in RadarMap)
    loadProgress.value.total = productOrder.value.length * sortedTs.length
    await Promise.all(productOrder.value.map(async (product) => {
      const stats = productStats.value[product]
      const urls = sortedTs.map(ts =>
        stats?.missingSet?.has(ts) ? null : api.explorerOverlayUrl(product, ts)
      )
      await radarMap.value?.loadProductFrames(product, urls, layerConfig.value[product].opacity)
      loadProgress.value.loaded += sortedTs.length
    }))

    isLoaded.value = true
    frameIndex.value = 0
    goToFrame(0)
    // Apply stacking order: IR_108 is last in productOrder → bottommost on map
    radarMap.value?.setProductOrder(productOrder.value)

  } catch (e) {
    console.error('Failed to load explorer data:', e)
  } finally {
    isLoading.value = false
  }
}

// ---- Layer reordering ----
function moveProductUp(product) {
  const arr = [...productOrder.value]
  const i = arr.indexOf(product)
  if (i <= 0) return
  arr.splice(i, 1)
  arr.splice(i - 1, 0, product)
  productOrder.value = arr
}

function moveProductDown(product) {
  const arr = [...productOrder.value]
  const i = arr.indexOf(product)
  if (i >= arr.length - 1) return
  arr.splice(i, 1)
  arr.splice(i + 1, 0, product)
  productOrder.value = arr
}

watch(productOrder, () => {
  if (radarMap.value && isLoaded.value) {
    radarMap.value.setProductOrder(productOrder.value)
    goToFrame(frameIndex.value)
  }
})

// ---- Slider ----
function onSliderInput(e) {
  goToFrame(Number(e.target.value))
}

// ---- Animation ----
function togglePlay() {
  if (isPlaying.value) stopAnimation(); else startAnimation()
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

// ---- Missing frames helpers (fix #12) ----
function toggleMissing(product) {
  showMissingFor.value = showMissingFor.value === product ? null : product
}
function formatMissingTs(isoTs) {
  // Format as the actual filename on disk: DD-MM-YYYY-HH-MM.hdf
  const dt = new Date(isoTs)
  const pad = n => String(n).padStart(2, '0')
  return `${pad(dt.getDate())}-${pad(dt.getMonth()+1)}-${dt.getFullYear()}-${pad(dt.getHours())}-${pad(dt.getMinutes())}.hdf`
}

onUnmounted(stopAnimation)
</script>

<style scoped>
/* Timeline slider */
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
/* VueDatePicker dark input — compact for sidebar */
.dp-explorer-date {
  width: 118px !important;
  height: 36px !important;
  border-radius: 0.5rem !important;
  border: 1px solid rgba(255, 255, 255, 0.12) !important;
  background: rgba(255, 255, 255, 0.06) !important;
  color: white !important;
  font-size: 0.8rem !important;
  padding: 0 0.5rem !important;
}
.dp-explorer-time {
  width: 96px !important;
  height: 36px !important;
  border-radius: 0.5rem !important;
  border: 1px solid rgba(255, 255, 255, 0.12) !important;
  background: rgba(255, 255, 255, 0.06) !important;
  color: white !important;
  font-size: 0.8rem !important;
  padding: 0 0.4rem !important;
}
.dp-explorer-date:focus,
.dp-explorer-time:focus {
  border-color: #60a5fa !important;
  outline: none !important;
}
</style>
