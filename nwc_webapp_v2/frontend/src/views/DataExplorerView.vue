<!--
  DataExplorerView.vue — Historical radar data browser.

  Lets users explore SRI_adj, VMI, ETM, and VIL radar layers over a
  selected date range (max 12 hours), with:
  - A timeline slider to animate through frames
  - Per-layer enable/disable and opacity control
  - Stacked colorbars for each active layer
  - Missing-data indicators

  No predictions or AI — pure radar data exploration.
-->
<template>
  <div class="h-[calc(100vh-3.5rem)] flex">

    <!-- ================================================================ -->
    <!-- LEFT: Map area                                                    -->
    <!-- ================================================================ -->
    <div class="flex-1 flex flex-col relative">
      <RadarMap
        ref="radarMap"
        class="flex-1"
      />

      <!-- Sidebar toggle (mobile) -->
      <button
        v-if="!sidebarOpen"
        @click="sidebarOpen = true"
        class="absolute top-3 right-3 z-[1001] lg:hidden
               w-10 h-10 flex items-center justify-center rounded-full
               bg-white shadow-lg border border-gray-200 text-gray-600
               hover:bg-gray-50 transition-colors"
        title="Open panel"
      >
        <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path d="M4 6h16M4 12h16M4 18h16" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
      </button>

      <!-- Stacked colorbars — bottom right, above timeline bar -->
      <div class="absolute bottom-[110px] right-[10px] z-[1001] flex flex-col gap-2 items-end">
        <div v-for="product in activeProducts" :key="product" class="flex flex-col items-center">
          <div class="text-white text-[9px] font-bold mb-0.5 bg-black/50 px-1 rounded">{{ product }}</div>
          <ColorBar :legend="radarProducts[product]" />
        </div>
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
        <!-- Current timestamp display -->
        <div class="flex items-center justify-between text-white mb-2">
          <div class="text-sm font-medium hidden sm:block text-gray-300">
            {{ activeProducts.length ? activeProducts.join(' + ') : 'No layers selected' }}
          </div>
          <div class="text-center">
            <span class="text-lg sm:text-2xl font-bold tabular-nums">
              {{ currentTimestampDisplay }}
            </span>
            <span class="ml-2 text-xs font-medium px-2 py-0.5 rounded-full bg-blue-500/30 text-blue-300">
              radar
            </span>
          </div>
          <!-- Speed control -->
          <div class="flex items-center gap-2 text-sm">
            <span class="text-gray-300 text-xs hidden sm:inline">Speed</span>
            <select
              v-model="playSpeed"
              class="bg-white/10 text-white text-xs rounded px-1 py-0.5 border border-white/20 outline-none"
            >
              <option :value="0.5">0.5×</option>
              <option :value="1">1×</option>
              <option :value="2">2×</option>
              <option :value="4">4×</option>
            </select>
          </div>
        </div>

        <!-- Timeline slider row -->
        <div class="flex items-center gap-3">
          <!-- Play/Pause -->
          <button
            @click="togglePlay"
            :disabled="!isLoaded || timestamps.length === 0"
            class="w-9 h-9 flex items-center justify-center rounded-full
                   bg-white/10 hover:bg-white/20 border border-white/20
                   text-white transition-colors flex-shrink-0
                   disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <svg v-if="!isPlaying" class="w-4 h-4 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
            <svg v-else class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
            </svg>
          </button>

          <!-- Slider -->
          <div class="flex-1 relative">
            <input
              type="range"
              :min="0"
              :max="Math.max(0, timestamps.length - 1)"
              :value="frameIndex"
              @input="onSliderInput"
              class="w-full h-1.5 rounded-full appearance-none cursor-pointer timeline-slider"
              :disabled="!isLoaded || timestamps.length === 0"
            />
            <!-- Tick marks — one per hour -->
            <div v-if="hourTicks.length > 0" class="relative mt-1 h-4">
              <span
                v-for="tick in hourTicks"
                :key="tick.label"
                class="absolute text-[9px] text-gray-400 -translate-x-1/2"
                :style="{ left: tick.pct + '%' }"
              >{{ tick.label }}</span>
            </div>
          </div>

          <!-- Frame count -->
          <div class="text-xs text-gray-400 flex-shrink-0 tabular-nums">
            {{ frameIndex + 1 }} / {{ timestamps.length || 0 }}
          </div>
        </div>
      </div>
    </div>

    <!-- ================================================================ -->
    <!-- RIGHT: Sidebar panel                                             -->
    <!-- ================================================================ -->
    <div
      class="w-72 bg-gray-900 border-l border-gray-700 flex flex-col overflow-y-auto
             transition-all duration-300
             fixed inset-y-0 right-0 z-[1002] lg:relative lg:z-auto"
      :class="sidebarOpen ? 'translate-x-0' : 'translate-x-full lg:translate-x-0'"
      style="top: 3.5rem;"
    >
      <!-- Close button (mobile) -->
      <button
        @click="sidebarOpen = false"
        class="lg:hidden absolute top-3 right-3 text-gray-400 hover:text-white"
      >
        <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path d="M6 18L18 6M6 6l12 12" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
      </button>

      <div class="p-4 space-y-5 flex-1">
        <!-- ---- Title ---- -->
        <div>
          <h2 class="text-white font-bold text-base">Data Explorer</h2>
          <p class="text-gray-400 text-xs mt-0.5">Browse historical radar data</p>
        </div>

        <!-- ---- Date/Time Range ---- -->
        <div class="space-y-2">
          <label class="text-gray-300 text-xs font-semibold uppercase tracking-wide">Date Range</label>
          <div class="space-y-2">
            <div>
              <label class="text-gray-400 text-xs mb-1 block">Start</label>
              <input
                v-model="startDateTime"
                type="datetime-local"
                class="w-full bg-gray-800 text-white text-xs rounded-md px-3 py-2
                       border border-gray-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label class="text-gray-400 text-xs mb-1 block">End</label>
              <input
                v-model="endDateTime"
                type="datetime-local"
                class="w-full bg-gray-800 text-white text-xs rounded-md px-3 py-2
                       border border-gray-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
          </div>
          <!-- Validation warning -->
          <p v-if="rangeWarning" class="text-amber-400 text-xs flex items-center gap-1">
            <svg class="w-3.5 h-3.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
            </svg>
            {{ rangeWarning }}
          </p>
        </div>

        <!-- ---- Load button ---- -->
        <button
          @click="loadData"
          :disabled="isLoading || !!rangeWarning || activeProducts.length === 0"
          class="w-full py-2.5 rounded-lg text-sm font-semibold transition-colors
                 bg-blue-600 hover:bg-blue-500 text-white
                 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          <svg v-if="isLoading" class="animate-spin h-4 w-4" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <span v-if="isLoading">Loading... {{ loadProgress.loaded }}/{{ loadProgress.total }}</span>
          <span v-else>Load Data</span>
        </button>

        <!-- ---- Radar Layers ---- -->
        <div class="space-y-2">
          <label class="text-gray-300 text-xs font-semibold uppercase tracking-wide">Radar Layers</label>
          <div class="space-y-3">
            <div
              v-for="product in productOrder"
              :key="product"
              class="bg-gray-800 rounded-lg p-3 space-y-2"
            >
              <!-- Header: checkbox + label + unit -->
              <div class="flex items-center gap-2">
                <input
                  type="checkbox"
                  :id="`layer-${product}`"
                  v-model="layerConfig[product].enabled"
                  class="w-4 h-4 rounded accent-blue-500 cursor-pointer"
                />
                <label :for="`layer-${product}`" class="text-white text-sm font-medium cursor-pointer flex-1">
                  {{ radarProducts[product]?.label || product }}
                </label>
                <span class="text-gray-400 text-xs">{{ radarProducts[product]?.unit || '' }}</span>
              </div>
              <!-- Opacity slider (only when enabled) -->
              <div v-if="layerConfig[product].enabled" class="flex items-center gap-2">
                <span class="text-gray-400 text-xs w-12">Opacity</span>
                <input
                  type="range"
                  min="0.1"
                  max="1"
                  step="0.05"
                  v-model.number="layerConfig[product].opacity"
                  @input="onOpacityChange(product)"
                  class="flex-1 h-1 accent-blue-400 cursor-pointer"
                />
                <span class="text-gray-400 text-xs w-8 text-right">{{ Math.round(layerConfig[product].opacity * 100) }}%</span>
              </div>
              <!-- Per-product data availability (shown after load) -->
              <div v-if="productStats[product]" class="text-xs text-gray-400">
                <span class="text-green-400">{{ productStats[product].found }}</span>
                / {{ productStats[product].expected }} frames found
                <span v-if="productStats[product].missing > 0" class="text-amber-400 ml-1">
                  ({{ productStats[product].missing }} missing)
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- ---- Missing data summary ---- -->
        <div v-if="isLoaded && totalMissing > 0" class="bg-amber-900/30 rounded-lg p-3 space-y-1">
          <p class="text-amber-400 text-xs font-semibold">Missing Frames</p>
          <p class="text-amber-300 text-xs">
            {{ totalMissing }} timestamp(s) have incomplete data across active layers.
          </p>
        </div>

        <!-- ---- Status bar (after load) ---- -->
        <div v-if="isLoaded" class="bg-gray-800 rounded-lg p-3 text-xs text-gray-300 space-y-1">
          <div class="flex justify-between">
            <span>Timestamps</span>
            <span class="text-white font-medium">{{ timestamps.length }}</span>
          </div>
          <div class="flex justify-between">
            <span>Duration</span>
            <span class="text-white font-medium">{{ durationDisplay }}</span>
          </div>
          <div class="flex justify-between">
            <span>Active layers</span>
            <span class="text-white font-medium">{{ activeProducts.length }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useConfigStore } from '../stores/config.js'
import api from '../api.js'
import RadarMap from '../components/RadarMap.vue'
import ColorBar from '../components/ColorBar.vue'

const configStore = useConfigStore()

// ---- Map ref ----
const radarMap = ref(null)

// ---- Sidebar ----
const sidebarOpen = ref(true)

// ---- Date range ----
// Default: last 2 hours rounded to 5 minutes
function defaultDateTime(offsetMinutes = 0) {
  const now = new Date()
  now.setMinutes(Math.floor(now.getMinutes() / 5) * 5, 0, 0)
  now.setMinutes(now.getMinutes() + offsetMinutes)
  // Format as YYYY-MM-DDTHH:MM for datetime-local input
  const pad = (n) => String(n).padStart(2, '0')
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}T${pad(now.getHours())}:${pad(now.getMinutes())}`
}

const startDateTime = ref(defaultDateTime(-120))
const endDateTime = ref(defaultDateTime(0))

// ---- Product config ----
const productOrder = ['SRI_adj', 'VMI', 'ETM', 'VIL']

const layerConfig = ref({
  SRI_adj: { enabled: true, opacity: 0.8 },
  VMI:     { enabled: false, opacity: 0.7 },
  ETM:     { enabled: false, opacity: 0.7 },
  VIL:     { enabled: false, opacity: 0.7 },
})

// ---- State ----
const timestamps = ref([])        // unified sorted list of ISO timestamp strings
const frameIndex = ref(0)
const isPlaying = ref(false)
const playSpeed = ref(1)
const isLoading = ref(false)
const isLoaded = ref(false)
const loadProgress = ref({ loaded: 0, total: 0 })
const productStats = ref({})      // { product: { found, missing, expected } }

let playInterval = null

// ---- Computed ----
const radarProducts = computed(() => configStore.radarProducts)

const activeProducts = computed(() =>
  productOrder.filter(p => layerConfig.value[p].enabled)
)

const rangeWarning = computed(() => {
  if (!startDateTime.value || !endDateTime.value) return null
  const start = new Date(startDateTime.value)
  const end = new Date(endDateTime.value)
  if (isNaN(start) || isNaN(end)) return 'Invalid date'
  if (end <= start) return 'End must be after start'
  const hours = (end - start) / 3600000
  if (hours > 12) return 'Range cannot exceed 12 hours'
  return null
})

const currentTimestampDisplay = computed(() => {
  if (!timestamps.value.length) return '--:--'
  const ts = timestamps.value[frameIndex.value]
  if (!ts) return '--:--'
  const dt = new Date(ts)
  return dt.toLocaleString('it-IT', {
    timeZone: 'Europe/Rome',
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
})

const durationDisplay = computed(() => {
  if (timestamps.value.length < 2) return '--'
  const first = new Date(timestamps.value[0])
  const last = new Date(timestamps.value[timestamps.value.length - 1])
  const mins = Math.round((last - first) / 60000)
  return mins >= 60 ? `${Math.floor(mins / 60)}h ${mins % 60}m` : `${mins}m`
})

// Hour tick marks for the slider
const hourTicks = computed(() => {
  if (timestamps.value.length < 2) return []
  const ticks = []
  const seen = new Set()
  timestamps.value.forEach((ts, i) => {
    const dt = new Date(ts)
    if (dt.getMinutes() === 0) {
      const label = dt.toLocaleTimeString('it-IT', { hour: '2-digit', minute: '2-digit', timeZone: 'Europe/Rome' })
      if (!seen.has(label)) {
        seen.add(label)
        ticks.push({ label, pct: (i / (timestamps.value.length - 1)) * 100 })
      }
    }
  })
  return ticks
})

const totalMissing = computed(() => {
  return Object.values(productStats.value).reduce((sum, s) => sum + (s.missing || 0), 0)
})

// ---- Methods ----

async function loadData() {
  if (rangeWarning.value || activeProducts.value.length === 0) return

  isLoading.value = true
  isLoaded.value = false
  isPlaying.value = false
  stopAnimation()
  timestamps.value = []
  productStats.value = {}
  loadProgress.value = { loaded: 0, total: 0 }

  if (radarMap.value) {
    radarMap.value.clearAllProducts()
  }

  const start = startDateTime.value
  const end = endDateTime.value

  try {
    // Fetch timestamps for each active product in parallel
    const results = await Promise.all(
      activeProducts.value.map(product =>
        api.explorerTimestamps(start, end, product).catch(() => ({
          timestamps: [], missing: [], total_expected: 0, total_found: 0, product,
        }))
      )
    )

    // Build unified sorted timestamp set (union of all found timestamps)
    const tsSet = new Set()
    results.forEach(r => r.timestamps.forEach(ts => tsSet.add(ts)))
    const sortedTs = Array.from(tsSet).sort()
    timestamps.value = sortedTs

    // Store per-product stats
    results.forEach((r, i) => {
      const product = activeProducts.value[i]
      productStats.value[product] = {
        found: r.total_found,
        missing: r.missing.length,
        expected: r.total_expected,
        missingSet: new Set(r.missing),
      }
    })

    if (sortedTs.length === 0) {
      isLoading.value = false
      return
    }

    // Build URL arrays for each active product (null where file is missing)
    loadProgress.value.total = activeProducts.value.length * sortedTs.length
    const loadPromises = activeProducts.value.map(async (product) => {
      const stats = productStats.value[product]
      const urls = sortedTs.map(ts => {
        // If this timestamp is known missing, pass null (RadarMap skips it)
        if (stats?.missingSet?.has(ts)) return null
        return api.explorerOverlayUrl(product, ts)
      })
      await radarMap.value?.loadProductFrames(
        product,
        urls,
        layerConfig.value[product].opacity,
      )
      loadProgress.value.loaded += sortedTs.length
    })

    await Promise.all(loadPromises)

    frameIndex.value = 0
    radarMap.value?.showAllAtFrame(0)
    isLoaded.value = true
  } catch (e) {
    console.error('Failed to load explorer data:', e)
  } finally {
    isLoading.value = false
  }
}

function onSliderInput(e) {
  const idx = Number(e.target.value)
  frameIndex.value = idx
  radarMap.value?.showAllAtFrame(idx)
}

function togglePlay() {
  if (isPlaying.value) {
    stopAnimation()
  } else {
    startAnimation()
  }
}

function startAnimation() {
  if (timestamps.value.length === 0) return
  isPlaying.value = true
  const fps = playSpeed.value * 2  // base 2 fps × speed multiplier
  playInterval = setInterval(() => {
    const next = (frameIndex.value + 1) % timestamps.value.length
    frameIndex.value = next
    radarMap.value?.showAllAtFrame(next)
  }, 1000 / fps)
}

function stopAnimation() {
  isPlaying.value = false
  if (playInterval) {
    clearInterval(playInterval)
    playInterval = null
  }
}

function onOpacityChange(product) {
  radarMap.value?.setProductOpacity(product, layerConfig.value[product].opacity)
}

// When speed changes mid-animation, restart with new interval
watch(playSpeed, () => {
  if (isPlaying.value) {
    stopAnimation()
    startAnimation()
  }
})

onUnmounted(() => {
  stopAnimation()
})
</script>

<style scoped>
/* Timeline slider track */
.timeline-slider {
  background: linear-gradient(to right, #3b82f6 0%, #3b82f6 var(--val, 0%), #4b5563 var(--val, 0%), #4b5563 100%);
}

.timeline-slider::-webkit-slider-thumb {
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: white;
  cursor: pointer;
  box-shadow: 0 1px 4px rgba(0,0,0,0.4);
}

.timeline-slider::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: white;
  cursor: pointer;
  border: none;
  box-shadow: 0 1px 4px rgba(0,0,0,0.4);
}
</style>
