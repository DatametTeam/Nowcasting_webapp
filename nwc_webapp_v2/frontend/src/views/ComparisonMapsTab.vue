<!--
  ComparisonMapsTab.vue — Synchronized multi-panel Leaflet map comparison.

  Two modes:
  - Single T: user picks one init timestamp; slider = lead time +5→+60 min.
    Groundtruth shows actual radar at T + lead_time.
    Each model shows forecast made at T for T + lead_time.
  - Range: user picks start/end and a fixed lead time.
    Slider moves through valid times (T_valid = start → end, 5-min steps).
    Groundtruth shows radar at T_valid.
    Each model shows prediction made at T_valid - lead_time targeting T_valid.

  FSS sidebar: slide-in panel with point FSS for current frame + range mean
  (range mode only). Data from /api/fss/lookup — pure CSV lookup, no computation.
-->
<template>
  <div class="flex flex-col" style="height: calc(100vh - 3.5rem)">

    <!-- ================================================================ -->
    <!-- CONFIG BAR                                                        -->
    <!-- ================================================================ -->
    <div class="bg-gradient-to-b from-gray-900 to-gray-800 px-4 py-3 shadow-lg flex-shrink-0">
      <div class="flex flex-wrap items-end gap-3">

        <!-- Mode toggle -->
        <div>
          <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Mode</label>
          <div class="flex rounded-lg overflow-hidden ring-1 ring-white/10">
            <button @click="mode = 'single'"
              :class="['px-3 py-1.5 text-xs font-medium transition-colors',
                mode === 'single' ? 'bg-blue-600 text-white' : 'bg-white/5 text-gray-400 hover:bg-white/10']">
              Single T
            </button>
            <button @click="mode = 'range'"
              :class="['px-3 py-1.5 text-xs font-medium transition-colors',
                mode === 'range' ? 'bg-blue-600 text-white' : 'bg-white/5 text-gray-400 hover:bg-white/10']">
              Range
            </button>
          </div>
        </div>

        <!-- Mode 1: single init timestamp -->
        <div v-if="mode === 'single'">
          <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Init Time (T)</label>
          <input type="datetime-local" v-model="singleTimestamp" step="300"
            class="h-[34px] px-2.5 rounded-lg text-xs bg-white/5 ring-1 ring-white/10 text-white
                   focus:outline-none focus:ring-blue-400" />
        </div>

        <!-- Mode 2: range + fixed lead time -->
        <template v-else>
          <div>
            <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Start</label>
            <input type="datetime-local" v-model="rangeStart" step="300"
              class="h-[34px] px-2.5 rounded-lg text-xs bg-white/5 ring-1 ring-white/10 text-white
                     focus:outline-none focus:ring-blue-400" />
          </div>
          <div>
            <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">End</label>
            <input type="datetime-local" v-model="rangeEnd" step="300"
              class="h-[34px] px-2.5 rounded-lg text-xs bg-white/5 ring-1 ring-white/10 text-white
                     focus:outline-none focus:ring-blue-400" />
          </div>
          <div>
            <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Lead Time</label>
            <select v-model="selectedLeadTimeMin"
              class="h-[34px] px-2.5 rounded-lg text-xs bg-gray-800 ring-1 ring-white/10 text-white
                     focus:outline-none focus:ring-blue-400">
              <option v-for="lt in [15, 30, 45, 60]" :key="lt" :value="lt">+{{ lt }} min</option>
            </select>
          </div>
        </template>

        <!-- Divider -->
        <div class="h-[34px] w-px bg-white/10 hidden lg:block" />

        <!-- Model selector -->
        <div>
          <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Models</label>
          <div class="flex flex-wrap gap-1.5">
            <label v-for="m in models" :key="m"
              class="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium cursor-pointer
                     transition-all select-none"
              :class="selectedModels.includes(m)
                ? 'bg-blue-500/30 text-blue-300 ring-1 ring-blue-400/50'
                : 'bg-white/5 text-gray-400 hover:bg-white/10'">
              <input type="checkbox" :value="m" v-model="selectedModels" class="sr-only" />
              {{ m }}
            </label>
          </div>
        </div>

        <!-- FSS toggle -->
        <button @click="fssSidebarOpen = !fssSidebarOpen"
          class="ml-auto flex-shrink-0 h-[34px] px-3 rounded-lg text-xs font-semibold transition-colors
                 flex items-center gap-1.5 ring-1"
          :class="fssSidebarOpen
            ? 'bg-emerald-600 text-white ring-emerald-500/50'
            : 'bg-white/5 text-gray-300 ring-white/10 hover:bg-white/10'">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round"
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          FSS
        </button>

      </div>
    </div>

    <!-- ================================================================ -->
    <!-- MAP AREA + SLIDER                                                 -->
    <!-- ================================================================ -->
    <div class="flex-1 min-h-0 flex flex-col relative">

      <!-- Too many panels warning -->
      <div v-if="tooManyPanels"
        class="flex-shrink-0 px-4 py-2 bg-amber-900/50 border-b border-amber-700/40 text-xs text-amber-300">
        Max 4 panels (groundtruth + 3 models). Deselect a model to continue.
      </div>

      <!-- Empty state -->
      <div v-if="!hasValidConfig"
        class="flex-1 flex flex-col items-center justify-center text-gray-500 gap-2">
        <svg class="w-12 h-12 text-gray-600" fill="none" stroke="currentColor" stroke-width="1" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round"
            d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
        </svg>
        <p class="text-sm">Select at least one model and set a timestamp to compare</p>
      </div>

      <!-- Panels grid -->
      <div v-show="hasValidConfig && !tooManyPanels && activePanels.length > 0"
        class="flex-1 min-h-0" :class="gridClass">
        <ComparisonPanel
          v-for="panel in activePanels"
          :key="panel.id"
          :ref="el => setPanelRef(panel.id, el)"
          :label="panel.label"
          :is-groundtruth="panel.isGroundtruth"
          :overlay-url="panelOverlayUrl(panel)"
          :show-zoom="panel.isGroundtruth"
        />
      </div>

      <!-- Slider bar -->
      <div v-if="hasValidConfig && !tooManyPanels"
        class="flex-shrink-0 bg-gray-900/95 backdrop-blur-sm border-t border-white/10 px-4 py-3">
        <!-- Mode 1: lead time slider -->
        <template v-if="mode === 'single'">
          <div class="flex items-center gap-3">
            <span class="text-xs text-gray-400 w-20 flex-shrink-0">Lead time</span>
            <input type="range" v-model.number="leadTimeIdx" min="0" max="11" step="1"
              class="flex-1 accent-blue-500 cursor-pointer" />
            <span class="text-xs font-bold text-blue-300 w-16 flex-shrink-0 text-right font-mono">
              +{{ (leadTimeIdx + 1) * 5 }} min
            </span>
          </div>
          <p class="text-[11px] text-gray-500 text-center mt-1">
            Valid: {{ validTimeSingle || '—' }}
          </p>
        </template>
        <!-- Mode 2: valid time slider -->
        <template v-else>
          <div class="flex items-center gap-3">
            <span class="text-xs text-gray-400 w-20 flex-shrink-0">Valid time</span>
            <input type="range" v-model.number="rangeSliderIdx" min="0"
              :max="Math.max(0, rangeStepCount - 1)" step="1"
              :disabled="rangeStepCount === 0"
              class="flex-1 accent-blue-500 cursor-pointer" />
            <span class="text-xs font-bold text-blue-300 w-28 flex-shrink-0 text-right font-mono">
              {{ validTimeRange || '—' }}
            </span>
          </div>
          <p class="text-[11px] text-gray-500 text-center mt-1">
            Init: {{ initTimeRange || '—' }} · +{{ selectedLeadTimeMin }} min
          </p>
        </template>
      </div>

      <!-- ============================================================== -->
      <!-- FSS SIDEBAR                                                     -->
      <!-- ============================================================== -->
      <Transition name="slide-right">
        <div v-if="fssSidebarOpen"
          class="absolute top-0 right-0 h-full w-72 bg-gray-900/97 backdrop-blur-md
                 border-l border-white/10 overflow-y-auto z-[2000] shadow-2xl flex flex-col">

          <!-- Sidebar header -->
          <div class="flex-shrink-0 px-4 py-3 border-b border-white/10 flex items-center justify-between">
            <h3 class="text-sm font-bold text-white">FSS Metrics</h3>
            <div class="flex items-center gap-2">
              <!-- Scale selector -->
              <div class="flex rounded overflow-hidden ring-1 ring-white/10">
                <button v-for="s in [1, 5, 20]" :key="s" @click="fssScale = s"
                  :class="['px-2 py-0.5 text-xs font-medium transition-colors',
                    fssScale === s ? 'bg-blue-600 text-white' : 'bg-white/5 text-gray-400 hover:bg-white/10']">
                  {{ s }}km
                </button>
              </div>
              <button @click="fssSidebarOpen = false"
                class="text-gray-400 hover:text-white text-xl leading-none w-6 h-6 flex items-center justify-center">
                &times;
              </button>
            </div>
          </div>

          <!-- Sidebar body -->
          <div class="flex-1 overflow-y-auto p-4">
            <div v-if="fssLoading" class="flex items-center gap-2 text-xs text-gray-400 py-8 justify-center">
              <svg class="animate-spin w-4 h-4" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Loading FSS...
            </div>
            <div v-else-if="!fssData"
              class="text-xs text-gray-500 text-center py-8">
              Move the slider to see FSS values
            </div>
            <template v-else>
              <!-- Current frame -->
              <p class="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-2">
                Current frame
                <span class="text-gray-600 font-normal normal-case tracking-normal ml-1">+{{ currentLeadTimeMin }} min</span>
              </p>
              <template v-if="fssData.point">
                <div v-for="model in fssData.models" :key="'p-' + model" class="mb-3">
                  <p class="text-xs font-semibold mb-1"
                    :class="selectedModels.includes(model) ? 'text-blue-300' : 'text-gray-500'">
                    {{ model }}
                  </p>
                  <div class="flex gap-1">
                    <div v-for="thr in fssData.thresholds" :key="thr"
                      class="flex-1 rounded px-1 py-1.5 text-center"
                      :class="fssCellClass(fssData.point[model]?.[`thr${thr}`])">
                      <div class="text-[9px] text-gray-400 mb-0.5">{{ thr }}mm</div>
                      <div class="text-xs font-bold font-mono">
                        {{ formatFss(fssData.point[model]?.[`thr${thr}`]) }}
                      </div>
                    </div>
                  </div>
                </div>
              </template>
              <p v-else class="text-xs text-gray-600 italic">No FSS data for this timestamp</p>

              <!-- Range mean (range mode only) -->
              <template v-if="mode === 'range' && fssData.range_mean">
                <div class="border-t border-white/10 mt-4 pt-4">
                  <p class="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-2">
                    Range mean
                  </p>
                  <div v-for="model in fssData.models" :key="'r-' + model" class="mb-3">
                    <p class="text-xs font-semibold mb-1"
                      :class="selectedModels.includes(model) ? 'text-blue-300' : 'text-gray-500'">
                      {{ model }}
                    </p>
                    <div class="flex gap-1">
                      <div v-for="thr in fssData.thresholds" :key="thr"
                        class="flex-1 rounded px-1 py-1.5 text-center"
                        :class="fssCellClass(fssData.range_mean[model]?.[`thr${thr}`])">
                        <div class="text-[9px] text-gray-400 mb-0.5">{{ thr }}mm</div>
                        <div class="text-xs font-bold font-mono">
                          {{ formatFss(fssData.range_mean[model]?.[`thr${thr}`]) }}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </template>

            </template>
          </div>
        </div>
      </Transition>

    </div>
  </div>
</template>

<script setup>
import { ref, computed, reactive, watch, nextTick, onUnmounted } from 'vue'
import ComparisonPanel from '../components/ComparisonPanel.vue'
import api from '../api.js'
import { useConfigStore } from '../stores/config.js'

const configStore = useConfigStore()
const models = computed(() => configStore.models.filter(m => m.toUpperCase() !== 'TEST'))

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const mode = ref('single')

// Mode 1
const singleTimestamp = ref('')    // datetime-local value: "YYYY-MM-DDTHH:MM"
const leadTimeIdx = ref(5)         // 0–11, default = +30 min

// Mode 2
const rangeStart = ref('')
const rangeEnd = ref('')
const selectedLeadTimeMin = ref(30)
const rangeSliderIdx = ref(0)

// Common
const selectedModels = ref([])

// FSS
const fssSidebarOpen = ref(false)
const fssScale = ref(5)
const fssData = ref(null)
const fssLoading = ref(false)
let fssFetchTimeout = null

// ---------------------------------------------------------------------------
// Panel management
// ---------------------------------------------------------------------------
const MAX_PANELS = 4  // 1 groundtruth + 3 models

const tooManyPanels = computed(() => selectedModels.value.length > MAX_PANELS - 1)

const activePanels = computed(() => {
  if (tooManyPanels.value) return []
  const modelPanels = selectedModels.value.map(m => ({ id: m, label: m, isGroundtruth: false }))
  if (modelPanels.length === 0) return []
  return [
    { id: 'groundtruth', label: 'Groundtruth', isGroundtruth: true },
    ...modelPanels,
  ]
})

const gridClass = computed(() => {
  const n = activePanels.value.length
  if (n <= 2) return 'grid grid-cols-2'
  if (n === 3) return 'grid grid-cols-3'
  return 'grid grid-cols-2 grid-rows-2'  // 4 panels
})

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
const hasValidConfig = computed(() => {
  if (selectedModels.value.length === 0) return false
  if (mode.value === 'single') return !!singleTimestamp.value
  return !!(rangeStart.value && rangeEnd.value && rangeStart.value < rangeEnd.value)
})

// ---------------------------------------------------------------------------
// Time helpers
// ---------------------------------------------------------------------------
function parseLocalDt(localStr) {
  // "YYYY-MM-DDTHH:MM" → Date object
  if (!localStr) return null
  return new Date(localStr)
}

function fmtDisplay(dt) {
  if (!dt) return null
  const hh = String(dt.getHours()).padStart(2, '0')
  const mm = String(dt.getMinutes()).padStart(2, '0')
  const dd = String(dt.getDate()).padStart(2, '0')
  const mo = String(dt.getMonth() + 1).padStart(2, '0')
  return `${dd}/${mo} ${hh}:${mm}`
}

function fmtIso(dt) {
  if (!dt) return null
  // Return local ISO string without Z suffix (backend uses same timezone as data)
  const yyyy = dt.getFullYear()
  const mo = String(dt.getMonth() + 1).padStart(2, '0')
  const dd = String(dt.getDate()).padStart(2, '0')
  const hh = String(dt.getHours()).padStart(2, '0')
  const mm = String(dt.getMinutes()).padStart(2, '0')
  return `${yyyy}-${mo}-${dd}T${hh}:${mm}`
}

function addMin(dt, minutes) {
  return new Date(dt.getTime() + minutes * 60000)
}

// Mode 1 — valid time shown below slider
const validTimeSingle = computed(() => {
  const base = parseLocalDt(singleTimestamp.value)
  if (!base) return null
  const offsetMin = (leadTimeIdx.value + 1) * 5
  return fmtDisplay(addMin(base, offsetMin))
})

// Mode 2 — number of 5-min steps in the range
const rangeStepCount = computed(() => {
  const s = parseLocalDt(rangeStart.value)
  const e = parseLocalDt(rangeEnd.value)
  if (!s || !e || e <= s) return 0
  return Math.floor((e - s) / (5 * 60000)) + 1
})

// Mode 2 — current valid time
const currentValidTimeDt = computed(() => {
  const s = parseLocalDt(rangeStart.value)
  if (!s) return null
  return addMin(s, rangeSliderIdx.value * 5)
})

const validTimeRange = computed(() => fmtDisplay(currentValidTimeDt.value))

// Mode 2 — init time for current frame
const currentInitTimeDt = computed(() => {
  const v = currentValidTimeDt.value
  if (!v) return null
  return addMin(v, -selectedLeadTimeMin.value)
})

const initTimeRange = computed(() => fmtDisplay(currentInitTimeDt.value))

const currentLeadTimeMin = computed(() =>
  mode.value === 'single' ? (leadTimeIdx.value + 1) * 5 : selectedLeadTimeMin.value
)

// ---------------------------------------------------------------------------
// Overlay URL per panel
// ---------------------------------------------------------------------------
function panelOverlayUrl(panel) {
  if (!hasValidConfig.value) return null

  if (mode.value === 'single') {
    const base = parseLocalDt(singleTimestamp.value)
    if (!base) return null
    const ltMin = (leadTimeIdx.value + 1) * 5
    if (panel.isGroundtruth) {
      // Actual radar at T + lead_time
      return api.groundtruthOverlayUrl(fmtIso(addMin(base, ltMin)), 'SRI_adj')
    } else {
      // Prediction made at T, lead_time index ltIdx
      return api.overlayUrl(panel.id, fmtIso(base), leadTimeIdx.value)
    }
  } else {
    // Range mode
    const validDt = currentValidTimeDt.value
    if (!validDt) return null
    const ltMin = selectedLeadTimeMin.value
    const ltIdx = ltMin / 5 - 1  // e.g. +30min → idx 5
    if (panel.isGroundtruth) {
      return api.groundtruthOverlayUrl(fmtIso(validDt), 'SRI_adj')
    } else {
      const initDt = addMin(validDt, -ltMin)
      return api.overlayUrl(panel.id, fmtIso(initDt), ltIdx)
    }
  }
}

// ---------------------------------------------------------------------------
// Leaflet.Sync wiring
// ---------------------------------------------------------------------------
const panelRefs = reactive({})  // { panelId: ComparisonPanel instance }

function setPanelRef(id, el) {
  if (el) panelRefs[id] = el
  else delete panelRefs[id]
}

// Re-wire sync whenever the active panel set changes
watch(activePanels, async (panels) => {
  await nextTick()

  const maps = panels
    .map(p => panelRefs[p.id]?.getMap())
    .filter(Boolean)

  // Unsync all existing pairs first
  maps.forEach(m1 => maps.forEach(m2 => {
    if (m1 !== m2) m1.unsync?.(m2)
  }))

  // Re-sync all pairs
  maps.forEach(m1 => maps.forEach(m2 => {
    if (m1 !== m2) m1.sync(m2)
  }))
}, { flush: 'post' })

// Reset slider when range changes
watch([rangeStart, rangeEnd], () => { rangeSliderIdx.value = 0 })

// ---------------------------------------------------------------------------
// FSS fetching
// ---------------------------------------------------------------------------
function scheduleFssFetch() {
  if (!fssSidebarOpen.value || !hasValidConfig.value) return
  clearTimeout(fssFetchTimeout)
  fssFetchTimeout = setTimeout(fetchFss, 350)
}

async function fetchFss() {
  if (!fssSidebarOpen.value || !hasValidConfig.value) return

  const params = new URLSearchParams({
    lt: String(currentLeadTimeMin.value),
    scale: String(fssScale.value),
  })

  if (mode.value === 'single') {
    const base = parseLocalDt(singleTimestamp.value)
    if (!base) return
    params.set('ts', fmtIso(base))
  } else {
    const initDt = currentInitTimeDt.value
    if (!initDt) return
    params.set('ts', fmtIso(initDt))
    // Range mean: init times span from (rangeStart - lt) to (rangeEnd - lt)
    const s = parseLocalDt(rangeStart.value)
    const e = parseLocalDt(rangeEnd.value)
    if (s && e) {
      params.set('start', fmtIso(addMin(s, -selectedLeadTimeMin.value)))
      params.set('end', fmtIso(addMin(e, -selectedLeadTimeMin.value)))
    }
  }

  fssLoading.value = true
  try {
    fssData.value = await api.fssLookup(params)
  } catch {
    fssData.value = null
  } finally {
    fssLoading.value = false
  }
}

// Trigger FSS fetch whenever relevant state changes
watch([leadTimeIdx, rangeSliderIdx, fssScale, mode, selectedLeadTimeMin], scheduleFssFetch)
watch(fssSidebarOpen, (open) => { if (open) fetchFss() })
watch([singleTimestamp, rangeStart, rangeEnd], () => {
  fssData.value = null
  scheduleFssFetch()
})

// ---------------------------------------------------------------------------
// FSS display helpers
// ---------------------------------------------------------------------------
function formatFss(val) {
  if (val === null || val === undefined) return '—'
  return val.toFixed(2)
}

function fssCellClass(val) {
  if (val === null || val === undefined) return 'bg-white/5 text-gray-600'
  if (val >= 0.6) return 'bg-green-900/60 text-green-300'
  if (val >= 0.3) return 'bg-yellow-900/60 text-yellow-300'
  return 'bg-red-900/60 text-red-400'
}

onUnmounted(() => clearTimeout(fssFetchTimeout))
</script>

<style scoped>
.slide-right-enter-active {
  transition: transform 0.25s ease-out, opacity 0.2s ease-out;
}
.slide-right-leave-active {
  transition: transform 0.2s ease-in, opacity 0.15s ease-in;
}
.slide-right-enter-from,
.slide-right-leave-to {
  transform: translateX(100%);
  opacity: 0;
}

/* Keep grid cells from shrinking below 0 — required for Leaflet to show */
:deep(.leaflet-container) {
  min-height: 100%;
}

/* Disable native range track on webkit for a slicker look */
input[type="range"] {
  height: 4px;
}
</style>
