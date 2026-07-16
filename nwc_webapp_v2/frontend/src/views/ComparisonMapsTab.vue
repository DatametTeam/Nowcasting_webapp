<!--
  ComparisonMapsTab.vue — Synchronized multi-panel Leaflet map comparison.

  Two modes:
  - Single T: one init timestamp; slider = lead time (+5→+60 min).
    Groundtruth shows radar at T + lead_time. Models show forecast made at T.
  - Range: start/end + fixed lead time. Slider = valid time.
    Groundtruth at T_valid. Models show prediction made at T_valid - lead_time.

  All frames are preloaded (browser Image() + hidden Leaflet overlays) before
  enabling the slider. This guarantees instant, synchronised frame switching.
-->
<template>
  <div class="flex flex-col h-full">

    <!-- ================================================================ -->
    <!-- CONFIG BAR                                                        -->
    <!-- ================================================================ -->
    <div class="flex-shrink-0 bg-gradient-to-b from-gray-900 to-gray-800 px-4 py-3 shadow-lg">
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
        <template v-if="mode === 'single'">
          <div>
            <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Date</label>
            <VueDatePicker :model-value="singlePickerDate" @update:model-value="onSingleDateChange"
              :time-config="{ enableTimePicker: false }" auto-apply :dark="true"
              :formats="dateFormats" model-type="yyyy-MM-dd" input-class-name="dp-dark-input" />
          </div>
          <div>
            <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Time</label>
            <VueDatePicker :model-value="singleTimeObj" @update:model-value="onSingleTimeChange"
              time-picker :dark="true" :is-24="true"
              :time-config="{ minutesIncrement: 5, minutesGridIncrement: 5 }"
              input-class-name="dp-dark-input dp-time-input" />
          </div>
        </template>

        <!-- Mode 2: range + fixed lead time -->
        <template v-else>
          <div>
            <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Start Date</label>
            <VueDatePicker :model-value="startPickerDate" @update:model-value="onStartDateChange"
              :time-config="{ enableTimePicker: false }" auto-apply :dark="true"
              :formats="dateFormats" model-type="yyyy-MM-dd" input-class-name="dp-dark-input" />
          </div>
          <div>
            <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Start Time</label>
            <VueDatePicker :model-value="startTimeObj" @update:model-value="onStartTimeChange"
              time-picker :dark="true" :is-24="true"
              :time-config="{ minutesIncrement: 5, minutesGridIncrement: 5 }"
              input-class-name="dp-dark-input dp-time-input" />
          </div>
          <div>
            <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">End Date</label>
            <VueDatePicker :model-value="endPickerDate" @update:model-value="onEndDateChange"
              :time-config="{ enableTimePicker: false }" auto-apply :dark="true"
              :formats="dateFormats" model-type="yyyy-MM-dd" input-class-name="dp-dark-input" />
          </div>
          <div>
            <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">End Time</label>
            <VueDatePicker :model-value="endTimeObj" @update:model-value="onEndTimeChange"
              time-picker :dark="true" :is-24="true"
              :time-config="{ minutesIncrement: 5, minutesGridIncrement: 5 }"
              input-class-name="dp-dark-input dp-time-input" />
          </div>
          <div>
            <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Lead Time</label>
            <select v-model="selectedLeadTimeMin"
              class="h-[42px] px-2.5 rounded-lg text-sm bg-gray-800 ring-1 ring-white/10 text-white
                     focus:outline-none focus:ring-blue-400">
              <option v-for="lt in FSS_LEAD_TIMES" :key="lt" :value="lt">+{{ lt }} min</option>
            </select>
          </div>
        </template>

        <!-- Divider -->
        <div class="hidden lg:block h-[42px] w-px bg-white/10" />

        <!-- Model checkboxes -->
        <div>
          <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Models</label>
          <div class="flex flex-wrap gap-1.5 h-[42px] items-center">
            <label v-for="m in models" :key="m"
              class="inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-medium
                     cursor-pointer transition-all select-none"
              :class="selectedModels.includes(m)
                ? 'bg-blue-500/30 text-blue-300 ring-1 ring-blue-400/50'
                : 'bg-white/5 text-gray-400 hover:bg-white/10'">
              <input type="checkbox" :value="m" v-model="selectedModels" class="sr-only" />
              {{ m }}
            </label>
          </div>
        </div>

        <!-- FSS sidebar toggle -->
        <button @click="toggleFss"
          class="ml-auto flex-shrink-0 h-[42px] px-4 rounded-lg text-sm font-semibold transition-colors
                 flex items-center gap-2 ring-1"
          :class="fssSidebarOpen
            ? 'bg-emerald-600 text-white ring-emerald-500/50'
            : 'bg-white/5 text-gray-300 ring-white/10 hover:bg-white/10'">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round"
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          FSS
        </button>
      </div>
    </div>

    <!-- ================================================================ -->
    <!-- MAIN: panels + FSS sidebar                                        -->
    <!-- ================================================================ -->
    <div class="flex flex-1 min-h-0">

      <!-- Left: panels grid + slider -->
      <div class="relative flex flex-col flex-1 min-w-0 min-h-0">

        <!-- Too many panels -->
        <div v-if="tooManyPanels"
          class="flex-shrink-0 px-4 py-2 bg-amber-900/50 border-b border-amber-700/40 text-xs text-amber-300">
          Max 4 panels (groundtruth + 3 models). Deselect a model.
        </div>

        <!-- Empty state -->
        <div v-if="!hasValidConfig || tooManyPanels || activePanels.length === 0"
          class="flex-1 flex flex-col items-center justify-center text-gray-500 gap-3 bg-gray-950">
          <svg class="w-12 h-12 text-gray-700" fill="none" stroke="currentColor" stroke-width="1" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round"
              d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
          </svg>
          <p class="text-sm">
            {{ tooManyPanels ? 'Too many models selected (max 3)' : 'Select a model and set a time to compare' }}
          </p>
        </div>

        <!-- Panels grid -->
        <div v-else class="flex-1 min-h-0" :class="gridClass">
          <ComparisonPanel
            v-for="panel in activePanels"
            :key="panel.id"
            :ref="el => setPanelRef(panel.id, el)"
            :label="panel.label"
            :is-groundtruth="panel.isGroundtruth"
            :show-zoom="panel.isGroundtruth"
          />
        </div>

        <!-- Preload progress overlay (covers panels + slider) -->
        <div v-if="preloading"
          class="absolute inset-0 z-[1001] bg-gray-950/85 flex flex-col items-center justify-center gap-4">
          <svg class="animate-spin w-10 h-10 text-blue-400" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <p class="text-white text-sm font-medium">
            Loading frames {{ preloadLoaded }}/{{ preloadTotal }}
          </p>
          <!-- Progress bar -->
          <div class="w-48 h-1.5 bg-white/10 rounded-full overflow-hidden">
            <div class="h-full bg-blue-400 rounded-full transition-all"
              :style="{ width: preloadTotal > 0 ? `${(preloadLoaded / preloadTotal) * 100}%` : '0%' }" />
          </div>
        </div>

        <!-- ============================================================ -->
        <!-- SLIDER BAR                                                    -->
        <!-- ============================================================ -->
        <div v-if="hasValidConfig && !tooManyPanels && activePanels.length > 0"
          class="flex-shrink-0 bg-gradient-to-b from-gray-800/95 to-gray-900/95 backdrop-blur-sm
                 border-t border-white/10 px-4 py-3">

          <!-- Mode 1: lead time slider -->
          <template v-if="mode === 'single'">
            <div class="flex items-start gap-3 sm:gap-4">
              <button @click="togglePlay"
                :disabled="preloading"
                class="w-10 h-10 flex-shrink-0 flex items-center justify-center rounded-full
                       bg-white/20 hover:bg-white/30 text-white transition-colors
                       disabled:opacity-40 disabled:cursor-not-allowed">
                <svg v-if="playing" class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="4" width="4" height="16" rx="1" /><rect x="14" y="4" width="4" height="16" rx="1" />
                </svg>
                <svg v-else class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
              </button>
              <div class="flex-1 min-w-0">
                <div class="h-10 flex items-center">
                  <input type="range" v-model.number="leadTimeIdx" min="0" max="11" step="1"
                    :disabled="preloading"
                    class="w-full h-2 appearance-none cursor-pointer rounded-full bg-white/20 accent-blue-400
                           [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4
                           [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full
                           [&::-webkit-slider-thumb]:bg-blue-400 [&::-webkit-slider-thumb]:shadow-lg
                           [&::-webkit-slider-thumb]:shadow-blue-400/50 disabled:opacity-40" />
                </div>
                <div class="flex justify-between px-0.5">
                  <span v-for="i in 12" :key="i"
                    class="text-[10px] tabular-nums w-0 text-center"
                    :class="[0, 2, 5, 8, 11].includes(i - 1) ? 'text-gray-400' : 'text-transparent'">
                    +{{ i * 5 }}
                  </span>
                </div>
              </div>
              <div class="flex-shrink-0 text-right w-24">
                <div class="text-sm font-bold text-blue-300 font-mono">+{{ (leadTimeIdx + 1) * 5 }} min</div>
                <div class="text-[11px] text-gray-400 mt-0.5">{{ validTimeSingle || '—' }}</div>
              </div>
            </div>
          </template>

          <!-- Mode 2: valid time slider -->
          <template v-else>
            <div class="flex items-start gap-3 sm:gap-4">
              <button @click="togglePlay"
                :disabled="preloading || rangeStepCount === 0"
                class="w-10 h-10 flex-shrink-0 flex items-center justify-center rounded-full
                       bg-white/20 hover:bg-white/30 text-white transition-colors
                       disabled:opacity-40 disabled:cursor-not-allowed">
                <svg v-if="playing" class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="4" width="4" height="16" rx="1" /><rect x="14" y="4" width="4" height="16" rx="1" />
                </svg>
                <svg v-else class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
              </button>
              <div class="flex-1 min-w-0">
                <div class="h-10 flex items-center">
                  <input type="range" v-model.number="rangeSliderIdx"
                    min="0" :max="Math.max(0, rangeStepCount - 1)" step="1"
                    :disabled="preloading || rangeStepCount === 0"
                    class="w-full h-2 appearance-none cursor-pointer rounded-full bg-white/20 accent-blue-400
                           [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4
                           [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full
                           [&::-webkit-slider-thumb]:bg-blue-400 [&::-webkit-slider-thumb]:shadow-lg
                           [&::-webkit-slider-thumb]:shadow-blue-400/50 disabled:opacity-40" />
                </div>
                <div class="flex justify-between text-[10px] text-gray-500 px-0.5">
                  <span>{{ fmtDisplay(parseLocalDt(startDt)) || '—' }}</span>
                  <span>{{ fmtDisplay(parseLocalDt(endDt)) || '—' }}</span>
                </div>
              </div>
              <div class="flex-shrink-0 text-right w-24">
                <div class="text-sm font-bold text-blue-300 font-mono">{{ validTimeRange || '—' }}</div>
                <div class="text-[11px] text-gray-400 mt-0.5">Init: {{ initTimeRange || '—' }}</div>
              </div>
            </div>
          </template>
        </div>

      </div>

      <!-- ============================================================== -->
      <!-- FSS SIDEBAR (inline, no overlay)                               -->
      <!-- ============================================================== -->
      <div v-if="fssSidebarOpen"
        class="w-72 flex-shrink-0 border-l border-white/10 bg-gray-900 flex flex-col overflow-hidden">

        <div class="flex-shrink-0 px-4 py-3 border-b border-white/10 flex items-center justify-between">
          <h3 class="text-sm font-bold text-white">FSS Metrics</h3>
          <div class="flex items-center gap-2">
            <div class="flex rounded overflow-hidden ring-1 ring-white/10">
              <button v-for="s in [1, 5, 20]" :key="s" @click="fssScale = s"
                :class="['px-2 py-0.5 text-xs font-medium transition-colors',
                  fssScale === s ? 'bg-blue-600 text-white' : 'bg-white/5 text-gray-400 hover:bg-white/10']">
                {{ s }}km
              </button>
            </div>
            <button @click="fssSidebarOpen = false"
              class="w-6 h-6 flex items-center justify-center text-gray-400 hover:text-white text-xl leading-none">
              &times;
            </button>
          </div>
        </div>

        <div class="flex-1 overflow-y-auto p-4">

          <!-- Lead time not in FSS set -->
          <div v-if="!fssAvailableForLt"
            class="text-center py-8 space-y-2">
            <p class="text-xs text-gray-400">
              FSS not computed at <span class="font-semibold text-white">+{{ currentLeadTimeMin }} min</span>.
            </p>
            <p class="text-[11px] text-gray-600">
              Available at: +{{ FSS_LEAD_TIMES.join(', +') }} min
            </p>
          </div>

          <template v-else>
            <div v-if="fssLoading" class="flex items-center justify-center gap-2 text-xs text-gray-400 py-8">
              <svg class="animate-spin w-4 h-4" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Loading FSS...
            </div>
            <div v-else-if="!fssData" class="text-xs text-gray-500 text-center py-8">
              Move the slider to see FSS values
            </div>
            <template v-else>

              <!-- Current frame -->
              <p class="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-3">
                Current frame
                <span class="text-gray-600 font-normal normal-case tracking-normal ml-1">
                  +{{ currentLeadTimeMin }} min
                </span>
              </p>
              <template v-if="fssData.point">
                <div v-for="model in fssData.models" :key="'p-' + model" class="mb-4">
                  <p class="text-xs font-semibold mb-1.5"
                    :class="selectedModels.includes(model) ? 'text-blue-300' : 'text-gray-600'">
                    {{ model }}
                  </p>
                  <div class="flex gap-1">
                    <div v-for="thr in fssData.thresholds" :key="thr"
                      class="flex-1 rounded px-1 py-2 text-center"
                      :class="fssCellClass(fssData.point[model]?.[`thr${thr}`])">
                      <div class="text-[9px] text-gray-400 mb-0.5">{{ thr }}mm</div>
                      <div class="text-xs font-bold font-mono">
                        {{ formatFss(fssData.point[model]?.[`thr${thr}`]) }}
                      </div>
                    </div>
                  </div>
                </div>
              </template>
              <p v-else class="text-xs text-gray-600 italic">No FSS for this timestamp</p>

              <!-- Range mean (range mode only) -->
              <template v-if="mode === 'range' && fssData.range_mean">
                <div class="border-t border-white/10 mt-2 pt-4">
                  <p class="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-3">Range mean</p>
                  <div v-for="model in fssData.models" :key="'r-' + model" class="mb-4">
                    <p class="text-xs font-semibold mb-1.5"
                      :class="selectedModels.includes(model) ? 'text-blue-300' : 'text-gray-600'">
                      {{ model }}
                    </p>
                    <div class="flex gap-1">
                      <div v-for="thr in fssData.thresholds" :key="thr"
                        class="flex-1 rounded px-1 py-2 text-center"
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
          </template>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref, computed, reactive, watch, nextTick, onUnmounted } from 'vue'
import { VueDatePicker } from '@vuepic/vue-datepicker'
import '@vuepic/vue-datepicker/dist/main.css'
import ComparisonPanel from '../components/ComparisonPanel.vue'
import api from '../api.js'
import { useConfigStore } from '../stores/config.js'

const configStore = useConfigStore()
const models = computed(() => configStore.models.filter(m => m.toUpperCase() !== 'TEST'))

const dateFormats = { input: 'dd/MM/yyyy' }

// FSS is only pre-computed at these lead times
const FSS_LEAD_TIMES = [15, 30, 45, 60]

// ---------------------------------------------------------------------------
// Mode
// ---------------------------------------------------------------------------
const mode = ref('single')

// ---------------------------------------------------------------------------
// Single-timestamp state
// ---------------------------------------------------------------------------
const singleDt = ref('')
const singlePickerDate = computed(() => singleDt.value?.split('T')[0] || null)
const singleTimeObj    = computed(() => dtToTimeObj(singleDt.value))
function onSingleDateChange(v) {
  const d = typeof v === 'string' ? v : ''
  singleDt.value = d ? `${d}T${singleDt.value?.split('T')[1] || '00:00'}` : ''
}
function onSingleTimeChange(v) {
  if (!v || !singlePickerDate.value) return
  singleDt.value = `${singlePickerDate.value}T${pad(v.hours)}:${pad(v.minutes)}`
}

// ---------------------------------------------------------------------------
// Range state
// ---------------------------------------------------------------------------
const startDt = ref('')
const endDt   = ref('')
const selectedLeadTimeMin = ref(30)
const rangeSliderIdx = ref(0)

const startPickerDate = computed(() => startDt.value?.split('T')[0] || null)
const startTimeObj    = computed(() => dtToTimeObj(startDt.value))
const endPickerDate   = computed(() => endDt.value?.split('T')[0] || null)
const endTimeObj      = computed(() => dtToTimeObj(endDt.value))

function onStartDateChange(v) {
  const d = typeof v === 'string' ? v : ''
  startDt.value = d ? `${d}T${startDt.value?.split('T')[1] || '00:00'}` : ''
}
function onStartTimeChange(v) {
  if (!v || !startPickerDate.value) return
  startDt.value = `${startPickerDate.value}T${pad(v.hours)}:${pad(v.minutes)}`
}
function onEndDateChange(v) {
  const d = typeof v === 'string' ? v : ''
  endDt.value = d ? `${d}T${endDt.value?.split('T')[1] || '00:00'}` : ''
}
function onEndTimeChange(v) {
  if (!v || !endPickerDate.value) return
  endDt.value = `${endPickerDate.value}T${pad(v.hours)}:${pad(v.minutes)}`
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------
const selectedModels = ref([])

// ---------------------------------------------------------------------------
// Time helpers
// ---------------------------------------------------------------------------
function pad(n) { return String(n).padStart(2, '0') }

function dtToTimeObj(s) {
  const time = s?.split('T')[1] || '00:00'
  const [h, m] = time.split(':')
  return { hours: parseInt(h) || 0, minutes: parseInt(m) || 0, seconds: 0 }
}

function parseLocalDt(s) {
  if (!s) return null
  const [dp, tp] = s.split('T')
  const [y, mo, d] = dp.split('-').map(Number)
  const [h, m] = (tp || '00:00').split(':').map(Number)
  return new Date(y, mo - 1, d, h, m, 0)
}

function fmtDisplay(dt) {
  if (!dt) return null
  return `${pad(dt.getDate())}/${pad(dt.getMonth() + 1)} ${pad(dt.getHours())}:${pad(dt.getMinutes())}`
}

function fmtIso(dt) {
  if (!dt) return null
  return `${dt.getFullYear()}-${pad(dt.getMonth() + 1)}-${pad(dt.getDate())}T${pad(dt.getHours())}:${pad(dt.getMinutes())}`
}

function addMin(dt, minutes) {
  return new Date(dt.getTime() + minutes * 60000)
}

// ---------------------------------------------------------------------------
// Derived time values
// ---------------------------------------------------------------------------
const leadTimeIdx = ref(5)  // 0-11, default +30 min

const currentLeadTimeMin = computed(() =>
  mode.value === 'single' ? (leadTimeIdx.value + 1) * 5 : selectedLeadTimeMin.value
)

const validTimeSingle = computed(() => {
  const base = parseLocalDt(singleDt.value)
  if (!base) return null
  return fmtDisplay(addMin(base, currentLeadTimeMin.value))
})

const rangeStepCount = computed(() => {
  const s = parseLocalDt(startDt.value)
  const e = parseLocalDt(endDt.value)
  if (!s || !e || e <= s) return 0
  return Math.floor((e - s) / (5 * 60000)) + 1
})

const currentValidTimeDt = computed(() => {
  const s = parseLocalDt(startDt.value)
  if (!s) return null
  return addMin(s, rangeSliderIdx.value * 5)
})

const validTimeRange  = computed(() => fmtDisplay(currentValidTimeDt.value))
const currentInitTimeDt = computed(() => {
  const v = currentValidTimeDt.value
  return v ? addMin(v, -selectedLeadTimeMin.value) : null
})
const initTimeRange = computed(() => fmtDisplay(currentInitTimeDt.value))

// ---------------------------------------------------------------------------
// Panels
// ---------------------------------------------------------------------------
const MAX_PANELS = 4
const tooManyPanels = computed(() => selectedModels.value.length > MAX_PANELS - 1)
const activePanels = computed(() => {
  if (tooManyPanels.value || selectedModels.value.length === 0) return []
  return [
    { id: 'groundtruth', label: 'Groundtruth', isGroundtruth: true },
    ...selectedModels.value.map(m => ({ id: m, label: m, isGroundtruth: false })),
  ]
})
const gridClass = computed(() => {
  const n = activePanels.value.length
  if (n <= 2) return 'grid grid-cols-2'
  if (n === 3) return 'grid grid-cols-3'
  return 'grid grid-cols-2 grid-rows-2'
})
const hasValidConfig = computed(() => {
  if (!selectedModels.value.length) return false
  if (mode.value === 'single') return !!singleDt.value
  return !!(startDt.value && endDt.value && startDt.value < endDt.value)
})

// ---------------------------------------------------------------------------
// Panel refs + Leaflet.Sync
// ---------------------------------------------------------------------------
const panelRefs = reactive({})

function setPanelRef(id, el) {
  if (el) panelRefs[id] = el
  else delete panelRefs[id]
}

function getMaps() {
  return activePanels.value.map(p => panelRefs[p.id]?.getMap()).filter(Boolean)
}

function invalidateAllMaps() {
  getMaps().forEach(m => m.invalidateSize())
}

// Full set of maps that were pairwise synced last time rewireSyncAndAlign()
// ran — including maps whose panel has since been removed. leaflet.sync
// keeps synced maps in each map's own internal list, so a survivor still
// references a since-destroyed map unless we explicitly unsync it here.
let syncedMaps = []

async function rewireSyncAndAlign() {
  await nextTick()
  const maps = getMaps()

  syncedMaps.forEach(m1 => syncedMaps.forEach(m2 => { if (m1 !== m2) m1.unsync?.(m2) }))
  syncedMaps = maps

  if (maps.length < 2) return

  maps.forEach(m1 => maps.forEach(m2 => { if (m1 !== m2) m1.sync(m2) }))

  // Align all maps to the reference (groundtruth) view
  const ref = maps[0]
  const center = ref.getCenter()
  const zoom = ref.getZoom()
  maps.slice(1).forEach(m => m.setView(center, zoom, { animate: false }))

  setTimeout(invalidateAllMaps, 80)
}

const fssSidebarOpen = ref(false)
function toggleFss() {
  fssSidebarOpen.value = !fssSidebarOpen.value
  nextTick(() => setTimeout(invalidateAllMaps, 80))
}

// ---------------------------------------------------------------------------
// Frame URL builders
// ---------------------------------------------------------------------------
function buildFrameUrl(panel, frameIdx) {
  if (mode.value === 'single') {
    const base = parseLocalDt(singleDt.value)
    if (!base) return null
    if (panel.isGroundtruth) {
      return api.groundtruthOverlayUrl(fmtIso(addMin(base, (frameIdx + 1) * 5)), 'SRI_adj')
    }
    return api.overlayUrl(panel.id, fmtIso(base), frameIdx)
  } else {
    const s = parseLocalDt(startDt.value)
    if (!s) return null
    const validDt = addMin(s, frameIdx * 5)
    const ltMin = selectedLeadTimeMin.value
    if (panel.isGroundtruth) {
      return api.groundtruthOverlayUrl(fmtIso(validDt), 'SRI_adj')
    }
    return api.overlayUrl(panel.id, fmtIso(addMin(validDt, -ltMin)), ltMin / 5 - 1)
  }
}

// ---------------------------------------------------------------------------
// Preloading
// ---------------------------------------------------------------------------
const preloadLoaded = ref(0)
const preloadTotal  = ref(0)
const preloading    = computed(() => preloadTotal.value > 0 && preloadLoaded.value < preloadTotal.value)

let preloadSession = 0  // bumped each time we start a new preload

async function preloadAllPanels() {
  if (!hasValidConfig.value || activePanels.value.length === 0) return

  const thisSession = ++preloadSession
  const panels = activePanels.value
  const nFrames = mode.value === 'single' ? 12 : rangeStepCount.value
  if (nFrames === 0) return

  preloadTotal.value  = panels.length * nFrames
  preloadLoaded.value = 0

  const onProgress = () => {
    if (thisSession === preloadSession) preloadLoaded.value++
  }

  await Promise.all(panels.map(panel => {
    const ref = panelRefs[panel.id]
    if (!ref) return Promise.resolve()
    const urls = Array.from({ length: nFrames }, (_, i) => buildFrameUrl(panel, i))
    return ref.preloadFrames(urls, onProgress)
  }))

  if (thisSession !== preloadSession) return  // superseded — discard

  // Show the current slider position
  showAllAtFrame(mode.value === 'single' ? leadTimeIdx.value : rangeSliderIdx.value)
}

// Trigger: new panels mounted OR config changed
watch(
  activePanels,
  async () => {
    await rewireSyncAndAlign()
    await preloadAllPanels()
  },
  { flush: 'post' },
)

// Trigger: timestamp or range changed (panels stay the same, but we still
// need to re-align views — user may have pre-selected models before setting
// the time, so the activePanels watcher already fired without a valid config)
watch(
  [singleDt, startDt, endDt, selectedLeadTimeMin, mode],
  async () => {
    if (!hasValidConfig.value || activePanels.value.length === 0) return
    await rewireSyncAndAlign()
    await preloadAllPanels()
  },
)

// Reset range slider when bounds change
watch([startDt, endDt], () => { rangeSliderIdx.value = 0 })

// ---------------------------------------------------------------------------
// Frame display — slider drives showAllAtFrame
// ---------------------------------------------------------------------------
function showAllAtFrame(idx) {
  activePanels.value.forEach(p => panelRefs[p.id]?.showFrame(idx))
}

watch(leadTimeIdx, (idx) => {
  if (!preloading.value) showAllAtFrame(idx)
})
watch(rangeSliderIdx, (idx) => {
  if (!preloading.value) showAllAtFrame(idx)
})

// ---------------------------------------------------------------------------
// Play / pause
// ---------------------------------------------------------------------------
const playing = ref(false)
let playInterval = null

function togglePlay() {
  playing.value = !playing.value
  if (playing.value) {
    playInterval = setInterval(() => {
      if (mode.value === 'single') {
        leadTimeIdx.value = (leadTimeIdx.value + 1) % 12
      } else {
        const max = Math.max(0, rangeStepCount.value - 1)
        rangeSliderIdx.value = rangeSliderIdx.value >= max ? 0 : rangeSliderIdx.value + 1
      }
    }, 700)
  } else {
    clearInterval(playInterval)
    playInterval = null
  }
}

watch([mode, singleDt, startDt, endDt], () => {
  if (playing.value) togglePlay()
})

// ---------------------------------------------------------------------------
// FSS
// ---------------------------------------------------------------------------
const fssScale   = ref(5)
const fssData    = ref(null)
const fssLoading = ref(false)
let fssFetchTimeout = null

const fssAvailableForLt = computed(() => FSS_LEAD_TIMES.includes(currentLeadTimeMin.value))

function scheduleFssFetch() {
  if (!fssSidebarOpen.value || !hasValidConfig.value || !fssAvailableForLt.value) return
  clearTimeout(fssFetchTimeout)
  fssFetchTimeout = setTimeout(fetchFss, 350)
}

async function fetchFss() {
  if (!fssSidebarOpen.value || !hasValidConfig.value || !fssAvailableForLt.value) return

  const params = new URLSearchParams({
    lt: String(currentLeadTimeMin.value),
    scale: String(fssScale.value),
  })

  if (mode.value === 'single') {
    const base = parseLocalDt(singleDt.value)
    if (!base) return
    params.set('ts', fmtIso(base))
  } else {
    const initDt = currentInitTimeDt.value
    if (!initDt) return
    params.set('ts', fmtIso(initDt))
    const s = parseLocalDt(startDt.value)
    const e = parseLocalDt(endDt.value)
    if (s && e) {
      params.set('start', fmtIso(addMin(s, -selectedLeadTimeMin.value)))
      params.set('end',   fmtIso(addMin(e, -selectedLeadTimeMin.value)))
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

// Clear stale FSS when switching to a lead time that has no FSS data
watch(fssAvailableForLt, (available) => { if (!available) fssData.value = null })

watch([leadTimeIdx, rangeSliderIdx, fssScale, selectedLeadTimeMin], scheduleFssFetch)
watch(fssSidebarOpen, (open) => { if (open) fetchFss() })
watch([singleDt, startDt, endDt, mode], () => {
  fssData.value = null
  scheduleFssFetch()
})

// ---------------------------------------------------------------------------
// FSS display
// ---------------------------------------------------------------------------
function formatFss(v) { return (v === null || v === undefined) ? '—' : v.toFixed(2) }
function fssCellClass(v) {
  if (v === null || v === undefined) return 'bg-white/5 text-gray-600'
  if (v >= 0.6) return 'bg-green-900/60 text-green-300'
  if (v >= 0.3) return 'bg-yellow-900/60 text-yellow-300'
  return 'bg-red-900/60 text-red-400'
}

onUnmounted(() => {
  clearInterval(playInterval)
  clearTimeout(fssFetchTimeout)
})
</script>

<style scoped>
:deep(.dp-dark-input) {
  height: 42px !important;
  border-radius: 0.5rem !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  background: rgba(255,255,255,0.05) !important;
  color: white !important;
  font-size: 0.875rem !important;
  padding: 0 0.75rem !important;
  width: 140px;
}
:deep(.dp-dark-input:focus) {
  border-color: #60a5fa !important;
  box-shadow: 0 0 0 1px #60a5fa !important;
}
:deep(.dp__input_wrap)             { width: 140px; }
:deep(.dp-time-input)              { width: 110px; }
:deep(.dp__input_wrap:has(.dp-time-input)) { width: 110px; }
:deep(.leaflet-container)          { background: #1a1a2e; }
:deep(.leaflet-image-layer)        { image-rendering: pixelated; }
</style>
