<!--
  RealTimeView.vue — Real-time prediction monitoring (main page).

  This is the heart of the app. It shows:
  - An interactive Leaflet map with radar overlay (past + future)
  - Animation controls: play/pause + timeline slider (-60 to +60 min)
  - Model selector to switch predictions
  - Start/Pause Real Time button for backend-driven prediction cycle
  - Precipitation colorbar legend
  - Status polling from the backend every 3s

  TIMELINE STRUCTURE (25 frames total):
  - Frames 0-12:  Past groundtruth (SRI data) — -60 to 0 minutes
  - Frames 13-24: Future predictions (model)  — +5 to +60 minutes

  The frame index maps to minutes via: (index - 12) * 5
    index 0  → -60 min (past SRI)
    index 12 →   0 min (current SRI)
    index 13 →  +5 min (prediction lead_time=0)
    index 24 → +60 min (prediction lead_time=11)

  BACKEND-DRIVEN CYCLE:
  The backend runs the prediction loop (HPC or mock). The frontend simply
  polls GET /api/realtime/status every 3s to stay in sync. This means:
  - The cycle survives browser close (backend keeps running)
  - Multiple browser tabs see the same state
  - Any tab can start or stop the service
-->
<template>
  <div class="h-[calc(100dvh-3rem)] sm:h-[calc(100vh-3.5rem)] flex">

    <!-- ================================================================ -->
    <!-- LEFT: Map area (takes all available width)                       -->
    <!-- ================================================================ -->
    <div class="flex-1 flex flex-col relative">
      <RadarMap
        ref="radarMap"
        class="flex-1"
        :overlay-opacity="overlayOpacity"
        @mapclick="onMapClick"
      />

      <!-- Notification toast — floating at top center of map -->
      <Transition name="toast">
        <div
          v-if="notification"
          class="absolute top-4 left-1/2 -translate-x-1/2 z-[1002]
                 bg-emerald-600 text-white px-5 py-3 rounded-xl shadow-lg
                 flex items-center gap-3 text-sm font-medium"
        >
          <!-- Radar icon -->
          <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path d="M5.636 18.364a9 9 0 010-12.728m12.728 0a9 9 0 010 12.728M9.172 14.828a4 4 0 010-5.656m5.656 0a4 4 0 010 5.656M12 12h.01" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
          {{ notification }}
        </div>
      </Transition>

      <!-- Colorbars — floating on the map, bottom right (above the timeline bar) -->
      <!-- Ensemble mode: single probability colorbar. Normal: SRI + optional IR. -->
      <div
        v-if="settings.showColorbars"
        class="colorbar-stack absolute right-[10px] z-[1001]
               flex flex-col justify-end gap-1.5 items-end
               overflow-y-auto"
      >
        <template v-if="ensembleActive">
          <ColorBar :legend="probLegend" product-name="P(%)" />
        </template>
        <template v-else>
          <ColorBar
            v-if="irEnabled"
            :legend="configStore.radarProducts['IR_108']"
            product-name="IR"
          />
          <ColorBar />
        </template>
      </div>

      <!-- ============================================================ -->
      <!-- BOTTOM BAR: Timeline controls (floating over the map)        -->
      <!-- ============================================================ -->
      <!-- Sidebar toggle button -->
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
        v-if="!latestSRI && !sidebarOpen"
        class="absolute top-14 right-3 z-[1000] flex items-center gap-2 px-3 h-9 rounded-full
               bg-white/90 shadow-lg border border-gray-200 text-gray-600 text-sm"
      >
        <svg class="animate-spin h-4 w-4 text-blue-500 flex-shrink-0" viewBox="0 0 24 24" fill="none">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <span class="text-xs font-medium">Loading…</span>
      </div>

      <div class="absolute bottom-0 left-0 right-0 z-[1000]
                  bg-gradient-to-t from-black/80 via-black/60 to-transparent
                  px-3 sm:px-6 pt-6 sm:pt-10
                  pb-[calc(1rem+env(safe-area-inset-bottom))] sm:pb-4">

        <!-- Current time display -->
        <div class="flex items-center justify-between text-white mb-2">
          <div class="text-sm font-medium hidden sm:block">
            <span class="text-gray-300">Model:</span>
            <span class="ml-1 font-bold">{{ ensembleActive ? 'Probabilistic' : (selectedModel || 'None') }}</span>
          </div>
          <div class="text-center">
            <span class="text-sm sm:text-2xl font-bold tabular-nums">
              {{ frameMinutesDisplay }}
            </span>
            <!-- Inline model label — mobile only; desktop shows it on the left -->
            <span class="sm:hidden ml-1.5 text-[10px] text-gray-300">
              <span class="text-gray-400">Model:</span>
              <span class="ml-0.5 font-semibold text-white">{{ ensembleActive ? 'Probabilistic' : (selectedModel || '—') }}</span>
            </span>
            <span
              class="ml-2 text-[10px] sm:text-xs font-medium px-1.5 sm:px-2 py-0.5 rounded-full"
              :class="frameIndex <= 12
                ? 'bg-emerald-500/30 text-emerald-300'
                : ensembleActive
                  ? 'bg-purple-500/30 text-purple-300'
                  : 'bg-blue-500/30 text-blue-300'"
            >
              {{ frameIndex <= 12 ? 'Observed' : ensembleActive ? 'Probabilistic Forecast' : 'Forecast' }}
            </span>
          </div>
          <div class="text-sm text-gray-300 hidden sm:block">
            {{ latestTimestampDisplay || 'No data' }}
          </div>
        </div>

        <!-- Timeline slider — items-start so the slider thumb (centered inside its
             own h-10 wrapper) lines up vertically with the play/speed buttons. -->
        <div class="flex items-start gap-2 sm:gap-4">
          <!-- Play/Pause button -->
          <button
            @click="togglePlay"
            class="w-10 h-10 flex items-center justify-center rounded-full
                   bg-white/20 hover:bg-white/30 text-white transition-colors
                   backdrop-blur-sm flex-shrink-0"
            :title="playing ? 'Pause' : 'Play'"
          >
            <!-- Pause icon -->
            <svg v-if="playing" class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
            <!-- Play icon -->
            <svg v-else class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          </button>

          <!-- Timeline slider (0-24, representing -60 to +60 min) -->
          <div class="flex-1 min-w-0">
            <div class="h-10 flex items-center">
              <input
                type="range"
                v-model.number="frameIndex"
                min="0"
                max="24"
                step="1"
                class="w-full h-2 appearance-none cursor-pointer rounded-full
                       bg-white/20 accent-blue-400
                       [&::-webkit-slider-thumb]:appearance-none
                       [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4
                       [&::-webkit-slider-thumb]:rounded-full
                       [&::-webkit-slider-thumb]:bg-blue-400
                       [&::-webkit-slider-thumb]:shadow-lg
                       [&::-webkit-slider-thumb]:shadow-blue-400/50"
              />
            </div>
            <!-- Tick marks — show every 15 minutes for readability -->
            <div class="flex justify-between px-0.5">
              <span
                v-for="i in 25"
                :key="i"
                class="text-[10px] tabular-nums w-0 text-center"
                :class="tickClass(i - 1)"
              >
                {{ tickLabel(i - 1) }}
              </span>
            </div>
          </div>

          <!-- Speed control -->
          <div class="h-10 flex items-center flex-shrink-0">
            <button
              @click="cycleSpeed"
              class="px-3 py-1.5 rounded-full bg-white/20 hover:bg-white/30
                     text-white text-xs font-medium transition-colors backdrop-blur-sm"
              title="Animation speed"
            >
              {{ speedLabel }}
            </button>
          </div>

        </div>
      </div>
    </div>

    <!-- ================================================================ -->
    <!-- RIGHT: Sidebar panel (drawer on mobile, fixed on desktop)        -->
    <!-- ================================================================ -->
    <!-- Backdrop -->
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

      <!-- Start / Pause Real Time Button -->
      <div class="p-4 border-b border-gray-700">
        <!-- Server mode: always-on, locked indicator -->
        <div
          v-if="configStore.isServer"
          class="w-full py-3 px-4 rounded-lg font-semibold text-sm
                 flex items-center justify-center gap-2
                 bg-green-900/40 text-green-400 border border-green-700 cursor-not-allowed select-none"
          title="Real-time predictions are always running in server mode"
        >
          <!-- Pulse dot -->
          <span class="relative flex h-2.5 w-2.5">
            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
            <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500"></span>
          </span>
          NWC Running
          <!-- Lock icon -->
          <svg class="w-3.5 h-3.5 ml-auto opacity-50" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
            <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
          </svg>
        </div>

        <!-- Normal mode: interactive start/stop button -->
        <button
          v-else
          @click="toggleRealTime"
          class="w-full py-3 px-4 rounded-lg font-semibold text-sm transition-all
                 flex items-center justify-center gap-2"
          :class="realTimeActive
            ? 'bg-red-900/40 text-red-400 border border-red-700 hover:bg-red-900/60'
            : 'bg-blue-600 text-white hover:bg-blue-700 shadow-sm'"
        >
          <svg v-if="realTimeActive" class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <rect x="6" y="4" width="4" height="16" rx="1" />
            <rect x="14" y="4" width="4" height="16" rx="1" />
          </svg>
          <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path d="M5 3l14 9-14 9V3z" />
          </svg>
          {{ realTimeActive ? 'Pause Real Time' : 'Start Real Time' }}
        </button>
        <p v-if="realTimeActive && !configStore.isServer" class="text-[10px] text-center text-gray-400 mt-1.5">
          Polling every 3s
        </p>
      </div>

      <!-- Model Selector -->
      <div class="p-4 border-b border-gray-700">
        <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Active Model
        </h3>
        <select
          v-model="selectedModel"
          class="w-full rounded-lg border border-gray-600 bg-gray-800 text-gray-200 px-3 py-2.5 text-sm
                 font-medium focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          <option value="" disabled>Select model...</option>
          <option v-for="model in models" :key="model" :value="model">
            {{ model }}
          </option>
        </select>
      </div>

      <!-- Latest Data -->
      <div class="p-4 border-b border-gray-700">
        <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Latest Data
        </h3>
        <div v-if="latestSRI" class="flex items-center gap-2">
          <div class="w-2 h-2 rounded-full" :class="latestSRI.latest_file ? 'bg-green-400' : 'bg-red-400'" />
          <span class="text-sm text-gray-200 font-medium">
            {{ latestSRI.latest_file ? formatSriFilename(latestSRI.latest_file) : 'No data' }}
          </span>
        </div>
        <div v-else class="flex items-center gap-2">
          <div class="w-2 h-2 rounded-full bg-gray-600 animate-pulse" />
          <span class="text-sm text-gray-400">Loading...</span>
        </div>
      </div>

      <!-- Model Status List -->
      <div class="p-4 border-b border-gray-700 flex-1">
        <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Model Status
        </h3>

        <div class="space-y-1">
          <div
            v-for="model in models"
            :key="model"
            @click="selectModel(model)"
            class="flex items-center justify-between py-2.5 px-3 rounded-lg cursor-pointer
                   transition-colors"
            :class="selectedModel === model && !ensembleActive
              ? 'bg-blue-900/50 border border-blue-500/50'
              : 'hover:bg-gray-800'"
          >
            <span
              class="text-sm"
              :class="selectedModel === model && !ensembleActive ? 'font-semibold text-blue-300' : 'text-gray-300'"
            >
              {{ model }}
            </span>
            <span
              class="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full"
              :class="statusClass(model)"
            >
              <span class="w-1.5 h-1.5 rounded-full" :class="statusDotClass(model)" />
              {{ statusText(model) }}
            </span>
          </div>
        </div>
      </div>

      <!-- Probabilistic Ensemble -->
      <div class="p-4 border-b border-gray-700">
        <div class="flex items-center justify-between mb-3">
          <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Probabilistic Ensemble
          </h3>
          <button
            @click="toggleEnsemble"
            class="text-xs px-2.5 py-1 rounded-full font-medium transition-colors"
            :class="ensembleActive
              ? 'bg-purple-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'"
          >
            {{ ensembleActive ? 'Active' : 'Show' }}
          </button>
        </div>

        <!-- Model checkboxes -->
        <div class="grid grid-cols-2 gap-x-2 gap-y-1.5 mb-3">
          <label
            v-for="model in models.filter(m => m !== 'Test')"
            :key="model"
            class="flex items-center gap-1.5 cursor-pointer group"
          >
            <input
              type="checkbox"
              :value="model"
              v-model="ensembleModels"
              class="w-3.5 h-3.5 rounded accent-purple-500 cursor-pointer"
            />
            <span class="text-xs text-gray-300 group-hover:text-gray-100 truncate">{{ model }}</span>
          </label>
        </div>

        <!-- Threshold slider — non-linear discrete steps (0.5–50 mm/h).
             Fine resolution at low values where 1→2 mm/h matters;
             coarser at high values where 30→35 is nearly irrelevant. -->
        <div class="flex items-center gap-2">
          <span class="text-xs text-gray-400 flex-shrink-0">Threshold</span>
          <input
            type="range"
            :value="thresholdIndex(ensembleThreshold)"
            @input="ensembleThreshold = THRESHOLD_STEPS[parseInt($event.target.value)]"
            @change="onThresholdCommit"
            min="0"
            :max="THRESHOLD_STEPS.length - 1"
            step="1"
            class="flex-1 accent-purple-500 cursor-pointer"
          />
          <span class="text-xs font-semibold text-purple-300 tabular-nums w-16 text-right">
            {{ fmtThreshold(ensembleThreshold) }} mm/h
          </span>
        </div>

        <div v-if="ensembleActive" class="flex items-center justify-between mt-2 gap-2">
          <p class="text-[10px] text-purple-400">
            Using {{ ensembleModels.length }} / {{ models.filter(m => m !== 'Test').length }} models
          </p>
          <button
            @click="ensembleContours = !ensembleContours"
            class="text-[10px] px-2 py-0.5 rounded-full font-medium transition-colors flex-shrink-0"
            :class="ensembleContours
              ? 'bg-purple-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'"
            title="Toggle dark probability contour lines"
          >
            {{ ensembleContours ? 'Hide contours' : 'Show contours' }}
          </button>
        </div>
      </div>

      <!-- Radar SRI Overlay -->
      <div class="p-4 border-b border-gray-700">
        <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Radar (SRI)
        </h3>
        <div class="flex items-center gap-2">
          <span class="text-xs text-gray-400 w-12 flex-shrink-0">Opacity</span>
          <input
            type="range"
            v-model.number="overlayOpacity"
            min="0.1"
            max="1"
            step="0.05"
            class="flex-1 h-1.5 accent-blue-400 cursor-pointer"
          />
          <span class="text-xs text-gray-400 w-8 text-right tabular-nums">
            {{ Math.round(overlayOpacity * 100) }}%
          </span>
        </div>
      </div>

      <!-- IR Satellite Overlay -->
      <div class="p-4 border-b border-gray-700">
        <div class="flex items-center justify-between mb-2">
          <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Satellite IR
          </h3>
          <label class="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              v-model="irEnabled"
              class="w-4 h-4 rounded accent-blue-500"
            />
            <span class="text-xs text-gray-300">Show</span>
          </label>
        </div>
        <p class="text-[10px] text-gray-400 mb-2">IR 10.8 µm cloud cover overlay</p>
        <div v-if="irEnabled" class="flex items-center gap-2">
          <span class="text-xs text-gray-400 w-12 flex-shrink-0">Opacity</span>
          <input
            type="range"
            v-model.number="irOpacity"
            min="0"
            max="1"
            step="0.05"
            class="flex-1 h-1.5 accent-blue-400 cursor-pointer"
          />
          <span class="text-xs text-gray-400 w-8 text-right tabular-nums">
            {{ Math.round(irOpacity * 100) }}%
          </span>
        </div>
      </div>

      <!-- Motion field layer (AMV / LK) -->
      <div class="p-4 space-y-2">
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
          <div v-if="motionMode !== 'none' && activeMotionTs" class="text-gray-500 text-[10px]">
            {{ motionMode.toUpperCase() }}: {{ activeMotionTs.replace('T', ' ') }} UTC
            <span v-if="motionMode === 'amv'" class="text-gray-600 ml-1">(20 min cadence)</span>
          </div>
          <div v-if="motionMode !== 'none' && !activeMotionTs && !motionLoading" class="text-amber-400 text-[10px]">
            No {{ motionMode.toUpperCase() }} data for current time
          </div>
        </div>
      </div>

      <!-- Info -->
      <div class="p-4">
        <p v-if="lastRefresh" class="text-[10px] text-gray-400">
          Status updated {{ lastRefresh }}
        </p>
      </div>
    </div><!-- /inner wrapper -->
    </div><!-- /sidebar outer -->
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import api from '../api.js'
import { useConfigStore } from '../stores/config.js'
import { useSettingsStore } from '../stores/settings.js'
import RadarMap from '../components/RadarMap.vue'
import ColorBar from '../components/ColorBar.vue'
import { useMotionLayer } from '../composables/useMotionLayer.js'

const configStore = useConfigStore()
const settings = useSettingsStore()
const models = computed(() => configStore.models)

// ---- Timezone helpers (display only — all data stays UTC) ----
const displayTz = computed(() =>
  settings.timeZone === 'utc' ? 'UTC' : 'Europe/Rome'
)

function formatTimeInTz(date) {
  return date.toLocaleTimeString('it-IT', {
    timeZone: displayTz.value, hour: '2-digit', minute: '2-digit', hour12: false
  })
}

function formatDateTimeInTz(date) {
  return date.toLocaleString('it-IT', {
    timeZone: displayTz.value, day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit', hour12: false
  })
}

// Legacy aliases kept for existing call sites
const formatTimeInRome = formatTimeInTz
const formatDateTimeInRome = formatDateTimeInTz

// ---- State ----
const radarMap = ref(null)
const sidebarOpen = ref(false)
const selectedModel = ref('')

// Cached prediction URLs (frames 13-24) from the last successful full load.
// Used to keep old predictions visible on the map while a new job is computing.
const lastPredUrls = ref([])
const frameIndex = ref(12)  // Start at index 12 = "0 min" (current time)

// Current radar timestamp for the motion layer composable ("YYYY-MM-DDTHH:MM")
const _motionCurrentTs = computed(() => {
  if (!latestTimestamp.value) return ''
  const baseDt = new Date(latestTimestamp.value)
  const frameDt = new Date(baseDt.getTime() + (frameIndex.value - 12) * 5 * 60000)
  const p = n => String(n).padStart(2, '0')
  return `${frameDt.getUTCFullYear()}-${p(frameDt.getUTCMonth()+1)}-${p(frameDt.getUTCDate())}T${p(frameDt.getUTCHours())}:${p(frameDt.getUTCMinutes())}`
})
const { motionMode, motionLoading, activeMotionTs, fetchTimestamps: fetchMotionTimestamps } =
  useMotionLayer(radarMap, _motionCurrentTs)

const playing = ref(false)
const speed = ref(1)
const latestSRI = ref(null)
const overlayOpacity = ref(1.0)
const lastRefresh = ref('')

// IR satellite overlay
const irEnabled = ref(false)
const irOpacity = ref(0.75)

// ---- Probabilistic ensemble ----
const ensembleActive = ref(false)
const ensembleModels = ref([])     // populated in onMounted once model list is loaded
const ensembleContours = ref(false) // dark contour overlay (off by default)

// Non-linear threshold steps: fine resolution at low rain rates where 1→2 mm/h
// matters a lot, coarser at high rates where 30→35 is nearly irrelevant.
const THRESHOLD_STEPS = [0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 15, 20, 25, 30, 40, 50]
const ensembleThreshold = ref(2)   // default: 2 mm/h (must be a value in THRESHOLD_STEPS)

function fmtThreshold(v) {
  return v % 1 === 0 ? String(v) : v.toFixed(1)
}
function thresholdIndex(v) {
  const i = THRESHOLD_STEPS.indexOf(v)
  return i !== -1 ? i : THRESHOLD_STEPS.reduce((best, s, i) =>
    Math.abs(s - v) < Math.abs(THRESHOLD_STEPS[best] - v) ? i : best, 0)
}

// Probability colorbar legend: Oranges palette (matplotlib), ticks in percent.
// Stays legible on dark, OSM, and satellite basemaps alike.
const probLegend = computed(() => ({
  unit: `P > ${fmtThreshold(ensembleThreshold.value)} mm/h`,
  thresholds: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
  colors: [
    'rgb(255,245,235)', 'rgb(254,230,206)', 'rgb(253,208,162)',
    'rgb(253,174,107)', 'rgb(253,141,60)',  'rgb(241,105,19)',
    'rgb(217,72,1)',    'rgb(166,54,3)',    'rgb(127,39,4)',
    'rgb(102,30,3)',    'rgb(80,22,2)',
  ],
}))

function toggleEnsemble() {
  ensembleActive.value = !ensembleActive.value
}

/** Select a single model — also deactivates ensemble mode. */
function selectModel(model) {
  selectedModel.value = model
  ensembleActive.value = false
}

// Real-time state (driven by backend)
const realTimeActive = ref(false)
const backendState = ref(null)   // Full state from GET /api/realtime/status
const notification = ref('')     // Toast message string (empty = hidden)

let playInterval = null
let statusPollInterval = null
let sriPollInterval = null      // Periodic SRI polling (runs even when RT is off)
let notificationTimer = null
let lastShownNotification = ''  // Track which notification we already displayed

// ---- Constants ----
const TOTAL_FRAMES = 25        // 13 past (including current) + 12 future
const CURRENT_INDEX = 12       // Index of "0 min" in the frame array
const POLL_INTERVAL_MS = 3000  // How often we poll the backend
const SRI_POLL_INTERVAL_MS = 60000  // How often we poll for new SRI data (1 min)

// ---- Speed control ----
const speeds = [0.5, 1, 2, 4]
const speedLabel = computed(() => `${speed.value}×`)

function cycleSpeed() {
  const idx = speeds.indexOf(speed.value)
  speed.value = speeds[(idx + 1) % speeds.length]
  if (playing.value) {
    stopPlay()
    startPlay()
  }
}

// ---- Frame display helpers ----

/**
 * Convert frame index to minute offset.
 * Index 0 → -60, Index 12 → 0, Index 24 → +60
 */
function frameToMinutes(index) {
  return (index - CURRENT_INDEX) * 5
}

// ---- Click-to-inspect popup ----
// Build the timestamp string the backend expects: "YYYY-MM-DDTHH:MM" (no Z).
function isoNoTz(dt) {
  const p = n => String(n).padStart(2, '0')
  return `${dt.getUTCFullYear()}-${p(dt.getUTCMonth()+1)}-${p(dt.getUTCDate())}T${p(dt.getUTCHours())}:${p(dt.getUTCMinutes())}:00`
}

function fmtMm(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return 'N/A'
  return Math.abs(v) < 10 ? v.toFixed(2) : v.toFixed(1)
}

async function onMapClick(latlng) {
  if (!radarMap.value) return

  if (!latestTimestamp.value) return

  const idx = frameIndex.value
  const isFuture = idx > CURRENT_INDEX
  const baseDt = new Date(latestTimestamp.value)
  const frameDt = new Date(baseDt.getTime() + frameToMinutes(idx) * 60000)

  const headerLabel = `${formatDateTimeInRome(frameDt)}`
  radarMap.value.showPopup(
    latlng,
    `<div class="pi-header">${headerLabel}</div>
     <div class="pi-row"><span class="pi-label">Loading…</span></div>`,
  )

  try {
    let data
    let mode  // 'ensemble' | 'future' | 'past'
    if (isFuture && ensembleActive.value && ensembleModels.value.length > 0) {
      mode = 'ensemble'
      const leadTime = idx - CURRENT_INDEX - 1
      data = await api.samplePixel({
        lat: latlng.lat,
        lon: latlng.lng,
        timestamp: isoNoTz(baseDt),
        products: [],
        models: ensembleModels.value,
        leadTime,
      })
    } else if (isFuture) {
      mode = 'future'
      if (!selectedModel.value) {
        radarMap.value.showPopup(
          latlng,
          `<div class="pi-header">${headerLabel}</div>
           <div class="pi-row"><span class="pi-label">No model selected</span></div>`,
        )
        return
      }
      const leadTime = idx - CURRENT_INDEX - 1
      data = await api.samplePixel({
        lat: latlng.lat,
        lon: latlng.lng,
        timestamp: isoNoTz(baseDt),
        products: [],
        model: selectedModel.value,
        leadTime,
      })
    } else {
      mode = 'past'
      data = await api.samplePixel({
        lat: latlng.lat,
        lon: latlng.lng,
        timestamp: isoNoTz(frameDt),
        products: ['SRI_adj'],
      })
    }

    let body
    if (!data.in_bounds) {
      body = `<div class="pi-row"><span class="pi-label">Outside radar grid</span></div>`
    } else {
      const rowPixel = `
        <div class="pi-row" style="margin-bottom:4px;">
          <span class="pi-label">pixel</span>
          <span class="pi-value">x ${data.x}, y ${data.y}</span>
        </div>`

      let rowValue
      if (mode === 'ensemble') {
        const thr = ensembleThreshold.value
        const perModel = data.models || {}
        const names = Object.keys(perModel)
        const valid = names.filter(n => perModel[n] !== null && Number.isFinite(perModel[n]))
        const exceed = valid.filter(n => perModel[n] > thr)
        const probPct = valid.length > 0
          ? Math.round((exceed.length / valid.length) * 100)
          : null

        // Header summary + per-model breakdown.
        const summary = `
          <div class="pi-row" style="margin-bottom:4px;">
            <span class="pi-label">P &gt; ${fmtThreshold(thr)} mm/h</span>
            <span class="pi-value" style="color:#fdba74;">
              ${probPct === null ? 'N/A' : probPct + '%'}
              ${valid.length ? `<span style="color:rgba(255,255,255,0.5);font-weight:400;">&nbsp;(${exceed.length}/${valid.length})</span>` : ''}
            </span>
          </div>`
        const perModelRows = names.map(n => {
          const v = perModel[n]
          const exceeds = v != null && Number.isFinite(v) && v > thr
          const dot = exceeds
            ? '<span style="color:#fdba74;">●</span>'
            : v == null ? '<span style="color:rgba(255,255,255,0.3);">○</span>'
                        : '<span style="color:rgba(255,255,255,0.3);">○</span>'
          return `
            <div class="pi-row">
              <span class="pi-label">${dot} ${n}</span>
              <span class="pi-value">${fmtMm(v)}${v != null ? ' mm/h' : ''}</span>
            </div>`
        }).join('')
        rowValue = summary + perModelRows
      } else if (mode === 'future') {
        const v = data.values?.[`__model__${selectedModel.value}`]
        rowValue = `
          <div class="pi-row">
            <span class="pi-label">${selectedModel.value}</span>
            <span class="pi-value">${fmtMm(v)}${v != null ? ' mm/h' : ''}</span>
          </div>`
      } else {
        const v = data.values?.SRI_adj
        rowValue = `
          <div class="pi-row">
            <span class="pi-label">SRI</span>
            <span class="pi-value">${fmtMm(v)}${v != null ? ' mm/h' : ''}</span>
          </div>`
      }
      body = rowPixel + rowValue
    }

    radarMap.value.showPopup(
      latlng,
      `<div class="pi-header">${headerLabel}</div>${body}`,
    )
  } catch (e) {
    radarMap.value.showPopup(
      latlng,
      `<div class="pi-header">${headerLabel}</div>
       <div class="pi-row"><span class="pi-label">Error: ${e.message || e}</span></div>`,
    )
  }
}

/**
 * Display the actual time of the current frame + offset in parentheses.
 * Example: "14:30 (-5 min)" or "14:35 (0 min)" or "14:40 (+5 min)"
 */
const frameMinutesDisplay = computed(() => {
  const mins = frameToMinutes(frameIndex.value)
  const offsetStr = mins === 0 ? '0 min' : `${mins > 0 ? '+' : ''}${mins} min`

  if (!latestTimestamp.value) return `(${offsetStr})`

  // Compute the actual time for this frame
  const baseDt = new Date(latestTimestamp.value)
  const frameDt = new Date(baseDt.getTime() + mins * 60000)

  return `${formatTimeInRome(frameDt)} (${offsetStr})`
})

/**
 * Tick label for the slider. Show a label every 15 minutes for readability.
 */
function tickLabel(index) {
  const mins = frameToMinutes(index)
  // Show labels at -60, -45, -30, -15, 0, +15, +30, +45, +60
  if (mins % 15 === 0) return mins === 0 ? '0' : `${mins > 0 ? '+' : ''}${mins}`
  return ''
}

function tickClass(index) {
  if (index === frameIndex.value) return 'text-blue-400 font-bold'
  const mins = frameToMinutes(index)
  if (mins % 15 !== 0) return 'invisible'
  if (mins === 0) return 'text-white/80 font-semibold'
  return 'text-gray-400'
}

// ---- Computed: latest timestamp from SRI filename ----
const latestTimestamp = computed(() => {
  if (!latestSRI.value?.latest_file) return null
  const filename = latestSRI.value.latest_file.replace('.hdf', '')
  const parts = filename.split('-')
  if (parts.length !== 5) return null
  const [day, month, year, hour, minute] = parts
  return `${year}-${month}-${day}T${hour}:${minute}:00Z`
})

const latestTimestampDisplay = computed(() => {
  if (!latestSRI.value?.latest_file) return null
  return formatSriFilename(latestSRI.value.latest_file)
})

// ---- Preload all 25 frames when model or timestamp changes ----

// keepPredictions: when true, reuse cached prediction URLs (frames 13-24) instead of
// generating new ones. Used when new SRI arrives during an active job so the old
// prediction stays visible on the map until the new one is ready.
async function preloadAllFrames({ keepPredictions = false } = {}) {
  if (!radarMap.value) return
  // Need at least a timestamp to show groundtruth, OR the Test model
  const isTest = selectedModel.value?.toUpperCase() === 'TEST'
  if (!isTest && !latestTimestamp.value) return

  const baseDt = latestTimestamp.value ? new Date(latestTimestamp.value) : new Date()
  const hasModel = !!selectedModel.value

  // Build 25 URLs: 13 past/current (groundtruth) + 12 future (predictions)
  //
  // NO MODEL SELECTED: Only groundtruth frames (0-12) are loaded from SRI.
  //   Future frames (13-24) return null → RadarMap shows blank (no overlay).
  //
  // TEST MODEL: All 25 frames from the static predictions.npy file.
  //
  // OTHER MODELS: Past from SRI files, future from per-timestamp .npy files.
  const urls = Array.from({ length: TOTAL_FRAMES }, (_, i) => {
    const minuteOffset = frameToMinutes(i)

    if (minuteOffset <= 0) {
      if (isTest) {
        const gtIndex = Math.min(i, 11)
        return api.overlayUrl('Test', latestTimestamp.value, gtIndex, 'groundtruth')
      }
      // Groundtruth from SRI files (works with or without a model selected)
      const pastDt = new Date(baseDt.getTime() + minuteOffset * 60000)
      const ts = formatIsoTimestamp(pastDt)
      return api.groundtruthOverlayUrl(ts)
    } else {
      // Prediction frames: keep showing old data while the new job is running
      if (keepPredictions && lastPredUrls.value.length > 0) {
        return lastPredUrls.value[i - (CURRENT_INDEX + 1)] || ''
      }
      const leadTimeIndex = Math.round(minuteOffset / 5) - 1
      // Ensemble mode: use probabilistic overlay
      if (ensembleActive.value && ensembleModels.value.length > 0) {
        return api.ensembleOverlayUrl(
          latestTimestamp.value, leadTimeIndex,
          ensembleThreshold.value, ensembleModels.value,
          ensembleContours.value,
        )
      }
      // Single model mode
      if (!hasModel) return null
      return api.overlayUrl(selectedModel.value, latestTimestamp.value, leadTimeIndex)
    }
  })

  // Filter out nulls for RadarMap — pass empty string so frame slots still line up
  const safeUrls = urls.map(u => u || '')

  // Save prediction URLs (frames 13-24) whenever we do a full load,
  // so they can be restored on the next keepPredictions call.
  if (!keepPredictions) {
    lastPredUrls.value = safeUrls.slice(CURRENT_INDEX + 1)
  }

  await radarMap.value.preloadFrames(safeUrls)
  radarMap.value.showFrame(frameIndex.value)

  // Load IR satellite overlay if enabled.
  // Past frames: actual IR timestamp. Future frames: clamp to t=0 (current IR).
  if (irEnabled.value) {
    const irUrls = Array.from({ length: TOTAL_FRAMES }, (_, i) => {
      const minuteOffset = frameToMinutes(i)
      // For future frames, show the current IR image (no satellite forecast)
      const effectiveOffset = Math.min(minuteOffset, 0)
      const frameDt = new Date(baseDt.getTime() + effectiveOffset * 60000)
      return api.groundtruthOverlayUrl(formatIsoTimestamp(frameDt), 'IR_108')
    })
    await radarMap.value.loadProductFrames('IR_108', irUrls, irOpacity.value)
    radarMap.value.showAllAtFrame(frameIndex.value)
    // Ensure SRI (frameLayers) renders above IR
    radarMap.value.bringFramesToFront()
  }
}

/**
 * Format a Date object as ISO timestamp string (YYYY-MM-DDTHH:MM) in UTC.
 * Used for building API URLs that match UTC filenames on disk.
 */
function formatIsoTimestamp(dt) {
  const year = dt.getUTCFullYear()
  const month = String(dt.getUTCMonth() + 1).padStart(2, '0')
  const day = String(dt.getUTCDate()).padStart(2, '0')
  const hours = String(dt.getUTCHours()).padStart(2, '0')
  const minutes = String(dt.getUTCMinutes()).padStart(2, '0')
  return `${year}-${month}-${day}T${hours}:${minutes}`
}

// When model changes → preload all frames for that model
watch(selectedModel, () => { preloadAllFrames() })

// When latest timestamp changes (new SRI data) → update groundtruth frames.
// If real-time is active, keep existing prediction frames visible while the
// new job computes; they'll be replaced when the model transitions to 'ready'.
watch(latestTimestamp, () => {
  const keepPredictions = realTimeActive.value && !!selectedModel.value
  preloadAllFrames({ keepPredictions })
})

// When frame index changes (slider drag) → instantly show that frame
watch(frameIndex, (newIdx) => {
  if (radarMap.value) {
    radarMap.value.showFrame(newIdx)
    if (irEnabled.value) radarMap.value.showAllAtFrame(newIdx)
  }
})

// When opacity slider changes → update the currently visible frame
watch(overlayOpacity, (newOpacity) => {
  if (radarMap.value) radarMap.value.setOverlayOpacity(newOpacity)
})

// IR overlay: toggle on/off
// preloadAllFrames already calls bringFramesToFront after loading IR
watch(irEnabled, async (enabled) => {
  if (enabled) {
    await preloadAllFrames()
  } else {
    radarMap.value?.removeProduct('IR_108')
  }
})

// Ensemble: any change to active state, models, or threshold → reload frames
watch(ensembleActive, () => { preloadAllFrames() })
watch(ensembleModels, () => { preloadAllFrames() }, { deep: true })
watch(ensembleContours, () => { if (ensembleActive.value) preloadAllFrames() })

// Threshold slider fires reload only on release (`change` event) — see the
// onThresholdCommit handler below. The intermediate `input` events still
// update the displayed value live, so the label tracks the slider, but no
// network request is sent until the user lets go.
function onThresholdCommit() {
  if (ensembleActive.value) preloadAllFrames()
}

// IR opacity: update the currently visible IR frame
watch(irOpacity, (opacity) => {
  if (radarMap.value && irEnabled.value) {
    radarMap.value.setProductOpacity('IR_108', opacity)
    radarMap.value.showAllAtFrame(frameIndex.value)
  }
})

// ---- Animation controls ----
function togglePlay() {
  if (playing.value) stopPlay()
  else startPlay()
}

function startPlay() {
  playing.value = true
  const intervalMs = 800 / speed.value
  playInterval = setInterval(() => {
    frameIndex.value = (frameIndex.value + 1) % TOTAL_FRAMES
  }, intervalMs)
}

function stopPlay() {
  playing.value = false
  if (playInterval) {
    clearInterval(playInterval)
    playInterval = null
  }
}

// ---- Real-Time: Backend-driven ----

/**
 * Toggle real-time prediction on/off.
 * Start: POST to backend, then start polling.
 * Stop: POST to backend, then stop polling and reset UI.
 */
async function toggleRealTime() {
  if (realTimeActive.value) {
    // --- Stop ---
    try {
      await api.stopRealTime()
    } catch (e) {
      console.error('Failed to stop real-time:', e)
    }
    stopStatusPolling()
    realTimeActive.value = false
    backendState.value = null
    notification.value = ''
    lastShownNotification = ''
  } else {
    // --- Start ---
    try {
      const result = await api.startRealTime()

      if (!result.ok && result.reason === 'already_running') {
        // Service was already running (e.g. started from another tab).
        // That's fine — just start polling.
        console.log('Real-time already running, joining existing session')
      }
    } catch (e) {
      console.error('Failed to start real-time:', e)
      return
    }

    realTimeActive.value = true

    // Quick check: which models already have predictions for the latest timestamp?
    // This gives instant "Ready" feedback instead of waiting for the first poll cycle.
    if (latestTimestamp.value) {
      const initialModels = {}
      const checks = await Promise.allSettled(
        models.value
          .filter(m => m.toUpperCase() !== 'TEST')
          .map(async (model) => {
            const check = await api.checkSinglePrediction(model, latestTimestamp.value)
            return { model, exists: check.exists }
          })
      )
      for (const result of checks) {
        if (result.status === 'fulfilled') {
          initialModels[result.value.model] = {
            status: result.value.exists ? 'ready' : 'queued'
          }
        }
      }
      backendState.value = {
        active: true,
        models: initialModels,
        latest_sri: latestSRI.value?.latest_file
      }
      // Preload immediately if we have a selected model with predictions
      await preloadAllFrames()
    }

    startStatusPolling()
  }
}

/**
 * Start polling the backend status every 3 seconds.
 */
function startStatusPolling() {
  stopStatusPolling()
  // Do an immediate poll, then schedule regular ones
  pollRealtimeStatus()
  statusPollInterval = setInterval(pollRealtimeStatus, POLL_INTERVAL_MS)
}

/**
 * Stop polling.
 */
function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval)
    statusPollInterval = null
  }
}

/**
 * Fetch the latest state from the backend and sync local UI.
 *
 * This is the core of the backend-driven approach: every 3 seconds we
 * ask "what's happening?" and update our refs accordingly. If the backend
 * reports active: false (e.g. it crashed or was stopped from another tab),
 * we stop polling and reset the UI.
 */
async function pollRealtimeStatus() {
  try {
    const state = await api.getRealTimeStatus()
    const prevState = backendState.value
    backendState.value = state

    // If the backend is no longer active, stop everything
    if (!state.active) {
      realTimeActive.value = false
      stopStatusPolling()
      notification.value = ''
      return
    }

    // Sync latest SRI info from backend state
    if (state.latest_sri) {
      latestSRI.value = { latest_file: state.latest_sri }
    }

    // Show notification toast when the backend sends a NEW one
    // (compare against lastShownNotification, not the display ref which auto-clears)
    if (state.notification && state.notification !== lastShownNotification) {
      lastShownNotification = state.notification
      showNotification(state.notification)
    }

    const sriChanged = state.latest_sri && state.latest_sri !== prevState?.latest_sri

    if (sriChanged) {
      // New SRI arrived. The latestTimestamp watcher fires automatically and
      // updates groundtruth frames while keeping prediction frames visible
      // (keepPredictions=true) so the map doesn't go blank mid-job.
      // Exception: if the model is already 'ready' for the new data (e.g. local
      // mock mode where predictions are instant), do a full reload immediately.
      const modelAlreadyReady = selectedModel.value &&
        state.models[selectedModel.value]?.status === 'ready'
      if (modelAlreadyReady) {
        await preloadAllFrames()
      }
    } else if (selectedModel.value && state.models[selectedModel.value]) {
      const prevModelStatus = prevState?.models?.[selectedModel.value]?.status
      const newModelStatus = state.models[selectedModel.value].status
      if (newModelStatus === 'ready' && prevModelStatus !== 'ready') {
        // Model just finished computing — load the new predictions.
        await preloadAllFrames()
      }
    }

    lastRefresh.value = new Date().toLocaleTimeString()
  } catch (e) {
    console.error('Failed to poll real-time status:', e)
  }
}

/**
 * Show a notification toast that auto-dismisses after 4 seconds.
 */
function showNotification(message) {
  notification.value = message
  if (notificationTimer) clearTimeout(notificationTimer)
  notificationTimer = setTimeout(() => {
    notification.value = ''
    notificationTimer = null
  }, 4000)
}

// ---- Status display helpers ----

/**
 * Get the display text for a model's status.
 * Reads from backendState when real-time is active.
 */
function statusText(model) {
  // Test model is always "Ready" — it uses static pre-existing data
  if (model.toUpperCase() === 'TEST') return 'Ready'

  if (!realTimeActive.value || !backendState.value) return 'Idle'

  const modelInfo = backendState.value.models[model]
  if (!modelInfo) return 'Idle'

  const s = modelInfo.status
  if (s === 'queued') return 'In Queue'
  if (s === 'computing') return 'Computing'
  if (s === 'ready') return 'Ready'
  if (s === 'failed') return 'Failed'

  return 'Idle'
}

function statusClass(model) {
  if (model.toUpperCase() === 'TEST') return 'bg-emerald-900/50 text-emerald-400'

  if (!realTimeActive.value || !backendState.value) return 'bg-gray-700 text-gray-400'

  const modelInfo = backendState.value.models[model]
  if (!modelInfo) return 'bg-gray-700 text-gray-400'

  const s = modelInfo.status
  if (s === 'queued') return 'bg-yellow-900/50 text-yellow-400'
  if (s === 'computing') return 'bg-blue-900/50 text-blue-400'
  if (s === 'ready') return 'bg-emerald-900/50 text-emerald-400'
  if (s === 'failed') return 'bg-red-900/50 text-red-400'

  return 'bg-gray-700 text-gray-400'
}

function statusDotClass(model) {
  if (model.toUpperCase() === 'TEST') return 'bg-emerald-500'

  if (!realTimeActive.value || !backendState.value) return 'bg-gray-400'

  const modelInfo = backendState.value.models[model]
  if (!modelInfo) return 'bg-gray-400'

  const s = modelInfo.status
  if (s === 'queued') return 'bg-yellow-500 animate-pulse'
  if (s === 'computing') return 'bg-blue-500 animate-spin-slow'
  if (s === 'ready') return 'bg-emerald-500'
  if (s === 'failed') return 'bg-red-500'

  return 'bg-gray-400'
}

function formatSriFilename(filename) {
  // "22-11-2025-20-00.hdf" → "22/11/2025 21:00" (UTC → Europe/Rome)
  const name = filename.replace('.hdf', '')
  const parts = name.split('-')
  if (parts.length !== 5) return filename
  const [day, month, year, hour, minute] = parts
  const utcDate = new Date(`${year}-${month}-${day}T${hour}:${minute}:00Z`)
  return formatDateTimeInRome(utcDate)
}

// ---- Data fetching ----
async function fetchLatestSRI() {
  try {
    latestSRI.value = await api.getLatestSRI()
  } catch (e) {
    console.error('Failed to fetch SRI:', e)
  }
}

/**
 * Periodic SRI polling — runs always (even when real-time is off).
 * This ensures the map always shows the latest groundtruth data and the
 * "Latest Data" indicator in the sidebar stays up-to-date.
 *
 * When real-time IS active, pollRealtimeStatus already syncs SRI data
 * every 3s, so we skip the independent fetch to avoid redundant calls.
 */
function startSriPolling() {
  stopSriPolling()
  sriPollInterval = setInterval(async () => {
    if (!realTimeActive.value) {
      await fetchLatestSRI()
    }
  }, SRI_POLL_INTERVAL_MS)
}

function stopSriPolling() {
  if (sriPollInterval) {
    clearInterval(sriPollInterval)
    sriPollInterval = null
  }
}

// ---- Lifecycle ----
onMounted(async () => {
  await fetchLatestSRI()

  // Immediately show groundtruth on the map (even before a model is selected)
  await preloadAllFrames()

  // Start periodic SRI polling so groundtruth updates even when RT is off
  startSriPolling()

  // Auto-select first model if available (this triggers watch → preload with predictions)
  if (models.value.length > 0 && !selectedModel.value) {
    selectedModel.value = models.value[0]
  }

  // Default: all real models selected for the ensemble (exclude Test)
  if (ensembleModels.value.length === 0) {
    ensembleModels.value = models.value.filter(m => m !== 'Test')
  }

  // Check if the backend service is already running (e.g. page refresh, second tab)
  try {
    const state = await api.getRealTimeStatus()
    if (state.active) {
      realTimeActive.value = true
      backendState.value = state
      // Sync SRI from backend state
      if (state.latest_sri) {
        latestSRI.value = { latest_file: state.latest_sri }
      }
      startStatusPolling()
      // Preload frames for the current data (on page refresh, watchers
      // skip because realTimeActive is already true by this point)
      await preloadAllFrames()
    }
  } catch (e) {
    console.error('Failed to check real-time status on mount:', e)
  }
})

onUnmounted(() => {
  stopPlay()
  stopStatusPolling()
  stopSriPolling()
  if (notificationTimer) {
    clearTimeout(notificationTimer)
    notificationTimer = null
  }
})
</script>

<style scoped>
/* Toast notification slide-in/out transitions */
.toast-enter-active {
  transition: all 0.4s ease-out;
}
.toast-leave-active {
  transition: all 0.3s ease-in;
}
.toast-enter-from {
  opacity: 0;
  transform: translate(-50%, -20px);
}
.toast-leave-to {
  opacity: 0;
  transform: translate(-50%, -20px);
}

/* Slow spin animation for "computing" status dots */
@keyframes spin-slow {
  to { transform: rotate(360deg); }
}
.animate-spin-slow {
  animation: spin-slow 2s linear infinite;
}

/* Colorbar stack — vertical position via plain CSS so we bypass any
   Tailwind v4 arbitrary-value parsing flakiness with calc(...env(...)). */
.colorbar-stack {
  /* Mobile (default): hug the timeline slider */
  top: 64px;
  bottom: calc(75px + env(safe-area-inset-bottom));
}
@media (min-width: 640px) {
  .colorbar-stack {
    top: auto;
    bottom: 110px;
  }
}
</style>
