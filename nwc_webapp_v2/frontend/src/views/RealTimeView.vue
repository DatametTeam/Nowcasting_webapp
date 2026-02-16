<!--
  RealTimeView.vue — Real-time prediction monitoring (main page).

  This is the heart of the app. It shows:
  - An interactive Leaflet map with radar overlay (past + future)
  - Animation controls: play/pause + timeline slider (-60 to +60 min)
  - Model selector to switch predictions
  - Start/Pause Real Time button for HPC simulation
  - Precipitation colorbar legend
  - Auto-refreshing model status

  TIMELINE STRUCTURE (25 frames total):
  - Frames 0-12:  Past groundtruth (SRI data) — -60 to 0 minutes
  - Frames 13-24: Future predictions (model)  — +5 to +60 minutes

  The frame index maps to minutes via: (index - 12) * 5
    index 0  → -60 min (past SRI)
    index 12 →   0 min (current SRI)
    index 13 →  +5 min (prediction lead_time=0)
    index 24 → +60 min (prediction lead_time=11)

  SIMULATION CYCLE (repeats every ~45 seconds when real-time is active):
    0s   → New SRI data generated → "New data found!" notification
    0s   → All models → "In Queue" (yellow)
    5s   → All models → "Computing" (blue, spinning)
    15s  → Each model randomly → "Ready" (80%) or "Failed" (20%)
    15s  → If selected model is ready → preload new frames on map
    45s  → Next cycle starts
-->
<template>
  <div class="h-[calc(100vh-3.5rem)] flex">

    <!-- ================================================================ -->
    <!-- LEFT: Map area (takes all available width)                       -->
    <!-- ================================================================ -->
    <div class="flex-1 flex flex-col relative">
      <RadarMap
        ref="radarMap"
        class="flex-1"
        :overlay-opacity="overlayOpacity"
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

      <!-- Colorbar — floating on the map, bottom right (above the timeline bar) -->
      <div class="absolute bottom-[110px] right-[10px] z-[1001]">
        <ColorBar />
      </div>

      <!-- ============================================================ -->
      <!-- BOTTOM BAR: Timeline controls (floating over the map)        -->
      <!-- ============================================================ -->
      <div class="absolute bottom-0 left-0 right-0 z-[1000]
                  bg-gradient-to-t from-black/80 via-black/60 to-transparent
                  px-6 pt-10 pb-4">

        <!-- Current time display -->
        <div class="flex items-center justify-between text-white mb-2">
          <div class="text-sm font-medium">
            <span class="text-gray-300">Model:</span>
            <span class="ml-1 font-bold">{{ selectedModel || 'None' }}</span>
          </div>
          <div class="text-center">
            <span class="text-2xl font-bold tabular-nums">
              {{ frameMinutesDisplay }}
            </span>
            <span
              class="ml-2 text-xs font-medium px-2 py-0.5 rounded-full"
              :class="frameIndex <= 12
                ? 'bg-emerald-500/30 text-emerald-300'
                : 'bg-blue-500/30 text-blue-300'"
            >
              {{ frameIndex <= 12 ? 'Observed' : 'Forecast' }}
            </span>
          </div>
          <div class="text-sm text-gray-300">
            {{ latestTimestampDisplay || 'No data' }}
          </div>
        </div>

        <!-- Timeline slider -->
        <div class="flex items-center gap-4">
          <!-- Play/Pause button -->
          <button
            @click="togglePlay"
            class="w-10 h-10 flex items-center justify-center rounded-full
                   bg-white/20 hover:bg-white/30 text-white transition-colors
                   backdrop-blur-sm"
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
          <div class="flex-1 relative">
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
            <!-- Tick marks — show every 15 minutes for readability -->
            <div class="flex justify-between mt-1 px-0.5">
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
          <button
            @click="cycleSpeed"
            class="px-3 py-1.5 rounded-full bg-white/20 hover:bg-white/30
                   text-white text-xs font-medium transition-colors backdrop-blur-sm"
            title="Animation speed"
          >
            {{ speedLabel }}
          </button>

          <!-- Opacity slider -->
          <div class="flex items-center gap-2" title="Overlay opacity">
            <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
            </svg>
            <input
              type="range"
              v-model.number="overlayOpacity"
              min="0.1"
              max="1"
              step="0.05"
              class="w-16 h-1.5 appearance-none cursor-pointer rounded-full
                     bg-white/20 accent-white
                     [&::-webkit-slider-thumb]:appearance-none
                     [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3
                     [&::-webkit-slider-thumb]:rounded-full
                     [&::-webkit-slider-thumb]:bg-white"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- ================================================================ -->
    <!-- RIGHT: Sidebar panel                                             -->
    <!-- ================================================================ -->
    <div class="w-72 bg-white border-l border-gray-200 flex flex-col overflow-y-auto">

      <!-- Start / Pause Real Time Button -->
      <div class="p-4 border-b border-gray-100">
        <button
          @click="toggleRealTime"
          class="w-full py-3 px-4 rounded-lg font-semibold text-sm transition-all
                 flex items-center justify-center gap-2"
          :class="realTimeActive
            ? 'bg-red-50 text-red-600 border border-red-200 hover:bg-red-100'
            : 'bg-blue-600 text-white hover:bg-blue-700 shadow-sm'"
        >
          <!-- Pause icon -->
          <svg v-if="realTimeActive" class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <rect x="6" y="4" width="4" height="16" rx="1" />
            <rect x="14" y="4" width="4" height="16" rx="1" />
          </svg>
          <!-- Play icon -->
          <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path d="M5 3l14 9-14 9V3z" />
          </svg>
          {{ realTimeActive ? 'Pause Real Time' : 'Start Real Time' }}
        </button>
        <p v-if="realTimeActive" class="text-[10px] text-center text-gray-400 mt-1.5">
          Simulation cycle: 45s
        </p>
      </div>

      <!-- Model Selector -->
      <div class="p-4 border-b border-gray-100">
        <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Active Model
        </h3>
        <select
          v-model="selectedModel"
          class="w-full rounded-lg border border-gray-200 bg-gray-50 px-3 py-2.5 text-sm
                 font-medium focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          <option value="" disabled>Select model...</option>
          <option v-for="model in models" :key="model" :value="model">
            {{ model }}
          </option>
        </select>
      </div>

      <!-- Latest Data -->
      <div class="p-4 border-b border-gray-100">
        <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Latest Data
        </h3>
        <div v-if="latestSRI" class="flex items-center gap-2">
          <div class="w-2 h-2 rounded-full" :class="latestSRI.latest_file ? 'bg-green-400' : 'bg-red-400'" />
          <span class="text-sm text-gray-700 font-medium">
            {{ latestSRI.latest_file ? formatSriFilename(latestSRI.latest_file) : 'No data' }}
          </span>
        </div>
        <div v-else class="flex items-center gap-2">
          <div class="w-2 h-2 rounded-full bg-gray-300 animate-pulse" />
          <span class="text-sm text-gray-400">Loading...</span>
        </div>
      </div>

      <!-- Model Status List -->
      <div class="p-4 border-b border-gray-100 flex-1">
        <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Model Status
        </h3>

        <div class="space-y-1">
          <div
            v-for="model in models"
            :key="model"
            @click="selectedModel = model"
            class="flex items-center justify-between py-2.5 px-3 rounded-lg cursor-pointer
                   transition-colors"
            :class="selectedModel === model
              ? 'bg-blue-50 border border-blue-200'
              : 'hover:bg-gray-50'"
          >
            <span
              class="text-sm"
              :class="selectedModel === model ? 'font-semibold text-blue-700' : 'text-gray-700'"
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

      <!-- Info -->
      <div class="p-4">
        <p v-if="lastRefresh" class="text-[10px] text-gray-400">
          Status updated {{ lastRefresh }}
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import api from '../api.js'
import { useConfigStore } from '../stores/config.js'
import RadarMap from '../components/RadarMap.vue'
import ColorBar from '../components/ColorBar.vue'

const configStore = useConfigStore()
const models = computed(() => configStore.models)

// ---- State ----
const radarMap = ref(null)
const selectedModel = ref('')
const frameIndex = ref(12)  // Start at index 12 = "0 min" (current time)
const playing = ref(false)
const speed = ref(1)
const latestSRI = ref(null)
const overlayOpacity = ref(0.7)
const modelStatuses = ref({})
const lastRefresh = ref('')

// Real-time simulation state
const realTimeActive = ref(false)
const simulationStatuses = ref({})   // { model: 'paused'|'queued'|'computing'|'ready'|'failed' }
const notification = ref('')         // Toast message string (empty = hidden)
const simulationTimers = []          // Pending setTimeout IDs for cleanup

let playInterval = null
let statusRefreshInterval = null

// ---- Constants ----
const TOTAL_FRAMES = 25        // 13 past (including current) + 12 future
const CURRENT_INDEX = 12       // Index of "0 min" in the frame array

// ---- Speed control ----
const speeds = [0.5, 1, 2]
const speedLabel = computed(() => `${speed.value}x`)

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
  const hh = String(frameDt.getHours()).padStart(2, '0')
  const mm = String(frameDt.getMinutes()).padStart(2, '0')

  return `${hh}:${mm} (${offsetStr})`
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
  return `${year}-${month}-${day}T${hour}:${minute}`
})

const latestTimestampDisplay = computed(() => {
  if (!latestSRI.value?.latest_file) return null
  return formatSriFilename(latestSRI.value.latest_file)
})

// ---- Preload all 25 frames when model or timestamp changes ----

async function preloadAllFrames() {
  if (!selectedModel.value || !latestTimestamp.value || !radarMap.value) return

  const baseDt = new Date(latestTimestamp.value)

  // Build 25 URLs: 13 past/current (groundtruth) + 12 future (predictions)
  const urls = Array.from({ length: TOTAL_FRAMES }, (_, i) => {
    const minuteOffset = frameToMinutes(i)

    if (minuteOffset <= 0) {
      // Past or current: use groundtruth (SRI) overlay
      const pastDt = new Date(baseDt.getTime() + minuteOffset * 60000)
      const ts = formatIsoTimestamp(pastDt)
      return api.groundtruthOverlayUrl(ts)
    } else {
      // Future: use prediction overlay (lead_time 0-11)
      const leadTimeIndex = Math.round(minuteOffset / 5) - 1
      return api.overlayUrl(selectedModel.value, latestTimestamp.value, leadTimeIndex)
    }
  })

  await radarMap.value.preloadFrames(urls)
}

/**
 * Format a Date object as ISO timestamp string (YYYY-MM-DDTHH:MM).
 */
function formatIsoTimestamp(dt) {
  const year = dt.getFullYear()
  const month = String(dt.getMonth() + 1).padStart(2, '0')
  const day = String(dt.getDate()).padStart(2, '0')
  const hours = String(dt.getHours()).padStart(2, '0')
  const minutes = String(dt.getMinutes()).padStart(2, '0')
  return `${year}-${month}-${day}T${hours}:${minutes}`
}

// When model changes → preload all frames for that model
watch(selectedModel, () => { preloadAllFrames() })

// When latest timestamp changes (new SRI data) → preload new frames.
// Skip during simulation: the simulation cycle manages preloading explicitly
// (at the 15s mark when the selected model is "ready").
watch(latestTimestamp, () => {
  if (!realTimeActive.value) preloadAllFrames()
})

// When frame index changes (slider drag) → instantly show that frame
watch(frameIndex, (newIdx) => {
  if (radarMap.value) radarMap.value.showFrame(newIdx)
})

// When opacity slider changes → update the currently visible frame
watch(overlayOpacity, (newOpacity) => {
  if (radarMap.value) radarMap.value.setOverlayOpacity(newOpacity)
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

// ---- Real-Time Simulation ----

/**
 * Toggle real-time simulation on/off.
 * Start: initialize simulation, kick off first cycle.
 * Stop: clear all timers, reset statuses to "paused".
 */
function toggleRealTime() {
  realTimeActive.value = !realTimeActive.value

  if (realTimeActive.value) {
    // Initialize all models to "paused" then start cycle
    for (const model of models.value) {
      simulationStatuses.value[model] = 'paused'
    }
    runSimulationCycle()
  } else {
    // Stop: clear all pending timers
    clearSimulationTimers()
    // Reset all statuses to paused
    for (const model of models.value) {
      simulationStatuses.value[model] = 'paused'
    }
    // Dismiss notification
    notification.value = ''
  }
}

/**
 * Clear all pending simulation timers (setTimeout IDs).
 */
function clearSimulationTimers() {
  while (simulationTimers.length > 0) {
    clearTimeout(simulationTimers.pop())
  }
}

/**
 * Show a notification toast that auto-dismisses after 4 seconds.
 */
function showNotification(message) {
  notification.value = message
  const id = setTimeout(() => {
    notification.value = ''
  }, 4000)
  simulationTimers.push(id)
}

/**
 * Run one full simulation cycle (45 seconds):
 *   0s  → generate new data, show notification, set all models to "queued"
 *   5s  → set all models to "computing"
 *  15s  → randomly set each model to "ready" (80%) or "failed" (20%)
 *  45s  → start next cycle
 */
async function runSimulationCycle() {
  // Guard: bail if simulation was stopped while awaiting
  if (!realTimeActive.value) return

  // --- 0s: Generate new mock data ---
  try {
    const result = await api.generateNextMockData()
    console.log('Mock data generated:', result.filename)
  } catch (e) {
    console.error('Failed to generate mock data:', e)
  }

  // Guard again after async call
  if (!realTimeActive.value) return

  // Refresh SRI info so the map knows about the new file
  await fetchLatestSRI()
  lastRefresh.value = new Date().toLocaleTimeString()

  if (!realTimeActive.value) return

  // Show notification with the time of the new data
  const timeStr = latestTimestampDisplay.value || 'unknown'
  showNotification(`New data found!  ${timeStr}`)

  // Set all models → "queued"
  for (const model of models.value) {
    simulationStatuses.value[model] = 'queued'
  }

  // --- 5s: All models → "computing" ---
  const computingTimer = setTimeout(() => {
    if (!realTimeActive.value) return
    for (const model of models.value) {
      simulationStatuses.value[model] = 'computing'
    }
  }, 5000)
  simulationTimers.push(computingTimer)

  // --- 15s: Each model randomly → "ready" (80%) or "failed" (20%) ---
  const resultsTimer = setTimeout(async () => {
    if (!realTimeActive.value) return

    for (const model of models.value) {
      simulationStatuses.value[model] = Math.random() < 0.8 ? 'ready' : 'failed'
    }

    // If the selected model is "ready", preload its new frames
    if (selectedModel.value && simulationStatuses.value[selectedModel.value] === 'ready') {
      await preloadAllFrames()
    }

    lastRefresh.value = new Date().toLocaleTimeString()
  }, 15000)
  simulationTimers.push(resultsTimer)

  // --- 45s: Next cycle ---
  const nextCycleTimer = setTimeout(() => {
    if (!realTimeActive.value) return
    runSimulationCycle()
  }, 45000)
  simulationTimers.push(nextCycleTimer)
}

// ---- Data fetching ----
async function fetchLatestSRI() {
  try {
    latestSRI.value = await api.getLatestSRI()
  } catch (e) {
    console.error('Failed to fetch SRI:', e)
  }
}

async function fetchModelStatuses() {
  const promises = models.value.map(async (model) => {
    try {
      const result = await api.getJobStatus(model)
      modelStatuses.value[model] = result
    } catch (e) {
      console.error(`Failed to fetch status for ${model}:`, e)
    }
  })
  await Promise.all(promises)
}

async function refreshAll() {
  await Promise.all([fetchLatestSRI(), fetchModelStatuses()])
  lastRefresh.value = new Date().toLocaleTimeString()
}

// ---- Auto-refresh model statuses (always on, every 30s) ----
function startStatusAutoRefresh() {
  stopStatusAutoRefresh()
  statusRefreshInterval = setInterval(fetchModelStatuses, 30000)
}

function stopStatusAutoRefresh() {
  if (statusRefreshInterval) {
    clearInterval(statusRefreshInterval)
    statusRefreshInterval = null
  }
}

// ---- Status display helpers ----

/**
 * When real-time simulation is active, use simulationStatuses.
 * Otherwise fall back to the original modelStatuses from the backend.
 */
function statusText(model) {
  if (!realTimeActive.value) return 'Paused'

  const simStatus = simulationStatuses.value[model]
  if (simStatus === 'queued') return 'In Queue'
  if (simStatus === 'computing') return 'Computing'
  if (simStatus === 'ready') return 'Ready'
  if (simStatus === 'failed') return 'Failed'

  // Fallback for brief moments before simulation sets a status
  return 'Paused'
}

function statusClass(model) {
  if (!realTimeActive.value) return 'bg-gray-100 text-gray-500'

  const simStatus = simulationStatuses.value[model]
  if (simStatus === 'queued') return 'bg-yellow-100 text-yellow-700'
  if (simStatus === 'computing') return 'bg-blue-100 text-blue-700'
  if (simStatus === 'ready') return 'bg-emerald-100 text-emerald-700'
  if (simStatus === 'failed') return 'bg-red-100 text-red-700'

  return 'bg-gray-100 text-gray-500'
}

function statusDotClass(model) {
  if (!realTimeActive.value) return 'bg-gray-400'

  const simStatus = simulationStatuses.value[model]
  if (simStatus === 'queued') return 'bg-yellow-500 animate-pulse'
  if (simStatus === 'computing') return 'bg-blue-500 animate-spin-slow'
  if (simStatus === 'ready') return 'bg-emerald-500'
  if (simStatus === 'failed') return 'bg-red-500'

  return 'bg-gray-400'
}

function formatSriFilename(filename) {
  // "22-11-2025-20-00.hdf" → "22/11/2025 20:00"
  const name = filename.replace('.hdf', '')
  const parts = name.split('-')
  if (parts.length !== 5) return filename
  return `${parts[0]}/${parts[1]}/${parts[2]} ${parts[3]}:${parts[4]}`
}

// ---- Lifecycle ----
onMounted(async () => {
  await refreshAll()
  // Start auto-refresh for model statuses (always on)
  startStatusAutoRefresh()
  // Auto-select first model if available
  if (models.value.length > 0 && !selectedModel.value) {
    selectedModel.value = models.value[0]
  }
})

onUnmounted(() => {
  stopPlay()
  clearSimulationTimers()
  stopStatusAutoRefresh()
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
</style>
