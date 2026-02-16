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
          Polling every 3s
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
const lastRefresh = ref('')

// Real-time state (driven by backend)
const realTimeActive = ref(false)
const backendState = ref(null)   // Full state from GET /api/realtime/status
const notification = ref('')     // Toast message string (empty = hidden)

let playInterval = null
let statusPollInterval = null
let notificationTimer = null
let lastShownNotification = ''  // Track which notification we already displayed

// ---- Constants ----
const TOTAL_FRAMES = 25        // 13 past (including current) + 12 future
const CURRENT_INDEX = 12       // Index of "0 min" in the frame array
const POLL_INTERVAL_MS = 3000  // How often we poll the backend

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
  // Show the frame matching the current slider position (not always frame 0)
  radarMap.value.showFrame(frameIndex.value)
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
// During real-time mode, preloading is triggered when the selected model
// transitions to "ready" (in pollRealtimeStatus), so we skip here.
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

    // Preload frames when:
    // 1. New SRI data arrives — always preload so groundtruth (frames 0-12)
    //    updates immediately, even if no model predictions are ready yet.
    //    Prediction frames (13-24) will show blank until a model is ready.
    // 2. Selected model transitions to "ready" — preload again so the
    //    prediction frames (13-24) now have data to show.
    const sriChanged = state.latest_sri && state.latest_sri !== prevState?.latest_sri

    if (sriChanged) {
      await preloadAllFrames()
    } else if (selectedModel.value && state.models[selectedModel.value]) {
      const prevModelStatus = prevState?.models?.[selectedModel.value]?.status
      const newModelStatus = state.models[selectedModel.value].status
      if (newModelStatus === 'ready' && prevModelStatus !== 'ready') {
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
  if (model.toUpperCase() === 'TEST') return 'bg-emerald-100 text-emerald-700'

  if (!realTimeActive.value || !backendState.value) return 'bg-gray-100 text-gray-500'

  const modelInfo = backendState.value.models[model]
  if (!modelInfo) return 'bg-gray-100 text-gray-500'

  const s = modelInfo.status
  if (s === 'queued') return 'bg-yellow-100 text-yellow-700'
  if (s === 'computing') return 'bg-blue-100 text-blue-700'
  if (s === 'ready') return 'bg-emerald-100 text-emerald-700'
  if (s === 'failed') return 'bg-red-100 text-red-700'

  return 'bg-gray-100 text-gray-500'
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
  // "22-11-2025-20-00.hdf" → "22/11/2025 20:00"
  const name = filename.replace('.hdf', '')
  const parts = name.split('-')
  if (parts.length !== 5) return filename
  return `${parts[0]}/${parts[1]}/${parts[2]} ${parts[3]}:${parts[4]}`
}

// ---- Data fetching (initial load) ----
async function fetchLatestSRI() {
  try {
    latestSRI.value = await api.getLatestSRI()
  } catch (e) {
    console.error('Failed to fetch SRI:', e)
  }
}

// ---- Lifecycle ----
onMounted(async () => {
  await fetchLatestSRI()

  // Auto-select first model if available
  if (models.value.length > 0 && !selectedModel.value) {
    selectedModel.value = models.value[0]
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
</style>
