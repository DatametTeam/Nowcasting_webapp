<!--
  LiveView.vue — Real-time multi-product radar monitor.

  Shows all 4 radar products (SRI, VMI, ETM, VIL) for a rolling lookback window
  (default 1h, up to 12h). Auto-loads on mount and polls every 5 minutes for
  new data. "Follow Live" toggle auto-jumps to the latest frame on new data;
  when off, the user's current frame position is preserved across updates.
-->
<template>
  <div class="h-[calc(100dvh-3rem)] sm:h-[calc(100vh-3.5rem)] flex overflow-hidden">

    <!-- ================================================================ -->
    <!-- LEFT: Map area                                                    -->
    <!-- ================================================================ -->
    <div class="flex-1 flex flex-col relative min-w-0">
      <RadarMap ref="radarMap" class="flex-1" @mapclick="onMapClick" />

      <!-- Mobile sidebar toggle — top-right (Leaflet layer/search controls now live on top-left) -->
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

      <!-- Stacked colorbars — bottom right, above timeline.
           top-16 on mobile keeps the column below the sidebar toggle (top-3 + h-10);
           overflow-y scrolls if too many products are enabled. -->
      <div
        v-if="settings.showColorbars"
        class="colorbar-stack absolute right-[10px] z-[1001]
               flex flex-col justify-end gap-1.5 items-end
               overflow-y-auto"
      >
        <ColorBar
          v-for="product in visibleProducts"
          :key="product"
          :legend="radarProducts[product]"
          :product-name="SHORT_NAMES[product]"
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
        <!-- Top row: layer names | datetime | (right spacer to keep datetime centered) -->
        <div class="flex items-center justify-between text-white mb-2">
          <div class="text-xs font-medium text-gray-300 hidden sm:block truncate max-w-[160px]">
            {{ visibleProducts.map(p => SHORT_NAMES[p]).join(' + ') || '—' }}
          </div>
          <div class="text-center">
            <span class="text-xs sm:text-xl font-bold tabular-nums tracking-tight">
              {{ currentTimestampDisplay }}
            </span>
          </div>
          <div class="hidden sm:block max-w-[160px] flex-1" />
        </div>

        <!-- Slider row — items-start so the slider thumb (centered inside its own
             h-9 wrapper) lines up vertically with the play button center. -->
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
                :class="tick.hideOnMobile ? 'hidden sm:inline' : ''"
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

    <!-- On mobile: fixed overlay (translate in/out).
         On desktop (lg+): in-flow flex child — map shrinks when open.
         Width on desktop transitions 0→18rem; mobile always w-72 (translate hides it). -->
    <div
      class="bg-gray-900 flex-shrink-0 overflow-hidden
             fixed right-0 top-12 sm:top-14 bottom-0 z-[1101] w-72
             lg:relative lg:top-auto lg:right-auto lg:bottom-auto lg:z-auto
             transition-all duration-200 ease-out"
      :class="sidebarOpen
        ? 'translate-x-0 border-l border-gray-700 lg:w-72'
        : 'translate-x-full lg:translate-x-0 lg:w-0 border-l-0'"
    >
      <!-- Inner wrapper keeps content at w-72 so it doesn't reflow during width animation -->
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
          <h2 class="text-white font-bold text-base">Real Time</h2>
          <p class="text-gray-400 text-xs mt-0.5">Live multi-product radar</p>
        </div>

        <!-- Live status card -->
        <div class="bg-gray-800 rounded-lg p-3 space-y-2.5">
          <!-- Status indicator -->
          <div class="flex items-center gap-2">
            <div
              class="w-2 h-2 rounded-full flex-shrink-0 transition-colors duration-300"
              :class="!isLoaded
                    ? 'bg-gray-500'
                    : productJustFoundFlag || wsConnected
                      ? 'bg-green-400'
                      : 'bg-yellow-400 animate-pulse'"
            />
            <span
              class="text-xs transition-colors duration-300"
              :class="productJustFoundFlag ? 'text-green-400 font-medium' : 'text-gray-300'"
            >{{ liveStatusText }}</span>
            <span v-if="isLoaded"
                  class="ml-auto text-[10px] text-gray-500 tabular-nums">
              next: {{ nextUpdateText }}
            </span>
            <span
              v-if="isLoaded"
              :title="wsConnected ? 'WebSocket connected — instant updates' : 'WebSocket offline — using 5-min poll'"
              class="ml-1 flex items-center gap-0.5"
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

          <!-- Jump to latest -->
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
              v-for="h in lookbackOptions"
              :key="h"
              @click="setLookback(h)"
              :disabled="isLoading"
              class="py-1.5 rounded text-xs font-semibold transition-colors border
                     disabled:cursor-not-allowed"
              :class="lookbackHours === h
                ? 'bg-blue-600 border-blue-500 text-white'
                : 'bg-gray-800 border-gray-700 text-gray-400 hover:bg-gray-700'"
            >
              {{ h }}h
            </button>
          </div>
        </div>

        <!-- Error message + retry -->
        <div v-if="loadError && !isLoading" class="bg-red-900/30 border border-red-700/50 rounded-lg p-3 space-y-2">
          <p class="text-red-400 text-xs leading-snug">{{ loadError }}</p>
          <button
            @click="loadData({ preserve: false })"
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
          <span>
            {{ loadProgress.total > 0
              ? `Loading ${loadProgress.loaded}/${loadProgress.total} frames…`
              : 'Fetching timestamps…' }}
          </span>
        </div>

        <!-- Layers -->
        <div class="space-y-2">
          <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Layers</h3>

          <div
            v-for="product in productOrder"
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
                {{ SHORT_NAMES[product] }}
              </label>
              <!-- ✓ for 5s after data lands, then spinner while still searching -->
              <span
                v-if="productJustFound[product]"
                class="text-green-400 text-xs font-bold flex-shrink-0"
              >✓</span>
              <svg
                v-else-if="pendingProducts[product]"
                class="animate-spin h-3 w-3 text-blue-400 flex-shrink-0"
                viewBox="0 0 24 24"
              >
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
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

            <div v-if="productStats[product]" class="text-xs">
              <span class="text-green-400 font-medium">{{ productStats[product].found }}</span>
              <span class="text-gray-500">/{{ productStats[product].expected }} frames</span>
              <button
                v-if="timelineMissingTs[product]?.length > 0"
                @click="toggleMissing(product)"
                class="text-amber-400 hover:text-amber-300 ml-1.5 underline underline-offset-2"
              >
                {{ timelineMissingTs[product].length }} missing
                {{ showMissingFor === product ? '▲' : '▼' }}
              </button>
            </div>

            <div
              v-if="showMissingFor === product && timelineMissingTs[product]?.length"
              class="mt-1 space-y-0.5 max-h-36 overflow-y-auto rounded bg-black/30 px-2 py-1.5"
            >
              <div
                v-for="ts in timelineMissingTs[product]"
                :key="ts"
                class="font-mono text-[10px] text-amber-300/80"
              >{{ formatMissingTs(ts) }}</div>
            </div>
          </div>
        </div>

        <!-- Motion field layer (AMV / LK) -->
        <div class="space-y-2">
          <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Motion Field</h3>
          <div class="bg-gray-800 rounded-lg p-3 space-y-2">
            <!-- Source selector: None / AMV / LK -->
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
            <!-- LK display sub-controls: Particles / Arrows / Both -->
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

        <!-- Summary -->
        <div v-if="isLoaded" class="bg-gray-800 rounded-lg p-3 text-xs text-gray-400 space-y-1">
          <div class="flex justify-between">
            <span>Frames</span>
            <span class="text-white font-medium">{{ timestamps.length }}</span>
          </div>
          <div class="flex justify-between">
            <span>Window</span>
            <span class="text-white font-medium">{{ lookbackHours }}h</span>
          </div>
        </div>

      </div>
    </div><!-- /inner wrapper -->
    </div><!-- /sidebar outer -->
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, onMounted, onUnmounted, onActivated, onDeactivated, nextTick } from 'vue'
import { useConfigStore } from '../stores/config.js'
import { useSettingsStore } from '../stores/settings.js'
import api from '../api.js'
import RadarMap from '../components/RadarMap.vue'
import ColorBar from '../components/ColorBar.vue'
import { useRealtimeWs } from '../composables/useRealtimeWs.js'
import { useRadarStatusWs } from '../composables/useRadarStatusWs.js'
import { useMotionLayer } from '../composables/useMotionLayer.js'

const configStore = useConfigStore()
const settings = useSettingsStore()
const radarMap = ref(null)
// Default open on desktop, closed on mobile
const sidebarOpen = ref(false)

const SHORT_NAMES = { SRI_adj: 'SRI', VMI: 'VMI', ETM: 'ETM', VIL: 'VIL', IR_108: 'IR' }
// Ordered top-to-bottom on the map (index 0 = topmost layer). IR_108 is last = bottommost.
const productOrder = ref(['SRI_adj', 'VMI', 'ETM', 'VIL', 'IR_108'])
const lookbackOptions = [1, 2, 4, 6, 12]
const POLL_MS = 5 * 60 * 1000  // 5-minute polling

// ---- Speed ----
const speeds = [0.5, 1, 2, 4]
const playSpeed = ref(settings.defaultSpeed)
function cycleSpeed() {
  const idx = speeds.indexOf(playSpeed.value)
  playSpeed.value = speeds[(idx + 1) % speeds.length]
  if (isPlaying.value) { stopAnimation(); startAnimation() }
}

// ---- Live state ----
const lookbackHours = ref(settings.defaultLookback)
const followLive    = ref(true)
const nextUpdateSecs = ref(POLL_MS / 1000)

// ---- Layer config ----
const layerConfig = ref({
  SRI_adj: { enabled: true, opacity: 0.8 },
  VMI:     { enabled: true, opacity: 0.7 },
  ETM:     { enabled: true, opacity: 0.7 },
  VIL:     { enabled: true, opacity: 0.7 },
  IR_108:  { enabled: true, opacity: 0.75 },
})

// ---- Timeline state ----
const timestamps     = ref([])
const frameIndex     = ref(0)
const radarStatuses  = ref({})  // { "YYYY-MM-DDTHH:MM": ["SITE1", ...] }
const isPlaying    = ref(false)
const isLoading    = ref(false)
const isLoaded     = ref(false)
const loadProgress = ref({ loaded: 0, total: 0 })
const productStats = ref({})
const showMissingFor = ref(null)
const loadError    = ref('')    // last error message, shown in UI

let playInterval    = null
let initialTimer    = null   // setTimeout — fires at the next 5-min clock mark
let pollTimer       = null   // setInterval — fires every 5 min after alignment
let countdownTimer  = null
let successTimer    = null   // setTimeout — clears productJustFoundFlag after 5s

// Per-timestamp async mutex: serialises concurrent onProductReady calls for the
// same timestamp. Without this, all 5 products arrive within milliseconds of each
// other. The first product hits `await appendProductFrames` and yields; the
// remaining 4 then each see isNewTs=true and all try to append a new slot,
// creating duplicate timeline entries and wrong frame counts.
const _tsLocks = new Map()

// Blocks onProductReady while a preserve=true sync is in flight.
// The sync fetches authoritative state from the API; any WS event that races
// with it would create duplicate timeline slots (onProductReady appends THEN
// the sync also appends the same timestamp). Events dropped here are safe to
// lose: the sync already reflects the latest server state, and the WS will
// push future events after the sync completes.
let _preserveSyncing = false

// True for 5 s after any product_ready event arrives (drives status dot flash).
const productJustFoundFlag = ref(false)

// Products that haven't yet delivered their file for the latest new timestamp.
// Shown as a spinner next to the product name until the WS event arrives.
const pendingProducts = reactive({})

// Per-product "just found" state: true for 5s after new data lands for that product.
// Used to show a ✓ next to the product name instead of the spinner.
const productJustFound = reactive({})
const productJustFoundTimers = {}
function markProductFound(product) {
  productJustFound[product] = true
  if (productJustFoundTimers[product]) clearTimeout(productJustFoundTimers[product])
  productJustFoundTimers[product] = setTimeout(() => {
    delete productJustFound[product]
  }, 5000)
}


// ---- Computed ----
const radarProducts = computed(() => configStore.radarProducts)

const visibleProducts = computed(() =>
  isLoaded.value ? productOrder.value.filter(p => layerConfig.value[p].enabled) : []
)

// Timezone-aware formatter — switches between Europe/Rome and UTC per settings
const displayTz = computed(() =>
  settings.timeZone === 'utc' ? 'UTC' : 'Europe/Rome'
)

const tsFormatter = computed(() => new Intl.DateTimeFormat('it-IT', {
  timeZone: displayTz.value,
  day: '2-digit', month: '2-digit', year: 'numeric',
  hour: '2-digit', minute: '2-digit', hour12: false,
}))

const currentTimestampDisplay = computed(() => {
  if (!timestamps.value.length) return '--/--/---- - --:--'
  const ts = timestamps.value[frameIndex.value]
  if (!ts) return '--/--/---- - --:--'
  const dt = new Date(ts + 'Z')
  const parts = tsFormatter.value.formatToParts(dt)
  const get = type => parts.find(p => p.type === type)?.value ?? '00'
  const suffix = settings.timeZone === 'utc' ? ' UTC' : ''
  return `${get('day')}/${get('month')}/${get('year')} - ${get('hour')}:${get('minute')}${suffix}`
})

// Slider tick labels — 5 evenly-spaced points (including first and last).
// On mobile only 3 are visible (first, middle, last) via the hideOnMobile flag,
// so the labels never overlap on a narrow screen but desktop gets the full set.
const hourTicks = computed(() => {
  const n = timestamps.value.length
  if (n < 2) return []
  const fmt = new Intl.DateTimeFormat('it-IT', {
    timeZone: displayTz.value,
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
  const COUNT = 5
  const ticks = []
  const seenIdx = new Set()
  for (let k = 0; k < COUNT; k++) {
    const i = Math.round((k / (COUNT - 1)) * (n - 1))
    if (seenIdx.has(i)) continue   // skip duplicate when n < COUNT
    seenIdx.add(i)
    const dt = new Date(timestamps.value[i] + 'Z')
    ticks.push({
      label: fmt.format(dt),
      pct: (i / (n - 1)) * 100,
      hideOnMobile: k === 1 || k === 3,
    })
  }
  return ticks
})

// Missing timestamps filtered to only those committed to the timeline.
// Raw productStats.missingTs can include future timestamps that are in the fresh
// API range but haven't been committed yet (e.g., 15:25 shown as missing at 15:31
// while still within the holdback window). Only show truly committed missing frames.
const timelineMissingTs = computed(() => {
  const inTimeline = new Set(timestamps.value)
  const result = {}
  for (const product of productOrder.value) {
    const stats = productStats.value[product]
    result[product] = stats ? stats.missingTs.filter(ts => inTimeline.has(ts)) : []
  }
  return result
})

const liveStatusText = computed(() => {
  if (isLoading.value) return 'Loading data…'
  if (!isLoaded.value) return 'Not loaded'
  if (productJustFoundFlag.value) return '✓ New data received'
  if (wsConnected.value) return 'Live'
  return 'Polling every 5 min'
})

const nextUpdateText = computed(() => {
  const s = nextUpdateSecs.value
  return s >= 60 ? `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s` : `${s}s`
})

// ---- Helpers ----

// Server delay: files land on disk ~6 min after their nominal timestamp.
//
// Two delay values are used depending on context:
//   STABLE (7 min): used for initial/background loads. floor(now-7min, 5min)
//     is always a boundary whose file is already on the server, so the
//     timeline loads cleanly with no missing frames.
//   FRESH (5 min): used by mark-aligned search windows. floor(now-5min, 5min)
//     at a mark M equals M-5min (the previous cycle's boundary), whose file
//     arrives ~1:30 after M and is caught by the 1s search loop.
//     Using exactly one cycle length (5 min) is important: it prevents the
//     range end from sliding forward mid-window.  With a smaller delay (e.g.
//     1 min), 1 minute into the window floor(M+1-1, 5)=M, pulling the NEXT
//     not-yet-arrived boundary into the range and causing spurious "missing".
const DATA_DELAY_STABLE_MS = 7 * 60 * 1000
const DATA_DELAY_FRESH_MS  = 5 * 60 * 1000

function computeRange(fresh = false) {
  // Data is stored in UTC. Use UTC throughout so file lookups match.
  // Subtract the delay then floor to the nearest 5-minute mark so that
  // the backend's expected timestamps (which step at 5-min intervals) align
  // with actual filenames (DD-MM-YYYY-HH-MM.hdf on 5-min boundaries).
  const delay = fresh ? DATA_DELAY_FRESH_MS : DATA_DELAY_STABLE_MS
  const endUtc = new Date(Date.now() - delay)
  endUtc.setUTCMinutes(Math.floor(endUtc.getUTCMinutes() / 5) * 5, 0, 0)
  const startUtc = new Date(endUtc - lookbackHours.value * 3600 * 1000)
  const fmt = dt => {
    const p = n => String(n).padStart(2, '0')
    return `${dt.getUTCFullYear()}-${p(dt.getUTCMonth()+1)}-${p(dt.getUTCDate())}T${p(dt.getUTCHours())}:${p(dt.getUTCMinutes())}`
  }
  return { start: fmt(startUtc), end: fmt(endUtc) }
}

// ---- Click-to-inspect popup ----
// Per-product unit (read from the radarProducts config so it follows YAML).
function unitFor(product) {
  return radarProducts.value?.[product]?.unit || ''
}

function fmtValue(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return 'N/A'
  // Compact: 2 decimals if |v|<10, otherwise 1 (cleaner for VMI/ETM ranges).
  return Math.abs(v) < 10 ? v.toFixed(2) : v.toFixed(1)
}

async function onMapClick(latlng) {
  if (!radarMap.value) return
  if (!timestamps.value.length) return
  const ts = timestamps.value[frameIndex.value]
  if (!ts) return

  // Sample only the products the user has currently enabled.
  const products = productOrder.value.filter(p => layerConfig.value[p].enabled)
  if (products.length === 0) return

  // Loading placeholder — popup opens immediately so the click feels responsive.
  const tzLabel = settings.timeZone === 'utc' ? 'UTC' : 'Local'
  const loadingHtml = `
    <div class="pi-header">${ts.replace('T', ' ')} (UTC)</div>
    <div class="pi-row"><span class="pi-label">Loading…</span></div>
  `
  radarMap.value.showPopup(latlng, loadingHtml)

  try {
    const data = await api.samplePixel({
      lat: latlng.lat,
      lon: latlng.lng,
      timestamp: ts,
      products,
    })

    let body
    if (!data.in_bounds) {
      body = `<div class="pi-row"><span class="pi-label">Outside radar grid</span></div>`
    } else {
      const rows = products.map(p => {
        const v = data.values?.[p]
        const u = unitFor(p)
        return `
          <div class="pi-row">
            <span class="pi-label">${SHORT_NAMES[p] || p}</span>
            <span class="pi-value">${fmtValue(v)}${v != null && u ? ' ' + u : ''}</span>
          </div>`
      }).join('')
      body = `
        <div class="pi-row" style="margin-bottom:4px;">
          <span class="pi-label">pixel</span>
          <span class="pi-value">x ${data.x}, y ${data.y}</span>
        </div>
        ${rows}`
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

    const html = `
      <div class="pi-header">${ts.replace('T', ' ')} (${tzLabel})</div>
      ${body}
    `
    radarMap.value.showPopup(latlng, html)
  } catch (e) {
    radarMap.value.showPopup(
      latlng,
      `<div class="pi-row"><span class="pi-label">Error: ${e.message || e}</span></div>`,
    )
  }
}

// ---- Frame navigation ----
function goToFrame(idx) {
  frameIndex.value = idx
  if (!radarMap.value || !isLoaded.value) return
  const opacities = {}
  for (const product of productOrder.value) {
    opacities[product] = layerConfig.value[product].enabled
      ? layerConfig.value[product].opacity
      : 0
  }
  radarMap.value.showAllAtFrame(idx, opacities)
  const ts = timestamps.value[idx]
  radarMap.value.updateRadarStatus(ts ? (radarStatuses.value[ts] ?? null) : null)
}

function goToLatest() {
  if (timestamps.value.length > 0) goToFrame(timestamps.value.length - 1)
}

// Re-render current frame when layer enabled/opacity changes
watch(layerConfig, () => {
  if (!isLoaded.value || timestamps.value.length === 0) return
  goToFrame(frameIndex.value)
}, { deep: true })

// ---- Core load ----
async function loadData({ preserve = false } = {}) {
  const { start, end } = computeRange()
  loadError.value = ''

  if (!preserve) {
    // Full reset: show spinner, block WS events, wipe the map.
    isLoading.value = true
    isLoaded.value = false
    timestamps.value = []
    productStats.value = {}
    showMissingFor.value = null
    loadProgress.value = { loaded: 0, total: 0 }
    radarMap.value?.clearAllProducts()
    Object.keys(pendingProducts).forEach(k => delete pendingProducts[k])
    _tsLocks.clear()
  }
  // preserve=true: no spinner, no reset — WS events keep flowing during the refresh.

  try {
    const [results, statusResult] = await Promise.all([
      Promise.all(
        productOrder.value.map(product =>
          api.explorerTimestamps(start, end, product, 'realtime-load').catch((err) => {
            console.error(`[LiveView] explorerTimestamps failed for ${product}:`, err)
            loadError.value = `API error (${product}): ${err.message}`
            return { timestamps: [], missing: [], total_expected: 0, total_found: 0 }
          })
        )
      ),
      api.radarStatusRange(start, end).catch(() => ({ statuses: {} })),
    ])
    radarStatuses.value = statusResult.statuses

    // Normalise to 16 chars ("YYYY-MM-DDTHH:MM") so initial-load timestamps
    // and WS-pushed timestamps (also normalised in onProductReady) are identical.
    // The backend returns isoformat() with seconds ("T16:05:00"); without this
    // slice both formats exist in the timeline for the same instant, causing
    // the WS slot to sort before the initial-load slot and breaking resolveProductFrame.
    const norm = ts => (ts || '').slice(0, 16)

    const tsSet = new Set()
    results.forEach(r => {
      r.timestamps.forEach(ts => tsSet.add(norm(ts)))
      r.missing.forEach(ts => tsSet.add(norm(ts)))
    })
    const sortedTs = Array.from(tsSet).sort()

    if (sortedTs.length === 0) {
      if (!loadError.value)
        loadError.value = `No files found for ${start} → ${end} (UTC). Check backend logs.`
      return
    }
    loadError.value = ''

    if (preserve) {
      // ── Background refresh: trim old frames, append missed ones, update stats ──
      // Block WS events for the duration of this sync. onProductReady running
      // concurrently would race with the currentSet snapshot below, causing the
      // same timestamp to be appended twice (once by WS, once here).
      _preserveSyncing = true
      try {

      // Trim timestamps that have scrolled out of the lookback window (always oldest-first).
      const trimCount = timestamps.value.filter(ts => ts < start).length
      if (trimCount > 0) {
        productOrder.value.forEach(p => radarMap.value?.trimProductFrames(p, trimCount))
        timestamps.value = timestamps.value.slice(trimCount)
        frameIndex.value = Math.max(0, frameIndex.value - trimCount)
      }

      // Append any timestamps the WS may have missed (not yet in the local timeline).
      const currentSet = new Set(timestamps.value)
      const newTs = sortedTs.filter(ts => !currentSet.has(ts))

      // Refresh stats from the authoritative server response.
      results.forEach((r, i) => {
        productStats.value[productOrder.value[i]] = {
          found:      r.total_found,
          expected:   r.total_expected,
          missingTs:  r.missing.map(norm),
          missingSet: new Set(r.missing.map(norm)),
        }
      })

      if (newTs.length > 0) {
        await Promise.all(productOrder.value.map(async (product) => {
          const stats = productStats.value[product]
          const urls  = newTs.map(ts =>
            stats?.missingSet?.has(ts) ? null : api.explorerOverlayUrl(product, ts)
          )
          await radarMap.value?.appendProductFrames(product, urls)
        }))
        timestamps.value = [...timestamps.value, ...newTs].sort()
        radarMap.value?.setProductOrder(productOrder.value)
      }

      // Merge any new status entries (already have the full statusResult from above)
      Object.assign(radarStatuses.value, statusResult.statuses)

      if (followLive.value) goToFrame(timestamps.value.length - 1)
      else goToFrame(Math.min(frameIndex.value, timestamps.value.length - 1))

      } finally {
        _preserveSyncing = false
      }
      return
    }

    // ── Full load continued ───────────────────────────────────────────────────
    timestamps.value = sortedTs

    results.forEach((r, i) => {
      productStats.value[productOrder.value[i]] = {
        found:      r.total_found,
        expected:   r.total_expected,
        missingTs:  r.missing.map(norm),
        missingSet: new Set(r.missing.map(norm)),
      }
    })

    loadProgress.value = { loaded: 0, total: productOrder.value.length * sortedTs.length }

    radarMap.value?.clearAllProducts()
    await Promise.all(productOrder.value.map(async (product) => {
      const stats = productStats.value[product]
      const urls  = sortedTs.map(ts =>
        stats?.missingSet?.has(ts) ? null : api.explorerOverlayUrl(product, ts)
      )
      await radarMap.value?.loadProductFrames(product, urls, layerConfig.value[product].opacity)
      loadProgress.value.loaded += sortedTs.length
    }))

    isLoaded.value = true

    // Apply stacking order BEFORE showing the first frame so that layers are
    // already in the correct z-order when goToFrame triggers CSS transitions.
    radarMap.value?.setProductOrder(productOrder.value)

    if (followLive.value) {
      goToFrame(sortedTs.length - 1)
    } else {
      goToFrame(0)
    }

  } catch (e) {
    console.error('LiveView: failed to load data:', e)
  } finally {
    isLoading.value = false
  }
}

// ---- Lookback change ----
async function setLookback(hours) {
  if (isLoading.value) return
  // Allow re-clicking the same button if the previous load failed (isLoaded=false)
  if (hours === lookbackHours.value && isLoaded.value) return
  lookbackHours.value = hours
  stopAnimation()
  await loadData({ preserve: false })
}

// Return the most recent non-missing overlay URL for a product.
// Used as a closest-frame placeholder when a new timestamp arrives for SRI
// but other products haven't landed yet — avoids a blank layer.
function fallbackUrlFor(product) {
  const stats = productStats.value[product]
  for (let i = timestamps.value.length - 1; i >= 0; i--) {
    const ts = timestamps.value[i]
    if (!stats?.missingSet?.has(ts)) {
      return api.explorerOverlayUrl(product, ts)
    }
  }
  return null
}

// Returns milliseconds until the next 5-minute clock boundary.
// Used to clock-align the fallback loadData() so it fires when new data is expected.
function msUntilNextFiveMinMark() {
  const ms = Date.now() % POLL_MS
  return POLL_MS - ms
}

// ---- WS-driven new-data handler ----
// Called by useRealtimeWs when the backend broadcasts product_ready.
// Replaces the polling loop: each product pushes itself the instant its file lands.
async function onProductReady(product, rawTimestamp) {
  // Normalize: strip seconds so WS timestamps ("2026-05-08T16:25:00") match
  // the initial-load format ("2026-05-08T16:25") used throughout the timeline.
  const timestamp = (rawTimestamp || '').slice(0, 16)
  if (!isLoaded.value || isLoading.value || _preserveSyncing || !timestamp) return

  // Acquire per-timestamp lock so concurrent arrivals are processed serially.
  // Each waiter receives the exclusive right to run only after the previous one
  // has finished and updated timestamps.value — ensuring isNewTs is correct.
  const prev = _tsLocks.get(timestamp) ?? Promise.resolve()
  let unlock
  _tsLocks.set(timestamp, new Promise(r => { unlock = r }))
  await prev

  // Re-check guards: state may have changed while waiting for the lock.
  if (!isLoaded.value || isLoading.value || _preserveSyncing) { unlock(); return }

  try {

  const isNewTs = !timestamps.value.includes(timestamp)

  if (isNewTs) {
    // First product for this timestamp: commit it to the timeline immediately.
    // Other products that haven't arrived yet get a closest-frame fallback so
    // their layer doesn't go blank; onProductReady will replace them when ready.
    const urls = productOrder.value.map(p =>
      p === product ? api.explorerOverlayUrl(p, timestamp) : fallbackUrlFor(p)
    )
    await Promise.all(productOrder.value.map((p, i) =>
      radarMap.value?.appendProductFrames(p, [urls[i]])
    ))
    timestamps.value = [...timestamps.value, timestamp].sort()
    radarMap.value?.setProductOrder(productOrder.value)

    // Mark every other product as pending (spinner until their event arrives).
    productOrder.value.forEach(p => {
      if (p === product) return
      if (!productStats.value[p]) {
        productStats.value[p] = { found: 0, expected: 0, missingTs: [], missingSet: new Set() }
      }
      productStats.value[p].missingSet.add(timestamp)
      if (!productStats.value[p].missingTs.includes(timestamp))
        productStats.value[p].missingTs.push(timestamp)
      pendingProducts[p] = true
    })

    // Re-fetch radar status for the current range every time a new frame arrives.
    // Non-blocking: goToFrame runs immediately (markers briefly white), then the
    // fetch completes and repaints them. This is the reliable path — independent
    // of the SITES cron /notify → WS chain which may lag by seconds.
    const { start, end } = computeRange()
    api.radarStatusRange(start, end)
      .then(r => {
        if (r.statuses && Object.keys(r.statuses).length) {
          Object.assign(radarStatuses.value, r.statuses)
          const ts = timestamps.value[frameIndex.value]
          radarMap.value?.updateRadarStatus(ts ? (radarStatuses.value[ts] ?? null) : null)
        }
      })
      .catch(() => {})

    if (followLive.value) goToFrame(timestamps.value.length - 1)

  } else {
    // Late arrival: replace the fallback URL with the real image.
    const idx = timestamps.value.indexOf(timestamp)
    if (idx !== -1) {
      const ok = await (radarMap.value?.resolveProductFrame(
        product, idx, api.explorerOverlayUrl(product, timestamp)
      ) ?? Promise.resolve(false))
      if (ok) {
        radarMap.value?.setProductOrder(productOrder.value)
        goToFrame(frameIndex.value)
      }
    }
  }

  // Update productStats: this product is now present for this timestamp.
  if (!productStats.value[product]) {
    productStats.value[product] = { found: 0, expected: 0, missingTs: [], missingSet: new Set() }
  }
  const wasMissing = productStats.value[product].missingSet.has(timestamp)
  productStats.value[product].missingSet.delete(timestamp)
  productStats.value[product].missingTs =
    productStats.value[product].missingTs.filter(ts => ts !== timestamp)
  // Increment found for: (a) the first product announcing a new timestamp, or
  // (b) a late arrival resolving a frame that was shown as missing.
  if (isNewTs || wasMissing) {
    productStats.value[product].found = (productStats.value[product].found || 0) + 1
  }
  // When a new timestamp slot is created, every product gains one expected frame.
  if (isNewTs) {
    productOrder.value.forEach(p => {
      if (!productStats.value[p]) {
        productStats.value[p] = { found: 0, expected: 0, missingTs: [], missingSet: new Set() }
      }
      productStats.value[p].expected = (productStats.value[p].expected || 0) + 1
    })
  }

  delete pendingProducts[product]
  markProductFound(product)

  // Flash the status dot green for 5 s.
  productJustFoundFlag.value = true
  if (successTimer) clearTimeout(successTimer)
  successTimer = setTimeout(() => { productJustFoundFlag.value = false }, 5000)

  } finally {
    unlock()
  }
}

// ---- 5-min clock-aligned fallback poll ----
// When the WS is healthy this fires every 5 min to trim old frames from the
// front of the sliding window and catch any edge cases the WS might have missed.
// When the WS is offline this is the only mechanism keeping the timeline fresh.
function startPolling() {
  stopPolling()
  const delay = msUntilNextFiveMinMark()
  nextUpdateSecs.value = Math.round(delay / 1000)

  initialTimer = setTimeout(async () => {
    initialTimer = null
    await loadData({ preserve: true })
    nextUpdateSecs.value = POLL_MS / 1000
    pollTimer = setInterval(async () => {
      await loadData({ preserve: true })
      nextUpdateSecs.value = POLL_MS / 1000
    }, POLL_MS)
  }, delay)

  countdownTimer = setInterval(() => {
    if (nextUpdateSecs.value > 0) nextUpdateSecs.value--
  }, 1000)
}

function stopPolling() {
  if (initialTimer)   { clearTimeout(initialTimer);   initialTimer   = null }
  if (pollTimer)      { clearInterval(pollTimer);      pollTimer      = null }
  if (countdownTimer) { clearInterval(countdownTimer); countdownTimer = null }
}

// ---- Slider + animation ----
function onSliderInput(e) {
  goToFrame(Number(e.target.value))
}

function togglePlay() {
  if (isPlaying.value) stopAnimation(); else startAnimation()
}
function startAnimation() {
  if (!timestamps.value.length) return
  isPlaying.value = true
  playInterval = setInterval(() => {
    goToFrame((frameIndex.value + 1) % timestamps.value.length)
  }, 1000 / (playSpeed.value * 3))
}
function stopAnimation() {
  isPlaying.value = false
  if (playInterval) { clearInterval(playInterval); playInterval = null }
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

// Apply z-order whenever the layer order changes
watch(productOrder, () => {
  if (radarMap.value && isLoaded.value) {
    radarMap.value.setProductOrder(productOrder.value)
    goToFrame(frameIndex.value)
  }
})

// ---- Missing frames ----
function toggleMissing(product) {
  showMissingFor.value = showMissingFor.value === product ? null : product
}
function formatMissingTs(isoTs) {
  const dt  = new Date(isoTs)
  const pad = n => String(n).padStart(2, '0')
  return `${pad(dt.getDate())}-${pad(dt.getMonth()+1)}-${dt.getFullYear()}-${pad(dt.getHours())}-${pad(dt.getMinutes())}.hdf`
}

// ---- Motion field layer (AMV / LK) ----
const currentTs = computed(() => (timestamps.value[frameIndex.value] ?? '').slice(0, 16))
const { motionMode, motionLoading, activeMotionTs, lkDisplayMode, lkArrowOpacity,
        updateMotionLayer, fetchTimestamps, prefetchData, sampleMotionAt } =
  useMotionLayer(radarMap, currentTs, lookbackHours)

// ---- WebSocket ----
// product_ready events drive onProductReady above (primary update path).
// state_update is still received for ML model status info (used by NowcastingView).
const { connected: wsConnected } = useRealtimeWs({ onProductReady })

// Listen for radar_status_updated from the cron script (fires after each SITES file download).
// Re-fetches the full current range so new timestamps get their status immediately.
useRadarStatusWs({
  onRadarStatusUpdated: async () => {
    console.log('[RadarStatusWs] onRadarStatusUpdated fired, isLoaded:', isLoaded.value)
    if (!isLoaded.value) return
    const { start, end } = computeRange()
    console.log('[RadarStatusWs] fetching range', start, '→', end)
    const result = await api.radarStatusRange(start, end).catch((e) => { console.error('[RadarStatusWs] fetch error', e); return { statuses: {} } })
    console.log('[RadarStatusWs] statuses keys:', Object.keys(result.statuses ?? {}))
    Object.assign(radarStatuses.value, result.statuses)
    const ts = timestamps.value[frameIndex.value]
    console.log('[RadarStatusWs] current ts:', ts, 'status:', radarStatuses.value[ts])
    radarMap.value?.updateRadarStatus(ts ? (radarStatuses.value[ts] ?? null) : null)
  },
})

// ---- Lifecycle ----
onMounted(async () => {
  // Wait for the browser to layout and size the Leaflet container before loading.
  // Without this, Leaflet may have zero-dimension tiles on first paint.
  await nextTick()
  await loadData({ preserve: false })
  startPolling()
  fetchTimestamps('amv')  // pre-warm AMV timestamps; LK loaded on demand when toggled
})

// keep-alive hooks: fired when navigating away/back without destroying the component.
onDeactivated(() => {
  stopAnimation()
  stopPolling()
})

onActivated(async () => {
  // Leaflet tiles go stale when the container is hidden; invalidateSize forces
  // a re-render so the map looks correct immediately on return.
  await nextTick()
  radarMap.value?.invalidateSize()
  // Re-draw the current frame (layers may have lost visibility while hidden).
  if (isLoaded.value) goToFrame(frameIndex.value)

  // Sync timeline on return: trim excess frames that accumulated via WS while the
  // tab was hidden and reset stats to authoritative server values.
  // If data is very stale (> 5 min behind) do a full reload; otherwise a
  // preserve=true sync is enough and keeps the current frame position.
  if (isLoaded.value && timestamps.value.length > 0) {
    const latestCommitted = new Date(timestamps.value[timestamps.value.length - 1])
    const expectedEnd     = new Date(computeRange().end)
    if (expectedEnd - latestCommitted >= POLL_MS) {
      await loadData({ preserve: false })
    } else {
      await loadData({ preserve: true })
    }
  }

  // Resume the 5-min fallback clock.
  startPolling()
})

onUnmounted(() => {
  stopAnimation()
  stopPolling()
  if (successTimer) { clearTimeout(successTimer); successTimer = null }
  Object.values(productJustFoundTimers).forEach(t => clearTimeout(t))
})
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

/* Colorbar stack — plain CSS to bypass any Tailwind v4 arbitrary-value
   parsing issue with calc(...env(...)). Mobile sits just above the slider. */
.colorbar-stack {
  top: 64px;
  bottom: calc(70px + env(safe-area-inset-bottom));
}
@media (min-width: 640px) {
  .colorbar-stack {
    top: auto;
    bottom: 80px;
    max-height: calc(100vh - 18rem);
  }
}
</style>