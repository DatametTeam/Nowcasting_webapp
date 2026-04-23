<!--
  LiveView.vue — Real-time multi-product radar monitor.

  Shows all 4 radar products (SRI, VMI, ETM, VIL) for a rolling lookback window
  (default 1h, up to 12h). Auto-loads on mount and polls every 5 minutes for
  new data. "Follow Live" toggle auto-jumps to the latest frame on new data;
  when off, the user's current frame position is preserved across updates.
-->
<template>
  <div class="h-[calc(100vh-3.5rem)] flex overflow-hidden">

    <!-- ================================================================ -->
    <!-- LEFT: Map area                                                    -->
    <!-- ================================================================ -->
    <div class="flex-1 flex flex-col relative min-w-0">
      <RadarMap ref="radarMap" class="flex-1" />

      <!-- Mobile sidebar toggle -->
      <button
        v-if="!sidebarOpen"
        @click="sidebarOpen = true"
        class="absolute top-3 right-3 z-[1001] lg:hidden
               w-10 h-10 flex items-center justify-center rounded-full
               bg-white shadow-lg border border-gray-200 text-gray-600"
      >
        <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path d="M4 6h16M4 12h16M4 18h16" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
      </button>

      <!-- Stacked colorbars — bottom right, above timeline -->
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
      <!-- BOTTOM: Timeline controls                                     -->
      <!-- ============================================================ -->
      <div
        class="absolute bottom-0 left-0 right-0 z-[1000]
               bg-gradient-to-t from-black/80 via-black/60 to-transparent
               px-3 sm:px-6 pt-10 pb-4"
        :class="{ 'pointer-events-none opacity-40': !isLoaded }"
      >
        <!-- Top row: layer names | datetime | speed -->
        <div class="flex items-center justify-between text-white mb-2">
          <div class="text-xs font-medium text-gray-300 hidden sm:block truncate max-w-[160px]">
            {{ visibleProducts.map(p => SHORT_NAMES[p]).join(' + ') || '—' }}
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

        <!-- Slider row -->
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
    <!-- RIGHT: Sidebar                                                    -->
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
          <h2 class="text-white font-bold text-base">Real Time</h2>
          <p class="text-gray-400 text-xs mt-0.5">Live multi-product radar</p>
        </div>

        <!-- Live status card -->
        <div class="bg-gray-800 rounded-lg p-3 space-y-2.5">
          <!-- Status indicator -->
          <div class="flex items-center gap-2">
            <div
              class="w-2 h-2 rounded-full flex-shrink-0"
              :class="(isUpdating || isSearching) ? 'bg-yellow-400 animate-pulse' : isLoaded ? 'bg-green-400' : 'bg-gray-500'"
            />
            <span class="text-xs text-gray-300">{{ liveStatusText }}</span>
            <span v-if="isLoaded && !isUpdating" class="ml-auto text-[10px] text-gray-500 tabular-nums">
              next: {{ nextUpdateText }}
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
              <!-- Spinner while this product's latest frame is still being fetched.
                   During phase 1 (searchWindowTs empty) show for all products;
                   in phase 2 show only for products still pending. -->
              <svg
                v-if="isSearching && (!searchWindowTs.length || searchingProducts.has(product))"
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
              <!-- Hide "N missing" while actively searching for this product's data -->
              <button
                v-if="productStats[product].missingTs.length > 0 && !(isSearching && (!searchWindowTs.length || searchingProducts.has(product)))"
                @click="toggleMissing(product)"
                class="text-amber-400 hover:text-amber-300 ml-1.5 underline underline-offset-2"
              >
                {{ productStats[product].missingTs.length }} missing
                {{ showMissingFor === product ? '▲' : '▼' }}
              </button>
            </div>

            <div
              v-if="showMissingFor === product && productStats[product]?.missingTs.length"
              class="mt-1 space-y-0.5 max-h-36 overflow-y-auto rounded bg-black/30 px-2 py-1.5"
            >
              <div
                v-for="ts in productStats[product].missingTs"
                :key="ts"
                class="font-mono text-[10px] text-amber-300/80"
              >{{ formatMissingTs(ts) }}</div>
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
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, onActivated, onDeactivated, nextTick } from 'vue'
import { useConfigStore } from '../stores/config.js'
import api from '../api.js'
import RadarMap from '../components/RadarMap.vue'
import ColorBar from '../components/ColorBar.vue'

const configStore = useConfigStore()
const radarMap = ref(null)
const sidebarOpen = ref(false)

const SHORT_NAMES = { SRI_adj: 'SRI', VMI: 'VMI', ETM: 'ETM', VIL: 'VIL', IR_108: 'IR' }
// Ordered top-to-bottom on the map (index 0 = topmost layer). IR_108 is last = bottommost.
const productOrder = ref(['SRI_adj', 'VMI', 'ETM', 'VIL', 'IR_108'])
const lookbackOptions = [1, 2, 4, 6, 12]
const POLL_MS = 5 * 60 * 1000  // 5-minute polling

// ---- Speed ----
const speeds = [0.5, 1, 2, 4]
const playSpeed = ref(1)
function cycleSpeed() {
  const idx = speeds.indexOf(playSpeed.value)
  playSpeed.value = speeds[(idx + 1) % speeds.length]
  if (isPlaying.value) { stopAnimation(); startAnimation() }
}

// ---- Live state ----
const lookbackHours = ref(1)
const followLive    = ref(true)
const isUpdating    = ref(false)    // true during a background poll reload
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
const timestamps   = ref([])
const frameIndex   = ref(0)
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
let searchTimer     = null   // setTimeout — drives the sequential search loop

// Poll every 1s for up to 3 min past each 5-min mark. A listdir on the server
// is cheap and data typically lands ~1:30 past the mark, so 1s gets us the
// file within a second of arrival.
// Hold-back window (first 10s): don't commit a frame with any missing products
// — wait until all products have arrived or the window expires.
// After 10s: commit whatever is available; late products get resolved in place
// via resolveProductFrame when their files eventually arrive.
const SEARCH_HOLDBACK_MS = 10  * 1000       // no-commit window at start
const SEARCH_INTERVAL    = 1   * 1000       // 1s throughout
const SEARCH_MAX_MS      = 3   * 60 * 1000  // raise error after 3 minutes

const isSearching    = ref(false)  // true while the search window is active
let   searchStart    = 0           // Date.now() when the current search began
let   searchFoundAny = false       // true if any data was committed in this window

// Timestamps committed to the timeline in this search window.
// Used to detect when all newly-added frames are fully resolved so we can
// stop early instead of running the full 3 minutes.
const searchWindowTs = ref([])

// ---- Computed ----
const radarProducts = computed(() => configStore.radarProducts)

const visibleProducts = computed(() =>
  isLoaded.value ? productOrder.value.filter(p => layerConfig.value[p].enabled) : []
)

// Rome timezone formatter — data timestamps are UTC, display in local (Rome) time
const romeFormatter = new Intl.DateTimeFormat('it-IT', {
  timeZone: 'Europe/Rome',
  day: '2-digit', month: '2-digit', year: 'numeric',
  hour: '2-digit', minute: '2-digit', hour12: false,
})

const currentTimestampDisplay = computed(() => {
  if (!timestamps.value.length) return '--/--/---- - --:--'
  const ts = timestamps.value[frameIndex.value]
  if (!ts) return '--/--/---- - --:--'
  // Append 'Z' so the browser parses the backend's naive UTC string as UTC,
  // then Intl converts to Rome time (UTC+1 winter, UTC+2 summer).
  const dt = new Date(ts + 'Z')
  const parts = romeFormatter.formatToParts(dt)
  const get = type => parts.find(p => p.type === type)?.value ?? '00'
  return `${get('day')}/${get('month')}/${get('year')} - ${get('hour')}:${get('minute')}`
})

const hourTicks = computed(() => {
  if (timestamps.value.length < 2) return []
  const ticks = []
  const seen = new Set()
  timestamps.value.forEach((ts, i) => {
    // Parse as UTC, display in Rome time
    const dt = new Date(ts + 'Z')
    const romeMinute = Number(new Intl.DateTimeFormat('it-IT', { timeZone: 'Europe/Rome', minute: 'numeric' }).format(dt))
    const romeHour   = new Intl.DateTimeFormat('it-IT', { timeZone: 'Europe/Rome', hour: '2-digit', hour12: false }).format(dt)
    if (romeMinute === 0) {
      const label = `${romeHour}:00`
      if (!seen.has(label)) {
        seen.add(label)
        ticks.push({ label, pct: (i / (timestamps.value.length - 1)) * 100 })
      }
    }
  })
  return ticks
})

// Products that still have unresolved frames in the current search window.
// Used to show a per-product spinner and hide the "N missing" count while polling.
const searchingProducts = computed(() => {
  if (!isSearching.value || searchWindowTs.value.length === 0) return new Set()
  const pending = new Set()
  for (const product of productOrder.value) {
    const missing = productStats.value[product]?.missingSet
    if (!missing) continue
    if (searchWindowTs.value.some(ts => missing.has(ts))) pending.add(product)
  }
  return pending
})

const liveStatusText = computed(() => {
  if (isLoading.value && !isUpdating.value) return 'Loading data…'
  if (isUpdating.value) return 'Checking for new data…'
  if (isSearching.value) return 'Waiting for new data…'
  if (!isLoaded.value) return 'Not loaded'
  return 'Live'
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

  isLoading.value = true
  loadError.value = ''

  if (!preserve) {
    // Full reset
    isLoaded.value = false
    timestamps.value = []
    productStats.value = {}
    showMissingFor.value = null
    loadProgress.value = { loaded: 0, total: 0 }
    radarMap.value?.clearAllProducts()
  }

  try {
    const results = await Promise.all(
      productOrder.value.map(product =>
        api.explorerTimestamps(start, end, product).catch((err) => {
          console.error(`[LiveView] explorerTimestamps failed for ${product}:`, err)
          loadError.value = `API error (${product}): ${err.message}`
          return { timestamps: [], missing: [], total_expected: 0, total_found: 0 }
        })
      )
    )

    const tsSet = new Set()
    results.forEach(r => {
      r.timestamps.forEach(ts => tsSet.add(ts))
      r.missing.forEach(ts => tsSet.add(ts))
    })
    const sortedTs = Array.from(tsSet).sort()

    if (sortedTs.length === 0) {
      if (!loadError.value)
        loadError.value = `No files found for ${start} → ${end} (UTC). Check backend logs.`
      return
    }
    loadError.value = ''

    // Decide where to land after reload
    const prevLen      = timestamps.value.length
    const prevFraction = prevLen > 1 ? frameIndex.value / (prevLen - 1) : 1

    timestamps.value = sortedTs

    results.forEach((r, i) => {
      productStats.value[productOrder.value[i]] = {
        found:      r.total_found,
        expected:   r.total_expected,
        missingTs:  r.missing,
        missingSet: new Set(r.missing),
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

    if (followLive.value) {
      goToFrame(sortedTs.length - 1)
    } else if (preserve && prevLen > 0) {
      // Keep approximate fractional position through the timeline
      const targetIdx = Math.min(
        Math.round(prevFraction * (sortedTs.length - 1)),
        sortedTs.length - 1
      )
      goToFrame(targetIdx)
    } else {
      goToFrame(0)
    }

    // Apply stacking order: IR_108 is last in productOrder → bottommost on map
    radarMap.value?.setProductOrder(productOrder.value)

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

// ---- Polling: sliding window ----
// Each poll: append new frames at the end, drop old frames from the front.
// Typically 1 new frame per product (4 PNG requests total) — stays fast.
// Returns true if actual new image data was loaded.
// Returns false to keep the search window alive (file not ready yet).
async function pollForNewData() {
  if (isLoading.value || isUpdating.value) return false
  isUpdating.value = true
  try {
    const { start, end } = computeRange(true)  // fresh: probe for just-arrived data

    const results = await Promise.all(
      productOrder.value.map(product =>
        api.explorerTimestamps(start, end, product).catch(() => ({
          timestamps: [], missing: [], total_expected: 0, total_found: 0,
        }))
      )
    )

    const tsSet = new Set()
    results.forEach(r => {
      r.timestamps.forEach(ts => tsSet.add(ts))
      r.missing.forEach(ts => tsSet.add(ts))
    })
    const newRangeTs = Array.from(tsSet).sort()
    if (newRangeTs.length === 0) return false

    if (!isLoaded.value) {
      await loadData({ preserve: false })
      return true
    }

    const newMissingAll = new Set()
    results.forEach(r => r.missing.forEach(ts => newMissingAll.add(ts)))

    const newRangeSet  = new Set(newRangeTs)
    const currentSet   = new Set(timestamps.value)
    const addedTs      = newRangeTs.filter(ts => !currentSet.has(ts))
    const droppedCount = timestamps.value.filter(ts => !newRangeSet.has(ts)).length

    // ---- Resolved frames: previously missing, now found, already in the timeline ----
    // Only care about in-timeline null slots (added after 90s timeout) — these need
    // resolveProductFrame to patch the slot in place.
    // Timestamps held back by the 90s delay are NOT in timestamps.value, so they
    // appear in addedTs as new entries and are handled by the addedFoundTs path below.
    // Compute BEFORE updating productStats (need to compare old vs new missing sets).
    const resolvedInTimeline = []   // { product, ts, idx }
    productOrder.value.forEach((product, i) => {
      const prevMissing = productStats.value[product]?.missingSet
      if (!prevMissing || prevMissing.size === 0) return
      const newMissingSet = new Set(results[i].missing)
      for (const ts of prevMissing) {
        if (!newMissingSet.has(ts) && currentSet.has(ts)) {
          const idx = timestamps.value.indexOf(ts)
          if (idx !== -1) resolvedInTimeline.push({ product, ts, idx })
        }
      }
    })
    const hasResolved = resolvedInTimeline.length > 0

    // ---- New timestamps: split into found vs still-missing ----
    const addedFoundTs   = addedTs.filter(ts => !newMissingAll.has(ts))
    const addedMissingTs = addedTs.filter(ts =>  newMissingAll.has(ts))

    // Hold-back: during the first SEARCH_HOLDBACK_MS don't commit any empty
    // frames — give all products a chance to arrive before showing blank slots.
    // After that, commit whatever is available; late products get resolved in
    // place via resolveProductFrame when their files eventually arrive.
    const elapsed      = searchStart > 0 ? Date.now() - searchStart : Infinity
    const delayMissing = addedFoundTs.length === 0 && addedMissingTs.length > 0
                         && !hasResolved && elapsed < SEARCH_HOLDBACK_MS

    if (addedTs.length === 0 && droppedCount === 0 && !hasResolved) return false

    // ---- Update productStats (after resolved computation) ----
    results.forEach((r, i) => {
      productStats.value[productOrder.value[i]] = {
        found:      r.total_found,
        expected:   r.total_expected,
        missingTs:  r.missing,
        missingSet: new Set(r.missing),
      }
    })

    // ---- Timestamps we're committing to the timeline this tick ----
    // Must be sorted chronologically so RadarMap layer indices match timestamps.value.
    // Without sort, found-first concat ([16:05, 16:10, 16:00]) would misalign layers.
    //
    // Never commit a timestamp that has ZERO products loaded for it — jumping
    // to an all-null frame hides every layer and leaves the map blank.
    // A timestamp is safe to commit once at least one product has real data.
    const hasAnyProduct = ts => productOrder.value.some(
      p => !productStats.value[p]?.missingSet?.has(ts)
    )
    const toAppend = (delayMissing
      ? addedFoundTs
      : [...addedFoundTs, ...addedMissingTs.filter(hasAnyProduct)]
    ).sort()

    // ---- Fix null slots already in the timeline ----
    if (resolvedInTimeline.length > 0) {
      await Promise.all(resolvedInTimeline.map(({ product, ts, idx }) =>
        radarMap.value?.resolveProductFrame(product, idx, api.explorerOverlayUrl(product, ts))
      ))
      // Remove fully-resolved timestamps from the search window tracker
      const resolvedTsSet = new Set(resolvedInTimeline.map(r => r.ts))
      const stillPending = productOrder.value.some(p =>
        [...resolvedTsSet].some(ts => productStats.value[p]?.missingSet?.has(ts))
      )
      if (!stillPending) {
        searchWindowTs.value = searchWindowTs.value.filter(ts => !resolvedTsSet.has(ts))
        // All search-window timestamps are now fully resolved (including late IR).
        // The runSearch early-stop won't fire because searchWindowTs is now empty,
        // so stop here directly to avoid running to the 3-minute timeout.
        if (searchWindowTs.value.length === 0 && isSearching.value) {
          stopSearching()
        }
      }
    }

    // ---- Append new timestamps to RadarMap ----
    if (toAppend.length > 0) {
      await Promise.all(productOrder.value.map(async (product) => {
        const stats = productStats.value[product]
        const urls  = toAppend.map(ts =>
          stats?.missingSet?.has(ts) ? null : api.explorerOverlayUrl(product, ts)
        )
        await radarMap.value?.appendProductFrames(product, urls)
      }))
    }

    // ---- Drop old frames from the front ----
    if (droppedCount > 0) {
      for (const product of productOrder.value) {
        radarMap.value?.trimProductFrames(product, droppedCount)
      }
    }

    // ---- Update timeline and frame pointer ----
    const prevFrameIndex = frameIndex.value
    const retained = timestamps.value.filter(ts => newRangeSet.has(ts))
    timestamps.value = [...retained, ...toAppend].sort()
    const adjustedIndex = Math.max(0, prevFrameIndex - droppedCount)

    if (followLive.value) {
      goToFrame(timestamps.value.length - 1)
    } else {
      goToFrame(Math.min(adjustedIndex, timestamps.value.length - 1))
    }

    radarMap.value?.setProductOrder(productOrder.value)

    // Refresh current frame so resolved images become visible
    if (hasResolved) goToFrame(frameIndex.value)

    // Track newly-committed timestamps so runSearch can stop early when resolved
    if (toAppend.length > 0) {
      const merged = new Set([...searchWindowTs.value, ...toAppend])
      searchWindowTs.value = [...merged]
    }

    const dataFound = addedFoundTs.length > 0 || hasResolved
    if (dataFound) searchFoundAny = true
    return dataFound

  } catch (e) {
    console.error('LiveView poll error:', e)
    return false
  } finally {
    isUpdating.value = false
  }
}

// Returns milliseconds until the next 5-minute clock boundary (00:05, 00:10, ...).
// Aligning polls to clock marks ensures we check right when new files should arrive,
// instead of drifting relative to whenever the page was loaded.
function msUntilNextFiveMinMark() {
  const ms = Date.now() % POLL_MS
  return POLL_MS - ms
}

// Stop the within-minute retry loop.
function stopSearching() {
  if (searchTimer) { clearTimeout(searchTimer); searchTimer = null }
  isSearching.value = false
}

// One search attempt: poll, then schedule the next one only after this one completes.
// Recursive setTimeout guarantees no concurrent polls.
// Keeps running until SEARCH_MAX_MS so late products still get resolved in place;
// raises a visible error if nothing new arrived by the timeout.
async function runSearch() {
  if (!isSearching.value) return
  const elapsed = Date.now() - searchStart
  if (elapsed >= SEARCH_MAX_MS) {
    if (!searchFoundAny) {
      const mark = new Date(searchStart).toLocaleTimeString('it-IT', {
        hour: '2-digit', minute: '2-digit', timeZone: 'Europe/Rome',
      })
      loadError.value = `No new data arrived within 3 minutes past the ${mark} mark — the server may be having issues.`
    }
    stopSearching()
    return
  }

  await pollForNewData()

  if (!isSearching.value) return

  // Stop early if all search-window timestamps are resolved across all products
  if (elapsed >= SEARCH_HOLDBACK_MS && searchWindowTs.value.length > 0) {
    const allResolved = searchWindowTs.value.every(ts =>
      productOrder.value.every(p => !productStats.value[p]?.missingSet?.has(ts))
    )
    if (allResolved) { stopSearching(); return }
  }

  searchTimer = setTimeout(runSearch, SEARCH_INTERVAL)
}

// Start a search window: kick off the first attempt immediately, then retry
// every 1s for up to 3 minutes.
function startDataSearch() {
  stopSearching()
  loadError.value = ''
  isSearching.value = true
  searchStart = Date.now()
  searchFoundAny = false
  searchWindowTs.value = []
  runSearch()
}

function startPolling() {
  stopPolling()

  const delay = msUntilNextFiveMinMark()
  nextUpdateSecs.value = Math.round(delay / 1000)

  // Step 1: fire at the exact next 5-minute clock mark
  initialTimer = setTimeout(() => {
    initialTimer = null
    startDataSearch()
    nextUpdateSecs.value = POLL_MS / 1000

    // Step 2: then repeat every 5 minutes exactly on the mark
    pollTimer = setInterval(() => {
      startDataSearch()
      nextUpdateSecs.value = POLL_MS / 1000
    }, POLL_MS)
  }, delay)

  countdownTimer = setInterval(() => {
    if (nextUpdateSecs.value > 0) nextUpdateSecs.value--
  }, 1000)
}

function stopPolling() {
  stopSearching()
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

// ---- Lifecycle ----
onMounted(async () => {
  // Wait for the browser to layout and size the Leaflet container before loading.
  // Without this, Leaflet may have zero-dimension tiles on first paint.
  await nextTick()
  await loadData({ preserve: false })
  startPolling()
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
  // Resume polling — it was stopped in onDeactivated.
  startPolling()
  // Immediately check for data that arrived while on another page.
  // startPolling() only arms the timer for the next 5-min mark; without this
  // call the map would show stale data until then.
  await pollForNewData()
})

onUnmounted(() => {
  stopAnimation()
  stopPolling()
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
</style>