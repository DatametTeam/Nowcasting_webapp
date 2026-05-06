<template>
  <div class="px-4 py-4 space-y-4 min-h-screen bg-gray-50">

    <!-- ── Controls bar ─────────────────────────────────────────────────── -->
    <div class="flex flex-wrap items-center gap-3">

      <!-- Scale selector -->
      <div class="flex items-center gap-2">
        <span class="text-sm font-medium text-gray-600">Scale</span>
        <div class="flex rounded-lg overflow-hidden border border-gray-300 bg-white">
          <button
            v-for="s in SCALES"
            :key="s"
            @click="selectedScale = s"
            class="px-3 py-1.5 text-xs font-medium transition-colors border-r border-gray-300 last:border-r-0"
            :class="selectedScale === s
              ? 'bg-blue-600 text-white'
              : 'text-gray-600 hover:bg-gray-100'"
          >{{ s }} km</button>
        </div>
      </div>

      <!-- Sub-tab -->
      <div class="flex rounded-lg overflow-hidden border border-gray-300 bg-white">
        <button
          @click="activeTab = 'recent'"
          class="px-4 py-1.5 text-xs font-medium transition-colors"
          :class="activeTab === 'recent'
            ? 'bg-blue-600 text-white'
            : 'text-gray-600 hover:bg-gray-100'"
        >Recent (24 h)</button>
        <button
          @click="activeTab = 'daily'"
          class="px-4 py-1.5 text-xs font-medium transition-colors border-l border-gray-300"
          :class="activeTab === 'daily'
            ? 'bg-blue-600 text-white'
            : 'text-gray-600 hover:bg-gray-100'"
        >Monthly (90 d)</button>
      </div>

      <!-- WS status dot -->
      <div
        class="w-2 h-2 rounded-full flex-shrink-0"
        :class="wsConnected ? 'bg-green-400' : 'bg-gray-300'"
        :title="wsConnected ? 'Live updates connected' : 'Live updates disconnected (polling fallback active)'"
      />

      <!-- Right side: spinner / last-updated -->
      <div class="ml-auto flex items-center gap-2 text-xs text-gray-500">
        <template v-if="loading">
          <svg class="w-3.5 h-3.5 animate-spin text-blue-500" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
          </svg>
          <span>Loading…</span>
        </template>
        <template v-else-if="lastUpdated">
          <span>Updated {{ formatUpdated(lastUpdated) }}</span>
        </template>
        <template v-else-if="error">
          <span class="text-red-500">{{ error }}</span>
        </template>
      </div>
    </div>

    <!-- ── Empty state ───────────────────────────────────────────────────── -->
    <div
      v-if="!loading && chartData && chartData.models.length === 0"
      class="text-center py-20 text-gray-400 text-sm"
    >
      No FSS data found at <code class="text-xs bg-gray-100 px-1 rounded">/data/FSS_metrics</code>
    </div>

    <!-- ── Chart grid ────────────────────────────────────────────────────── -->
    <template v-else>
      <div class="overflow-x-auto">
        <div class="chart-grid" :style="gridStyle">

          <!-- Top-left spacer -->
          <div />

          <!-- Column headers: thresholds -->
          <div
            v-for="thr in THRESHOLDS"
            :key="thr"
            class="text-center text-xs font-semibold text-gray-500 uppercase tracking-wide pb-1"
          >
            {{ thr }} mm/h
          </div>

          <!-- Rows: one per lead time -->
          <template v-for="lt in LEAD_TIMES" :key="lt">
            <!-- Row label -->
            <div class="flex items-center justify-end pr-3 text-xs font-semibold text-gray-500">
              +{{ lt }} min
            </div>

            <!-- Three charts (one per threshold) -->
            <div
              v-for="thr in THRESHOLDS"
              :key="`${lt}-${thr}`"
              :ref="el => setChartRef(lt, thr, el)"
              class="chart-cell rounded border border-gray-200 bg-white"
            />
          </template>
        </div>
      </div>

      <!-- ── Shared legend ───────────────────────────────────────────────── -->
      <div v-if="chartData?.models?.length" class="flex flex-wrap justify-center gap-6 pt-1">
        <div
          v-for="model in chartData.models"
          :key="model"
          class="flex items-center gap-2 text-sm text-gray-600"
        >
          <svg width="28" height="6">
            <line x1="0" y1="3" x2="28" y2="3" :stroke="modelColor(model)" stroke-width="2.5"/>
          </svg>
          {{ model }}
        </div>
        <!-- 0.5 skill line -->
        <div class="flex items-center gap-2 text-sm text-gray-400">
          <svg width="28" height="6">
            <line x1="0" y1="3" x2="28" y2="3" stroke="#aaa" stroke-width="1.5" stroke-dasharray="4,3"/>
          </svg>
          FSS = 0.5
        </div>
        <!-- Gap / no-data indicator -->
        <div class="flex items-center gap-2 text-sm text-gray-400">
          <svg width="28" height="6">
            <line x1="0" y1="3" x2="28" y2="3" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4,3" opacity="0.7"/>
          </svg>
          no data
        </div>
      </div>

      <!-- ── Min-valid filter info (monthly tab only) ──────────────────────── -->
      <div
        v-if="activeTab === 'daily' && chartData?.min_valid"
        class="flex justify-center pt-1"
      >
        <p class="text-xs text-gray-400">
          Monthly filter — days excluded if n valid &lt;
          <span v-for="(thr, i) in THRESHOLDS" :key="thr">
            <b>{{ chartData.min_valid[String(thr)] ?? '—' }}</b>
            <span class="text-gray-300"> ({{ thr }} mm/h)</span>
            <span v-if="i < THRESHOLDS.length - 1"> · </span>
          </span>
        </p>
      </div>
    </template>

  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import * as echarts from 'echarts'
import { useFssWs } from '../composables/useFssWs.js'
import api from '../api.js'

// ── Constants ───────────────────────────────────────────────────────────────

const LEAD_TIMES = [15, 30, 45, 60]
const THRESHOLDS = [5, 10, 25]
const SCALES     = [1, 5, 20]

const MODEL_COLORS = {
  ConvLSTM:    '#3b82f6',
  SPROG:       '#f97316',
  ED_ConvLSTM: '#22c55e',
  IAM4VP:      '#a855f7',
  PredFormer:  '#ec4899',
  DynamicUnet: '#14b8a6',
}
const FALLBACK_COLORS = ['#6366f1', '#84cc16', '#eab308', '#f43f5e', '#06b6d4']

function modelColor(model) {
  return MODEL_COLORS[model] ?? FALLBACK_COLORS[0]
}

// ── State ───────────────────────────────────────────────────────────────────

const selectedScale = ref(5)
const activeTab     = ref('recent')
const loading       = ref(false)
const error         = ref(null)
const lastUpdated   = ref(null)
const chartData     = ref(null)

// ── Grid layout ─────────────────────────────────────────────────────────────

const gridStyle = computed(() => ({
  display: 'grid',
  gridTemplateColumns: `80px repeat(${THRESHOLDS.length}, 1fr)`,
  gap: '8px',
  minWidth: '640px',
}))

// ── ECharts instances ────────────────────────────────────────────────────────

const chartEls       = {}   // { 'lt15-thr5': HTMLElement }
const chartInstances = {}   // { 'lt15-thr5': echarts instance }

function chartKey(lt, thr) { return `lt${lt}-thr${thr}` }

function setChartRef(lt, thr, el) {
  if (el) chartEls[chartKey(lt, thr)] = el
}

function initCharts() {
  for (const lt of LEAD_TIMES) {
    for (const thr of THRESHOLDS) {
      const key = chartKey(lt, thr)
      const el  = chartEls[key]
      if (el && !chartInstances[key]) {
        chartInstances[key] = echarts.init(el, null, { renderer: 'canvas' })
      }
    }
  }
}

function disposeCharts() {
  for (const key of Object.keys(chartInstances)) {
    chartInstances[key]?.dispose()
    delete chartInstances[key]
  }
}

// ── Build ECharts option for one cell ────────────────────────────────────────

function buildOption(lt, thr) {
  const data    = chartData.value
  const isRecent = activeTab.value === 'recent'
  const ltKey   = `lt${lt}`
  const thrKey  = `thr${thr}`

  const regularSeries = []
  const gapSeries     = []

  for (const model of (data?.models ?? [])) {
    const points = data?.series?.[model]?.[ltKey]?.[thrKey] ?? []
    const color  = modelColor(model)

    // Keep null entries so ECharts can detect and render gaps
    const seriesData = points.map(p => ({
      value:  [new Date(p.t).getTime(), (p.v !== null && p.v !== undefined) ? p.v : null],
      nValid: p.n ?? null,
    }))

    // Solid line — breaks at null values
    regularSeries.push({
      name:         model,
      type:         'line',
      data:         seriesData,
      lineStyle:    { color, width: 2 },
      itemStyle:    { color },
      showSymbol:   false,
      smooth:       false,
      connectNulls: false,
      z:            3,
    })

    // Dashed gap-bridge — same data but connects through nulls, rendered below solid
    gapSeries.push({
      name:         `__gap_${model}`,
      type:         'line',
      data:         seriesData,
      lineStyle:    { color, width: 1, type: 'dashed', opacity: 0.45 },
      itemStyle:    { color, opacity: 0 },
      showSymbol:   false,
      smooth:       false,
      connectNulls: true,
      z:            2,
      silent:       true,
    })
  }

  // Render gap series first (z:2) so solid line (z:3) overdraws them where data exists
  const series = [...gapSeries, ...regularSeries]

  // 0.5 skill-score reference line on the first REGULAR series
  if (regularSeries.length > 0) {
    regularSeries[0].markLine = {
      silent:    true,
      symbol:    'none',
      lineStyle: { color: '#bbb', type: 'dashed', width: 1 },
      label:     { show: false },
      data:      [{ yAxis: 0.5 }],
    }
  }

  const xLabelFormat = isRecent
    ? '{HH}:{mm}'
    : (value) => {
        const d = new Date(value)
        return `${String(d.getDate()).padStart(2,'0')}/${String(d.getMonth()+1).padStart(2,'0')}`
      }

  return {
    backgroundColor: 'transparent',
    animation: false,
    tooltip: {
      trigger:   'axis',
      confine:   true,
      textStyle: { fontSize: 11 },
      axisPointer: { type: 'line', lineStyle: { color: '#ccc' } },
      formatter(params) {
        if (!params.length) return ''
        const date = new Date(params[0].axisValue)
        const label = isRecent
          ? date.toLocaleString('it-IT', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })
          : date.toLocaleDateString('it-IT', { day: '2-digit', month: '2-digit', year: '2-digit' })
        // Filter out the dashed gap-bridge series from tooltip display
        const visible = params.filter(p => !p.seriesName.startsWith('__gap_'))
        const rows = visible
          .filter(p => p.value?.[1] != null)
          .map(p => `${p.marker} ${p.seriesName}: <b>${p.value[1].toFixed(3)}</b>`)
        const nValid = visible[0]?.data?.nValid
        const nLabel = isRecent ? 'n valid' : 'mean daily n valid'
        const nLine = nValid != null
          ? `<br/><span style="color:#aaa;font-size:10px">${nLabel}: ${nValid.toLocaleString()}</span>`
          : ''
        const noData = rows.length === 0
          ? `<br/><span style="color:#bbb;font-size:10px">no data</span>`
          : ''
        return `${label}${rows.length ? '<br/>' + rows.join('<br/>') : ''}${nLine}${noData}`
      },
    },
    grid: {
      top:    regularSeries.length > 0 ? 28 : 14,
      right:  6,
      bottom: 24,
      left:   40,
    },
    xAxis: {
      type:       'time',
      axisLabel:  {
        fontSize:  9,
        color:     '#888',
        formatter: isRecent ? '{HH}:{mm}' : xLabelFormat,
        rotate:    isRecent ? 0 : 30,
      },
      splitLine:  { show: false },
      axisLine:   { lineStyle: { color: '#e0e0e0' } },
      axisTick:   { lineStyle: { color: '#e0e0e0' } },
    },
    yAxis: {
      type:       'value',
      min:        0,
      max:        1,
      interval:    0.2,
      axisLabel:  {
        fontSize:  9,
        color:     '#888',
        formatter: v => v.toFixed(1),
      },
      splitLine:  { lineStyle: { color: '#f0f0f0' } },
      axisLine:   { lineStyle: { color: '#e0e0e0' } },
    },
    series,
    // Colored mean annotation: one text element per model
    graphic: buildMeanGraphics(data, ltKey, thrKey),
  }
}

function buildMeanGraphics(data, ltKey, thrKey) {
  const models = data?.models ?? []
  const elements = []
  let xPos = 44  // start just inside the plot area (grid.left = 40)

  // μ prefix in neutral gray
  elements.push({
    type: 'text', left: xPos, top: 5,
    style: { text: 'μ ', fontSize: 9, fill: '#aaa' },
  })
  xPos += 12

  for (const model of models) {
    const mean = data?.means?.[model]?.[ltKey]?.[thrKey]
    if (mean === null || mean === undefined) continue
    const label = `${model}: ${mean.toFixed(3)}  `
    elements.push({
      type: 'text', left: xPos, top: 5,
      style: { text: label, fontSize: 9, fill: modelColor(model) },
    })
    xPos += label.length * 5.3  // ~5.3 px per character at fontSize 9
  }

  return elements.length > 1 ? elements : []
}

function updateCharts() {
  for (const lt of LEAD_TIMES) {
    for (const thr of THRESHOLDS) {
      const key      = chartKey(lt, thr)
      const instance = chartInstances[key]
      if (instance) {
        instance.setOption(buildOption(lt, thr), { notMerge: true })
      }
    }
  }
}

// ── Data fetching ────────────────────────────────────────────────────────────

async function fetchData() {
  loading.value = true
  error.value   = null
  try {
    if (activeTab.value === 'recent') {
      chartData.value  = await api.fssRecent(selectedScale.value)
      lastUpdated.value = chartData.value.last_updated
    } else {
      chartData.value  = await api.fssDaily(selectedScale.value)
      lastUpdated.value = null
    }
    await nextTick()
    initCharts()
    updateCharts()
  } catch (e) {
    error.value = e.message ?? 'Failed to load FSS data'
  } finally {
    loading.value = false
  }
}

// ── WebSocket push + fallback poll ───────────────────────────────────────────

const { connected: wsConnected } = useFssWs({
  onFssUpdated: () => fetchData(),
})

let pollTimer = null

// ── Resize handling ──────────────────────────────────────────────────────────

function handleResize() {
  for (const inst of Object.values(chartInstances)) {
    inst?.resize()
  }
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

onMounted(async () => {
  await nextTick()
  initCharts()
  await fetchData()
  pollTimer = setInterval(fetchData, 5 * 60 * 1000)
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  clearInterval(pollTimer)
  window.removeEventListener('resize', handleResize)
  disposeCharts()
})

// Re-fetch (and rebuild charts) when scale or tab changes
watch([selectedScale, activeTab], async () => {
  chartData.value = null
  await nextTick()
  // Dispose so charts reinitialise with fresh options (avoids axis-type mismatch)
  disposeCharts()
  await nextTick()
  initCharts()
  await fetchData()
})

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatUpdated(iso) {
  if (!iso) return ''
  try {
    return new Date(iso).toLocaleTimeString('it-IT', { hour: '2-digit', minute: '2-digit' })
  } catch {
    return iso
  }
}
</script>

<style scoped>
.chart-cell {
  height: 200px;
}
</style>
