<!--
  ModelComparisonView.vue — Side-by-side model comparison with all 12 lead times.

  Recreates the old Streamlit model_comparison.py layout:
  - 12 rows (one per lead time: +5 to +60 min)
  - Each row: GT + model prediction images (4/5 width) + CSI table (1/5 width)
  - Synchronized zoom/pan within each row (scroll to zoom, drag to pan)
  - CSI computed once via POST /api/metrics/comparison, displayed per row
-->
<template>
  <div class="min-h-[calc(100vh-3.5rem)] bg-gray-50">

    <!-- ================================================================ -->
    <!-- TOP BAR: Config panel (dark gradient, like real-time bottom bar)  -->
    <!-- ================================================================ -->
    <div class="bg-gradient-to-b from-gray-900 to-gray-800 px-6 py-5 shadow-lg">
      <div class="w-full max-w-full mx-auto">

        <!-- Title row -->
        <div class="flex items-center justify-between mb-4">
          <h1 class="text-xl font-bold text-white tracking-wide">Model Comparison</h1>
          <span
            v-if="selectedDateTime"
            class="text-xs font-medium px-3 py-1 rounded-full bg-white/10 text-gray-300"
          >
            {{ formatDateDisplay(selectedDateTime) }}
          </span>
        </div>

        <!-- Controls row -->
        <div class="flex items-end gap-4">

          <!-- Date/time group -->
          <div class="flex items-end gap-2">
            <!-- Date picker (VueDatePicker — large, dark-themed calendar) -->
            <div>
              <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Date</label>
              <VueDatePicker
                :model-value="pickerDate"
                @update:model-value="onPickerChange"
                @update-month-year="onMonthYearChange"
                :time-config="{ enableTimePicker: false }"
                :highlight="highlightedDates"
                auto-apply
                :dark="true"
                :format="formatPickerDate"
                model-type="yyyy-MM-dd"
                input-class-name="dp-dark-input"
              />
            </div>

            <!-- Small divider between date and time -->
            <div class="w-px h-[42px] bg-white/10 mx-1" />

            <!-- Time picker (single VueDatePicker with hour + minute) -->
            <div>
              <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Time</label>
              <VueDatePicker
                :model-value="timePickerValue"
                @update:model-value="onTimePickerChange"
                time-picker
                :dark="true"
                :is-24="true"
                :time-config="{ minutesIncrement: 5, minutesGridIncrement: 5 }"
                input-class-name="dp-dark-input dp-time-input"
              />
            </div>
          </div>

          <!-- Vertical divider -->
          <div class="h-[42px] w-px bg-white/10" />

          <!-- Model selector (chip-style checkboxes) + availability text -->
          <div class="flex-1">
            <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">
              Models
            </label>
            <div class="flex flex-wrap gap-2">
              <label
                v-for="model in models"
                :key="model"
                class="inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-medium
                       cursor-pointer transition-all select-none"
                :class="selectedModels.includes(model)
                  ? 'bg-blue-500/30 text-blue-300 ring-1 ring-blue-400/50'
                  : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-300'"
              >
                <input
                  type="checkbox"
                  :value="model"
                  v-model="selectedModels"
                  class="sr-only"
                />
                {{ model }}
              </label>
            </div>
            <!-- Availability text (shown after auto-check) -->
            <p v-if="availabilitySummary" class="text-[11px] mt-1.5" :class="allSelectedAvailable ? 'text-emerald-400' : 'text-amber-400'">
              {{ availabilitySummary }}
            </p>
          </div>

          <!-- Load / Compute button -->
          <button
            @click="loadComparison"
            :disabled="!canLoad"
            class="flex-shrink-0 h-[42px] px-5 rounded-lg font-semibold text-sm transition-all
                   flex items-center gap-2 self-end"
            :class="canLoad
              ? (allSelectedAvailable
                ? 'bg-blue-600 text-white hover:bg-blue-500 shadow-sm shadow-blue-500/30'
                : 'bg-amber-600 text-white hover:bg-amber-500 shadow-sm shadow-amber-500/30')
              : 'bg-white/10 text-gray-500 cursor-not-allowed'"
          >
            <svg v-if="loading" class="animate-spin w-4 h-4" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            {{ buttonLabel }}
          </button>
        </div>

        <!-- Error (inside dark bar) -->
        <div v-if="error" class="mt-3 px-4 py-2 rounded-lg bg-red-500/20 border border-red-500/30">
          <p class="text-sm text-red-300">{{ error }}</p>
        </div>

        <!-- Time availability panel (shows when date is selected and has data) -->
        <div v-if="dayDetail && Object.keys(dayDetail.slots).length > 0" class="mt-3">
          <button
            @click="dayDetailExpanded = !dayDetailExpanded"
            class="flex items-center gap-2 text-xs font-semibold text-gray-400 uppercase tracking-wider
                   hover:text-gray-300 transition-colors"
          >
            <svg
              class="w-3 h-3 transition-transform"
              :class="dayDetailExpanded ? 'rotate-90' : ''"
              fill="currentColor" viewBox="0 0 20 20"
            >
              <path fill-rule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clip-rule="evenodd" />
            </svg>
            Prediction Availability
            <span class="text-gray-500 normal-case font-normal tracking-normal">
              — {{ Object.keys(dayDetail.slots).length }} timestamps on {{ formatDateShort(dayDetail.date) }}
            </span>
          </button>

          <Transition name="slide">
            <div v-if="dayDetailExpanded" class="mt-2 bg-white/5 rounded-lg p-3 max-h-52 overflow-y-auto">
              <div class="space-y-0.5">
                <div
                  v-for="(models, time) in dayDetail.slots"
                  :key="time"
                  @click="selectTimeFromPanel(time)"
                  class="flex items-center gap-2 px-2 py-1.5 rounded-md cursor-pointer transition-colors
                         hover:bg-white/10"
                  :class="isCurrentTime(time) ? 'bg-blue-500/20 ring-1 ring-blue-400/40' : ''"
                >
                  <!-- All models indicator -->
                  <span
                    class="w-4 text-center text-xs flex-shrink-0"
                    :class="models.length === dayDetail.total_models
                      ? 'text-emerald-400'
                      : 'text-amber-400'"
                  >
                    {{ models.length === dayDetail.total_models ? '✓' : '◐' }}
                  </span>
                  <!-- Time -->
                  <span class="text-sm text-gray-200 font-mono w-12 flex-shrink-0">{{ time }}</span>
                  <!-- Model badges -->
                  <div class="flex gap-1 flex-wrap">
                    <span
                      v-for="model in models"
                      :key="model"
                      class="text-[10px] px-1.5 py-0.5 rounded font-medium"
                      :class="models.length === dayDetail.total_models
                        ? 'bg-emerald-500/20 text-emerald-300'
                        : 'bg-blue-500/20 text-blue-300'"
                    >
                      {{ model }}
                    </span>
                  </div>
                  <!-- Count on the right -->
                  <span class="ml-auto text-[10px] text-gray-500 flex-shrink-0">
                    {{ models.length }}/{{ dayDetail.total_models }}
                  </span>
                </div>
              </div>
            </div>
          </Transition>
        </div>
        <!-- Day detail: no data message -->
        <div v-else-if="dayDetail && Object.keys(dayDetail.slots).length === 0 && !dayDetailLoading" class="mt-3">
          <p class="text-xs text-gray-500">No predictions found for this date</p>
        </div>
        <div v-if="dayDetailLoading" class="mt-3">
          <p class="text-xs text-gray-500 animate-pulse">Loading availability...</p>
        </div>
      </div>
    </div>

    <!-- ================================================================ -->
    <!-- MAIN CONTENT: 12 lead-time sections                              -->
    <!-- ================================================================ -->
    <div class="w-full max-w-full mx-auto px-6 py-6">

      <!-- Zoom info bar + Download All -->
      <div v-if="showComparison" class="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-6 flex items-center justify-between">
        <p class="text-sm text-blue-700">
          <strong>Synchronized Zoom:</strong> scroll to zoom, drag to pan, double-click to reset.
          All images in the same row zoom and pan together.
        </p>
        <button
          @click="downloadAllImages"
          :disabled="downloadingAll"
          class="flex-shrink-0 ml-4 px-4 py-2 text-sm font-medium text-blue-700 hover:text-blue-800
                 border border-blue-300 rounded-lg hover:bg-blue-100 transition-colors flex items-center gap-1.5"
        >
          <svg v-if="downloadingAll" class="animate-spin w-4 h-4" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 15V3" />
          </svg>
          {{ downloadingAll ? 'Downloading...' : 'Download All Images' }}
        </button>
      </div>

      <!-- 12 lead-time sections -->
      <div v-if="showComparison" class="space-y-6">
        <div
          v-for="ltIdx in 12"
          :key="ltIdx - 1"
          class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden"
        >
          <!-- Section header -->
          <div class="px-5 py-3 bg-gray-50 border-b border-gray-100 flex items-center gap-3">
            <span class="text-sm font-bold text-gray-800">
              {{ formatLeadTimeLabel(ltIdx - 1) }}
            </span>
            <span class="text-xs font-medium px-2 py-0.5 rounded-full bg-blue-100 text-blue-700">
              +{{ ltIdx * 5 }} min
            </span>
            <span class="text-xs text-gray-400 ml-auto mr-2">Lead time {{ ltIdx }}/12</span>
            <button
              @click="downloadRow(ltIdx - 1)"
              class="p-1.5 rounded-lg text-gray-400 hover:text-blue-500 hover:bg-blue-50 transition-colors"
              title="Download images for this lead time"
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 15V3" />
              </svg>
            </button>
          </div>

          <!-- Content: images (left) + CSI table (right) -->
          <div class="flex gap-4 p-4">
            <!-- Images container (~4/5 width) -->
            <div class="flex-1 min-w-0">
              <div class="flex gap-2">
                <!-- Groundtruth -->
                <div class="flex-1 min-w-0">
                  <p class="text-xs font-semibold text-emerald-600 mb-1 text-center">Groundtruth</p>
                  <div
                    class="zoom-wrapper"
                    @wheel.prevent="onWheel($event, ltIdx - 1)"
                    @mousedown="onMouseDown($event, ltIdx - 1)"
                    @dblclick="resetZoom(ltIdx - 1)"
                    :style="{ cursor: dragging === ltIdx - 1 ? 'grabbing' : 'grab' }"
                  >
                    <img
                      :src="api.figureUrl(availableModels[0], selectedDateTime, ltIdx - 1, 'groundtruth')"
                      :style="zoomStyle(ltIdx - 1)"
                      class="w-full block pointer-events-none select-none"
                      alt="Groundtruth"
                      draggable="false"
                    />
                  </div>
                </div>

                <!-- One column per available model -->
                <div
                  v-for="model in availableModels"
                  :key="model"
                  class="flex-1 min-w-0"
                >
                  <p class="text-xs font-semibold text-blue-600 mb-1 text-center">{{ model }}</p>
                  <div
                    class="zoom-wrapper"
                    @wheel.prevent="onWheel($event, ltIdx - 1)"
                    @mousedown="onMouseDown($event, ltIdx - 1)"
                    @dblclick="resetZoom(ltIdx - 1)"
                    :style="{ cursor: dragging === ltIdx - 1 ? 'grabbing' : 'grab' }"
                  >
                    <img
                      :src="api.figureUrl(model, selectedDateTime, ltIdx - 1, 'prediction')"
                      :style="zoomStyle(ltIdx - 1)"
                      class="w-full block pointer-events-none select-none"
                      :alt="`${model} prediction`"
                      draggable="false"
                    />
                  </div>
                </div>
              </div>
            </div>

            <!-- CSI table (compact, rotated headers) -->
            <div class="flex-shrink-0">
              <p class="text-xs font-semibold text-gray-500 mb-1">CSI Scores</p>
              <div
                v-if="csiData && csiData.lead_times"
                class="csi-table-wrap"
              >
                <table class="csi-table">
                  <thead>
                    <tr>
                      <th class="corner-cell"><span class="text-[9px] text-gray-400 font-normal">mm/h</span></th>
                      <th
                        v-for="model in csiData.models"
                        :key="model"
                        class="rotated-header"
                        :title="model"
                      >
                        <div class="rotated-label">{{ model }}</div>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="th in csiData.thresholds"
                      :key="th"
                    >
                      <td class="row-label">{{ th }}</td>
                      <td
                        v-for="model in csiData.models"
                        :key="model"
                        class="csi-cell"
                        :class="csiCellClass(getCsiValue(ltIdx - 1, model, th))"
                      >
                        {{ formatCsi(getCsiValue(ltIdx - 1, model, th)) }}
                      </td>
                    </tr>
                    <tr class="avg-row">
                      <td class="row-label font-semibold">Avg</td>
                      <td
                        v-for="model in csiData.models"
                        :key="model"
                        class="csi-cell font-semibold"
                        :class="csiCellClass(getCsiAvg(ltIdx - 1, model))"
                      >
                        {{ formatCsi(getCsiAvg(ltIdx - 1, model)) }}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div v-else-if="csiLoading" class="flex items-center gap-2 text-xs text-gray-400 italic py-4">
                <svg class="animate-spin w-3 h-3" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Computing CSI...
              </div>
              <p v-else class="text-xs text-gray-400 italic">No CSI data</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Computing progress card -->
      <div v-if="computing" class="flex items-center justify-center py-20">
        <div class="bg-white rounded-xl shadow-lg border border-gray-100 p-6 w-full max-w-md">
          <h2 class="text-lg font-bold text-gray-800 mb-4">Computing Predictions</h2>
          <div class="space-y-3 mb-5">
            <div
              v-for="model in Object.keys(computeStatus)"
              :key="model"
            >
              <div class="flex items-center gap-3">
                <!-- Done -->
                <svg v-if="computeStatus[model].state === 'done'" class="w-5 h-5 text-emerald-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
                </svg>
                <!-- Error -->
                <svg v-else-if="computeStatus[model].state === 'error'" class="w-5 h-5 text-red-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
                <!-- Queued -->
                <svg v-else-if="computeStatus[model].state === 'queued'" class="w-5 h-5 text-amber-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="10" />
                  <path stroke-linecap="round" d="M12 6v6l4 2" />
                </svg>
                <!-- Submitting / Running (spinner) -->
                <svg v-else class="animate-spin w-5 h-5 text-blue-500 flex-shrink-0" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <!-- Model name + status -->
                <span class="text-sm font-medium text-gray-800">{{ model }}</span>
                <span
                  class="text-xs ml-auto"
                  :class="{
                    'text-emerald-600': computeStatus[model].state === 'done',
                    'text-red-500': computeStatus[model].state === 'error',
                    'text-amber-600': computeStatus[model].state === 'queued',
                    'text-blue-500': !['done', 'error', 'queued'].includes(computeStatus[model].state),
                  }"
                >
                  {{ computeStatusLabel(computeStatus[model].state) }}
                </span>
                <!-- View Log button (only for errors with a log) -->
                <button
                  v-if="computeStatus[model].state === 'error' && computeStatus[model].errorLog"
                  @click="toggleErrorLog(model)"
                  class="text-xs text-red-500 hover:text-red-700 underline ml-1"
                >
                  {{ expandedErrorLogs[model] ? 'Hide Log' : 'View Log' }}
                </button>
              </div>
              <!-- Expandable error log -->
              <div
                v-if="computeStatus[model].state === 'error' && computeStatus[model].errorLog && expandedErrorLogs[model]"
                class="mt-2 ml-8 rounded-lg bg-red-50 border border-red-200 p-3 max-h-48 overflow-auto"
              >
                <pre class="text-xs text-red-800 whitespace-pre-wrap font-mono">{{ computeStatus[model].errorLog }}</pre>
              </div>
            </div>
          </div>
          <p class="text-xs text-gray-500 mb-4">
            {{ computeProgress.done }}/{{ computeProgress.total }} models ready
          </p>
          <button
            @click="cancelCompute"
            class="w-full px-4 py-2 rounded-lg text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>

      <!-- Empty state (nothing loaded yet) -->
      <div v-if="!showComparison && !loading && !computing && !error" class="text-center py-20">
        <svg class="mx-auto w-16 h-16 text-gray-300 mb-4" fill="none" stroke="currentColor" stroke-width="1" viewBox="0 0 24 24">
          <path d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
        <p class="text-gray-400 text-sm">Select 2+ models and a date/time, then click <strong>Load Comparison</strong></p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, reactive, watch, onMounted, onBeforeUnmount } from 'vue'
import { VueDatePicker } from '@vuepic/vue-datepicker'
import '@vuepic/vue-datepicker/dist/main.css'
import api from '../api.js'
import { useConfigStore } from '../stores/config.js'

const configStore = useConfigStore()

// Filter out Test model — only shown in RealTime
const models = computed(() => configStore.models.filter(m => m.toUpperCase() !== 'TEST'))

// ---------------------------------------------------------------------------
// Calendar highlighting (dates with predictions)
// ---------------------------------------------------------------------------
const highlightedDates = computed(() => ({ dates: calendarDates.value }))
const calendarDates = ref([])
const dayDetail = ref(null)       // { date, models, slots, total_models }
const dayDetailLoading = ref(false)
const dayDetailExpanded = ref(true)

/**
 * Fetch which dates in a month have predictions, for calendar highlighting.
 * Called on mount and whenever the user navigates to a different month.
 */
async function fetchCalendarAvailability(year, month) {
  if (!models.value.length) return
  try {
    const res = await api.getCalendarAvailability(models.value, year, month)
    // Convert "YYYY-MM-DD" strings to Date objects for VueDatePicker highlight
    calendarDates.value = res.dates.map(d => new Date(d + 'T12:00:00'))
  } catch (e) {
    console.error('Failed to fetch calendar availability:', e)
  }
}

/** Called by VueDatePicker when the user navigates months. */
function onMonthYearChange({ year, month }) {
  // VueDatePicker months are 0-indexed (0=Jan, 11=Dec), backend expects 1-indexed
  fetchCalendarAvailability(year, month + 1)
}

/**
 * Fetch per-timestamp model availability for the selected date.
 * Shows which models have predictions at each 5-minute slot.
 */
async function fetchDayDetail(dateStr) {
  if (!dateStr) { dayDetail.value = null; return }
  if (!models.value.length) return

  dayDetailLoading.value = true
  try {
    dayDetail.value = await api.getDayDetail(models.value, dateStr)
  } catch (e) {
    console.error('Failed to fetch day detail:', e)
    dayDetail.value = null
  } finally {
    dayDetailLoading.value = false
  }
}

/** Clicking a time in the availability panel sets the time picker. */
function selectTimeFromPanel(timeStr) {
  const [h, m] = timeStr.split(':')
  selectedDateTime.value = buildDateTime(datePart.value, h, m)
}

/** Check if a time string matches the currently selected time. */
function isCurrentTime(timeStr) {
  return timeStr === `${hourValue.value}:${minuteValue.value}`
}

/** Format "YYYY-MM-DD" → "DD/MM/YYYY" */
function formatDateShort(dateStr) {
  if (!dateStr) return ''
  const [y, m, d] = dateStr.split('-')
  return `${d}/${m}/${y}`
}

/** Custom format function for VueDatePicker (ensures DD/MM/YYYY display). */
function formatPickerDate(date) {
  if (!date) return ''
  // With model-type="yyyy-MM-dd", date is a string like "2026-02-12"
  if (typeof date === 'string') {
    const parts = date.split('-')
    if (parts.length === 3) return `${parts[2]}/${parts[1]}/${parts[0]}`
    return date
  }
  const d = date instanceof Date ? date : new Date(date)
  return `${String(d.getDate()).padStart(2, '0')}/${String(d.getMonth() + 1).padStart(2, '0')}/${d.getFullYear()}`
}

// ---------------------------------------------------------------------------
// Date/time state — native date input + hour/minute dropdowns
// ---------------------------------------------------------------------------
const now = new Date()
const todayStr = `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')}`
const selectedDateTime = ref(`${todayStr}T00:00`)

// Parse date part as a Date object for the picker
const pickerDate = computed(() => {
  if (!selectedDateTime.value) return null
  // With model-type="yyyy-MM-dd", the picker expects a string like "2026-02-16"
  return selectedDateTime.value.split('T')[0] || null
})

// Format a Date back to "YYYY-MM-DD"
function dateToStr(d) {
  if (!d) return ''
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const dd = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${dd}`
}

const datePart = computed(() => {
  if (!selectedDateTime.value) return ''
  return selectedDateTime.value.split('T')[0] || ''
})
const hourValue = computed(() => {
  if (!selectedDateTime.value?.includes('T')) return ''
  return selectedDateTime.value.split('T')[1]?.split(':')[0] || ''
})
const minuteValue = computed(() => {
  if (!selectedDateTime.value?.includes('T')) return ''
  return selectedDateTime.value.split('T')[1]?.split(':')[1] || ''
})

function buildDateTime(date, hour, minute) {
  if (!date) return ''
  return `${date}T${hour || '00'}:${minute || '00'}`
}
function onPickerChange(val) {
  // With model-type="yyyy-MM-dd", val is a string like "2026-02-16"
  const ds = typeof val === 'string' ? val : dateToStr(val)
  if (ds) {
    selectedDateTime.value = buildDateTime(ds, hourValue.value, minuteValue.value)
  }
}
// VueDatePicker combined time picker value (object: { hours, minutes, seconds })
const timePickerValue = computed(() => ({
  hours: hourValue.value ? parseInt(hourValue.value) : 0,
  minutes: minuteValue.value ? parseInt(minuteValue.value) : 0,
  seconds: 0,
}))

function onTimePickerChange(val) {
  if (val && val.hours !== undefined && val.minutes !== undefined) {
    const h = String(val.hours).padStart(2, '0')
    const m = String(val.minutes).padStart(2, '0')
    selectedDateTime.value = buildDateTime(datePart.value, h, m)
  }
}

function formatDateDisplay(val) {
  if (!val?.includes('T')) return ''
  const [date, time] = val.split('T')
  const [y, m, d] = date.split('-')
  return `${d}/${m}/${y} ${time}`
}

// Fetch day detail whenever the date part changes
watch(datePart, (newDate) => {
  fetchDayDetail(newDate)
})

// Fetch initial calendar availability on mount
onMounted(() => {
  const d = datePart.value ? new Date(datePart.value) : new Date()
  fetchCalendarAvailability(d.getFullYear(), d.getMonth() + 1)
  if (datePart.value) fetchDayDetail(datePart.value)
})

// ---------------------------------------------------------------------------
// Model selection + availability
// ---------------------------------------------------------------------------
const selectedModels = ref([])
const availabilityMap = ref({})  // { model: true/false }
let checkAbort = null

// Auto-check availability whenever datetime changes
watch(selectedDateTime, async (newVal) => {
  if (!newVal || !newVal.includes('T') || newVal.length < 16) return

  if (checkAbort) checkAbort.cancelled = true
  const thisCheck = { cancelled: false }
  checkAbort = thisCheck

  const modelList = models.value
  const results = await Promise.allSettled(
    modelList.map(model => api.checkSinglePrediction(model, newVal))
  )

  if (thisCheck.cancelled) return

  const map = {}
  results.forEach((r, i) => {
    map[modelList[i]] = r.status === 'fulfilled' && r.value.exists
  })
  availabilityMap.value = map
})

// Availability summary text (shown below model chips)
const availabilitySummary = computed(() => {
  if (selectedModels.value.length === 0 || Object.keys(availabilityMap.value).length === 0) return ''
  const available = selectedModels.value.filter(m => availabilityMap.value[m])
  const missing = selectedModels.value.filter(m => availabilityMap.value[m] === false)
  if (missing.length === 0) return `All ${available.length} selected models have predictions`
  if (available.length === 0) return `No predictions found for selected models`
  return `${available.length} available, ${missing.length} missing: ${missing.join(', ')}`
})

// Are all selected models available?
const allSelectedAvailable = computed(() => {
  if (selectedModels.value.length === 0) return true
  return selectedModels.value.every(m => availabilityMap.value[m])
})

// ---------------------------------------------------------------------------
// Loading / results state
// ---------------------------------------------------------------------------
const loading = ref(false)
const computing = ref(false)
const showComparison = ref(false)
const error = ref(null)
const checkResults = ref([])

const csiData = ref(null)
const csiLoading = ref(false)

const canLoad = computed(() =>
  selectedModels.value.length >= 2 && selectedDateTime.value && selectedDateTime.value.length >= 16 && !loading.value && !computing.value
)

const availableModels = computed(() =>
  checkResults.value.filter(r => r.exists).map(r => r.model)
)

// Dynamic button label
const buttonLabel = computed(() => {
  if (loading.value) return 'Loading...'
  if (computing.value) return 'Computing...'
  if (!allSelectedAvailable.value && selectedModels.value.length >= 2) return 'Compute Predictions'
  return 'Load Comparison'
})

// ---------------------------------------------------------------------------
// Computing state (job submission + polling for missing predictions)
// ---------------------------------------------------------------------------
const computeStatus = reactive({})  // { model: { state, jobId, errorLog? } }
const expandedErrorLogs = reactive({})  // { model: bool }
let pollAbort = null

const computeProgress = computed(() => {
  const models = Object.keys(computeStatus)
  const done = models.filter(m => computeStatus[m].state === 'done').length
  return { done, total: models.length }
})

function computeStatusLabel(state) {
  switch (state) {
    case 'submitting': return 'Submitting...'
    case 'queued': return 'Queued'
    case 'running': return 'Running...'
    case 'done': return 'Done'
    case 'error': return 'Failed'
    default: return state
  }
}

/**
 * Submit jobs for missing models, poll HPC jobs until complete.
 * Resolves when all jobs are done (or cancelled/timed out).
 */
async function computeMissing(missingModels) {
  computing.value = true
  for (const k of Object.keys(computeStatus)) delete computeStatus[k]
  const abort = { cancelled: false }
  pollAbort = abort

  // Init per-model status
  for (const model of missingModels) {
    computeStatus[model] = { state: 'submitting', jobId: null }
  }

  // Submit all jobs in parallel
  const results = await Promise.allSettled(
    missingModels.map(m =>
      api.submitJob(m, selectedDateTime.value, selectedDateTime.value)
    )
  )

  if (abort.cancelled) return

  const hpcModels = []
  results.forEach((r, i) => {
    const model = missingModels[i]
    if (r.status === 'fulfilled' && r.value.success) {
      if (r.value.is_mock) {
        computeStatus[model] = { state: 'done', jobId: r.value.job_id }
      } else {
        computeStatus[model] = { state: 'queued', jobId: r.value.job_id }
        hpcModels.push(model)
      }
    } else {
      computeStatus[model] = { state: 'error', jobId: null }
    }
  })

  // Poll HPC jobs until complete
  if (hpcModels.length > 0 && !abort.cancelled) {
    await pollUntilComplete(hpcModels, abort)
  }

  if (!abort.cancelled) {
    computing.value = false
  }
}

/**
 * Poll job status + prediction existence every 3s until all done.
 * When a job leaves the PBS queue, we give it 10 consecutive checks (~30s)
 * to find the prediction file. If not found, we declare it failed and
 * fetch the PBS error log.
 */
async function pollUntilComplete(hpcModels, abort) {
  const pending = new Set(hpcModels)
  const failedChecks = new Map()  // model → consecutive "no predictions" count

  while (pending.size > 0 && !abort.cancelled) {
    await new Promise(r => setTimeout(r, 3000))
    if (abort.cancelled) break

    for (const model of [...pending]) {
      try {
        // Pass jobId for direct PBS lookup (avoids matching wrong jobs)
        const storedJobId = computeStatus[model].jobId
        const jobStatus = await api.getJobStatus(model, storedJobId)
        if (jobStatus.status === 'Q') {
          computeStatus[model] = { ...computeStatus[model], state: 'queued' }
        } else if (jobStatus.status === 'R') {
          computeStatus[model] = { ...computeStatus[model], state: 'running' }
          failedChecks.delete(model)  // reset if job is still running
        } else {
          // Job left queue (status is null) — check if prediction file appeared
          const pred = await api.checkSinglePrediction(model, selectedDateTime.value)
          if (pred.exists) {
            computeStatus[model] = { ...computeStatus[model], state: 'done' }
            pending.delete(model)
            failedChecks.delete(model)
          } else {
            // Predictions not found — track consecutive misses
            const count = (failedChecks.get(model) || 0) + 1
            failedChecks.set(model, count)
            if (count >= 10) {
              // ~30 seconds with no predictions after job left queue → failure
              const jobId = storedJobId
              let errorLog = null
              if (jobId) {
                try {
                  const logResult = await api.getJobErrorLog(model, jobId)
                  if (logResult.found) errorLog = logResult.log
                } catch { /* ignore */ }
              }
              computeStatus[model] = { ...computeStatus[model], state: 'error', errorLog }
              pending.delete(model)
              failedChecks.delete(model)
            }
          }
        }
      } catch {
        computeStatus[model] = { ...computeStatus[model], state: 'error' }
        pending.delete(model)
      }
    }
  }
}

function toggleErrorLog(model) {
  expandedErrorLogs[model] = !expandedErrorLogs[model]
}

function cancelCompute() {
  if (pollAbort) pollAbort.cancelled = true
  computing.value = false
  loading.value = false
  for (const k of Object.keys(computeStatus)) delete computeStatus[k]
  for (const k of Object.keys(expandedErrorLogs)) delete expandedErrorLogs[k]
}

// ---------------------------------------------------------------------------
// Lead time label: shows target time with offset, e.g. "11:25 (+5 min)"
// ---------------------------------------------------------------------------
function formatLeadTimeLabel(ltIdx) {
  if (!selectedDateTime.value?.includes('T')) return ''
  try {
    const base = new Date(selectedDateTime.value)
    const offsetMin = (ltIdx + 1) * 5
    const target = new Date(base.getTime() + offsetMin * 60000)
    const hh = String(target.getHours()).padStart(2, '0')
    const mm = String(target.getMinutes()).padStart(2, '0')
    return `${hh}:${mm}`
  } catch { return '' }
}

// ---------------------------------------------------------------------------
// Zoom state — one entry per lead-time row (0-11)
// ---------------------------------------------------------------------------
const zoom = reactive(
  Array.from({ length: 12 }, () => ({ scale: 1, tx: 0, ty: 0 }))
)
const dragging = ref(null)
let dragStart = { x: 0, y: 0, tx: 0, ty: 0 }

function zoomStyle(rowIdx) {
  const z = zoom[rowIdx]
  return {
    transform: `translate(${z.tx}px, ${z.ty}px) scale(${z.scale})`,
    transformOrigin: 'center center',
    transition: dragging.value === rowIdx ? 'none' : 'transform 0.15s ease-out',
  }
}

function onWheel(event, rowIdx) {
  const delta = event.deltaY > 0 ? -0.15 : 0.15
  const z = zoom[rowIdx]
  z.scale = Math.min(Math.max(z.scale + delta, 1), 6)
  if (z.scale <= 1) { z.tx = 0; z.ty = 0 }
}

function onMouseDown(event, rowIdx) {
  dragging.value = rowIdx
  const z = zoom[rowIdx]
  dragStart = { x: event.clientX, y: event.clientY, tx: z.tx, ty: z.ty }
  window.addEventListener('mousemove', onMouseMove)
  window.addEventListener('mouseup', onMouseUp)
}

function onMouseMove(event) {
  if (dragging.value === null) return
  const z = zoom[dragging.value]
  z.tx = dragStart.tx + (event.clientX - dragStart.x)
  z.ty = dragStart.ty + (event.clientY - dragStart.y)
}

function onMouseUp() {
  dragging.value = null
  window.removeEventListener('mousemove', onMouseMove)
  window.removeEventListener('mouseup', onMouseUp)
}

function resetZoom(rowIdx) {
  zoom[rowIdx].scale = 1
  zoom[rowIdx].tx = 0
  zoom[rowIdx].ty = 0
}

onBeforeUnmount(() => {
  window.removeEventListener('mousemove', onMouseMove)
  window.removeEventListener('mouseup', onMouseUp)
  if (pollAbort) pollAbort.cancelled = true
})

// ---------------------------------------------------------------------------
// CSI helpers
// ---------------------------------------------------------------------------
function getCsiValue(ltIdx, model, threshold) {
  if (!csiData.value?.lead_times?.[ltIdx]?.csi?.[model]) return null
  return csiData.value.lead_times[ltIdx].csi[model][String(threshold)]
}

function getCsiAvg(ltIdx, model) {
  if (!csiData.value?.lead_times?.[ltIdx]?.csi?.[model]) return null
  return csiData.value.lead_times[ltIdx].csi[model].avg
}

function formatCsi(val) {
  if (val === null || val === undefined) return '-'
  return val.toFixed(3)
}

function csiCellClass(val) {
  if (val === null || val === undefined) return 'text-gray-300'
  if (val >= 0.6) return 'text-green-700 bg-green-50'
  if (val >= 0.3) return 'text-yellow-700 bg-yellow-50'
  return 'text-red-700 bg-red-50'
}

// ---------------------------------------------------------------------------
// Image download helpers
// ---------------------------------------------------------------------------
const downloadingAll = ref(false)

async function downloadImage(url, filename) {
  try {
    const response = await fetch(url)
    const blob = await response.blob()
    const blobUrl = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = blobUrl
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(blobUrl)
  } catch (e) {
    console.error(`Failed to download ${filename}:`, e)
  }
}

function imageFilename(type, model, ltIdx) {
  const ts = selectedDateTime.value.replace('T', '_').replaceAll(':', '')
  const lt = String((ltIdx + 1) * 5).padStart(2, '0')
  if (type === 'groundtruth') return `GT_${ts}_+${lt}min.png`
  return `${model}_${ts}_+${lt}min.png`
}

async function downloadRow(ltIdx) {
  // Download groundtruth
  await downloadImage(
    api.figureUrl(availableModels.value[0], selectedDateTime.value, ltIdx, 'groundtruth'),
    imageFilename('groundtruth', null, ltIdx)
  )
  // Download each model prediction
  for (const model of availableModels.value) {
    await new Promise(r => setTimeout(r, 100))
    await downloadImage(
      api.figureUrl(model, selectedDateTime.value, ltIdx, 'prediction'),
      imageFilename('prediction', model, ltIdx)
    )
  }
}

async function downloadAllImages() {
  downloadingAll.value = true
  try {
    for (let i = 0; i < 12; i++) {
      await downloadRow(i)
      await new Promise(r => setTimeout(r, 200))
    }
  } finally {
    downloadingAll.value = false
  }
}

// ---------------------------------------------------------------------------
// Load comparison data
// ---------------------------------------------------------------------------
async function loadComparison() {
  error.value = null
  loading.value = true
  checkResults.value = []
  showComparison.value = false
  csiData.value = null

  for (let i = 0; i < 12; i++) {
    zoom[i].scale = 1; zoom[i].tx = 0; zoom[i].ty = 0
  }

  try {
    // Step 1: Check which models have predictions
    const promises = selectedModels.value.map(model =>
      api.checkSinglePrediction(model, selectedDateTime.value)
    )
    checkResults.value = await Promise.all(promises)

    const missing = selectedModels.value.filter(
      m => !checkResults.value.find(r => r.model === m)?.exists
    )

    // Step 2: If some models are missing, submit jobs and wait
    if (missing.length > 0) {
      loading.value = false
      await computeMissing(missing)

      // If cancelled, abort the whole flow
      if (pollAbort?.cancelled) return

      // Re-check all models after compute
      loading.value = true
      const recheck = selectedModels.value.map(model =>
        api.checkSinglePrediction(model, selectedDateTime.value)
      )
      checkResults.value = await Promise.all(recheck)

      // Update availability map so chips reflect new state
      const map = {}
      checkResults.value.forEach(r => { map[r.model] = r.exists })
      availabilityMap.value = map
    }

    if (availableModels.value.length < 2) {
      error.value = 'Need at least 2 models with available predictions for comparison.'
      return
    }

    showComparison.value = true
    fetchCsiData()
  } catch (e) {
    error.value = `Failed to load comparison: ${e.message}`
  } finally {
    loading.value = false
  }
}

async function fetchCsiData() {
  csiLoading.value = true
  try {
    csiData.value = await api.computeComparison(
      availableModels.value,
      selectedDateTime.value
    )
  } catch (e) {
    console.error('CSI computation failed:', e)
  } finally {
    csiLoading.value = false
  }
}
</script>

<style scoped>
.zoom-wrapper {
  overflow: hidden;
  border-radius: 0.375rem;
  border: 1px solid #e5e7eb;
  background: #f9fafb;
  position: relative;
}

/* ---- VueDatePicker dark input override ---- */
:deep(.dp-dark-input) {
  height: 42px !important;
  border-radius: 0.5rem !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  background: rgba(255, 255, 255, 0.05) !important;
  color: white !important;
  font-size: 0.875rem !important;
  padding: 0 0.75rem !important;
  width: 160px;
}
:deep(.dp-dark-input:focus) {
  border-color: #60a5fa !important;
  box-shadow: 0 0 0 1px #60a5fa !important;
}
:deep(.dp__input_wrap) {
  width: 160px;
}

/* ---- Time picker input width ---- */
:deep(.dp-time-input) {
  width: 120px;
}

/* ---- Make VueDatePicker select/action buttons bigger ---- */
:deep(.dp__action_row) {
  padding: 8px 12px;
}
:deep(.dp__action_button) {
  height: 36px;
  padding: 0 16px;
  font-size: 0.875rem;
  font-weight: 600;
}

/* ---- Compact CSI table with rotated headers ---- */
.csi-table-wrap {
  padding-top: 70px;  /* room for angled labels */
}

.csi-table {
  border-collapse: collapse;
  font-size: 11px;
}

.csi-table .corner-cell {
  width: 34px;
  border: 1px solid #e5e7eb;
  background: #f9fafb;
  text-align: center;
  vertical-align: bottom;
  padding: 2px 4px;
}

.csi-table .rotated-header {
  position: relative;
  height: 0;
  padding: 0;
  vertical-align: bottom;
  width: 48px;
  min-width: 48px;
}

.csi-table .rotated-label {
  position: absolute;
  bottom: 4px;
  left: 50%;
  /* Text starts at column center, rotates upward-left */
  transform-origin: bottom left;
  transform: rotate(-50deg);
  white-space: nowrap;
  font-weight: 600;
  font-size: 10px;
  color: #374151;
}

.csi-table .row-label {
  border: 1px solid #e5e7eb;
  padding: 2px 6px;
  font-weight: 500;
  background: #f9fafb;
  color: #6b7280;
  text-align: right;
  font-size: 10px;
}

.csi-table .csi-cell {
  border: 1px solid #e5e7eb;
  padding: 2px 4px;
  text-align: center;
  font-variant-numeric: tabular-nums;
  font-size: 10px;
}

.csi-table .avg-row .row-label {
  background: #f3f4f6;
  color: #374151;
}

/* ---- Highlight today in calendar ---- */
:deep(.dp__today) {
  border: 2px solid #ef4444 !important;
}

/* ---- Highlight dates with predictions ---- */
:deep(.dp__cell_highlight) {
  background-color: rgba(16, 185, 129, 0.25) !important;
  border-radius: 50% !important;
}

/* ---- Slide transition for availability panel ---- */
.slide-enter-active {
  transition: all 0.25s ease-out;
}
.slide-leave-active {
  transition: all 0.2s ease-in;
}
.slide-enter-from,
.slide-leave-to {
  opacity: 0;
  max-height: 0;
  margin-top: 0;
  padding-top: 0;
  padding-bottom: 0;
  overflow: hidden;
}
</style>
