<!--
  MetricsView.vue — CSI / POD / FAR / FSS / RMSE metrics analysis.

  Full rewrite to match Streamlit csi_analysis.py feature set:
  - Dark top bar with date range pickers, model chips, availability checking
  - Job submission for missing predictions (same pattern as ModelComparisonView)
  - Tabbed results: CSI, FSS, RMSE
  - Color-coded tables + Chart.js line charts
-->
<template>
  <div class="min-h-[calc(100vh-3.5rem)] bg-gray-50">

    <!-- ================================================================ -->
    <!-- TOP BAR: Config panel (dark gradient)                            -->
    <!-- ================================================================ -->
    <div class="bg-gradient-to-b from-gray-900 to-gray-800 px-6 py-5 shadow-lg">
      <div class="w-full max-w-full mx-auto">

        <!-- Title row -->
        <div class="flex items-center justify-between mb-4">
          <h1 class="text-xl font-bold text-white tracking-wide">Metrics Analysis</h1>
          <span
            v-if="startDateTime && endDateTime"
            class="text-xs font-medium px-3 py-1 rounded-full bg-white/10 text-gray-300"
          >
            {{ formatDateDisplay(startDateTime) }} &mdash; {{ formatDateDisplay(endDateTime) }}
          </span>
        </div>

        <!-- Controls row -->
        <div class="flex items-end gap-4 flex-wrap">

          <!-- Start date/time group -->
          <div class="flex items-end gap-2">
            <div>
              <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Start Date</label>
              <VueDatePicker
                :model-value="startDate"
                @update:model-value="onStartDateChange"
                :time-config="{ enableTimePicker: false }"
                auto-apply
                :dark="true"
                format="dd/MM/yyyy"
                model-type="yyyy-MM-dd"
                no-today
                input-class-name="dp-dark-input"
              />
            </div>
            <div class="w-px h-[42px] bg-white/10 mx-1" />
            <div>
              <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Start Time</label>
              <VueDatePicker
                :model-value="startTimeObj"
                @update:model-value="onStartTimeChange"
                time-picker
                :dark="true"
                :is-24="true"
                :time-config="{ minutesIncrement: 5, minutesGridIncrement: 5 }"
                input-class-name="dp-dark-input dp-time-input"
              />
            </div>
          </div>

          <!-- Arrow -->
          <div class="text-gray-500 text-lg pb-2">&rarr;</div>

          <!-- End date/time group -->
          <div class="flex items-end gap-2">
            <div>
              <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">End Date</label>
              <VueDatePicker
                :model-value="endDate"
                @update:model-value="onEndDateChange"
                :time-config="{ enableTimePicker: false }"
                auto-apply
                :dark="true"
                format="dd/MM/yyyy"
                model-type="yyyy-MM-dd"
                no-today
                input-class-name="dp-dark-input"
              />
            </div>
            <div class="w-px h-[42px] bg-white/10 mx-1" />
            <div>
              <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">End Time</label>
              <VueDatePicker
                :model-value="endTimeObj"
                @update:model-value="onEndTimeChange"
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

          <!-- Model chips + availability -->
          <div class="flex-1 min-w-[200px]">
            <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Models</label>
            <div class="flex flex-wrap gap-2">
              <label
                v-for="model in configStore.models"
                :key="model"
                class="inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-medium
                       cursor-pointer transition-all select-none"
                :class="selectedModels.includes(model)
                  ? 'bg-blue-500/30 text-blue-300 ring-1 ring-blue-400/50'
                  : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-300'"
              >
                <input type="checkbox" :value="model" v-model="selectedModels" class="sr-only" />
                {{ model }}
              </label>
            </div>
            <p v-if="availabilitySummary" class="text-[11px] mt-1.5" :class="allSelectedAvailable ? 'text-emerald-400' : 'text-amber-400'">
              {{ availabilitySummary }}
            </p>
          </div>

          <!-- Action button -->
          <button
            @click="handleAction"
            :disabled="!canCompute"
            class="flex-shrink-0 h-[42px] px-5 rounded-lg font-semibold text-sm transition-all
                   flex items-center gap-2 self-end"
            :class="canCompute
              ? (allSelectedAvailable
                ? 'bg-blue-600 text-white hover:bg-blue-500 shadow-sm shadow-blue-500/30'
                : 'bg-amber-600 text-white hover:bg-amber-500 shadow-sm shadow-amber-500/30')
              : 'bg-white/10 text-gray-500 cursor-not-allowed'"
          >
            <svg v-if="loading || metricsLoading" class="animate-spin w-4 h-4" viewBox="0 0 24 24">
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
      </div>
    </div>

    <!-- ================================================================ -->
    <!-- MAIN CONTENT                                                      -->
    <!-- ================================================================ -->
    <div class="w-full max-w-full mx-auto px-6 py-6">

      <!-- Computing progress card (job submission) -->
      <div v-if="computing" class="flex items-center justify-center py-20">
        <div class="bg-white rounded-xl shadow-lg border border-gray-100 p-6 w-full max-w-md">
          <h2 class="text-lg font-bold text-gray-800 mb-4">Computing Predictions</h2>
          <div class="space-y-3 mb-5">
            <div v-for="model in Object.keys(computeStatus)" :key="model">
              <div class="flex items-center gap-3">
                <svg v-if="computeStatus[model].state === 'done'" class="w-5 h-5 text-emerald-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
                </svg>
                <svg v-else-if="computeStatus[model].state === 'error'" class="w-5 h-5 text-red-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
                <svg v-else-if="computeStatus[model].state === 'queued'" class="w-5 h-5 text-amber-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="10" />
                  <path stroke-linecap="round" d="M12 6v6l4 2" />
                </svg>
                <svg v-else class="animate-spin w-5 h-5 text-blue-500 flex-shrink-0" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
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

      <!-- Metrics loading spinner -->
      <div v-if="metricsLoading" class="flex items-center justify-center py-20">
        <div class="bg-white rounded-xl shadow-lg border border-gray-100 p-6 w-full max-w-md text-center">
          <svg class="animate-spin w-8 h-8 text-blue-500 mx-auto mb-4" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <p class="text-sm text-gray-600">Computing metrics... This may take a while for large date ranges.</p>
        </div>
      </div>

      <!-- Results tabs -->
      <div v-if="results && !metricsLoading && !computing">

        <!-- Tab bar -->
        <div class="flex gap-1 mb-6 border-b border-gray-200">
          <button
            v-for="tab in tabs"
            :key="tab.key"
            @click="activeTab = tab.key"
            class="px-5 py-2.5 text-sm font-semibold border-b-2 transition-colors"
            :class="activeTab === tab.key
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'"
          >
            {{ tab.label }}
          </button>
          <button
            @click="exportAll"
            class="ml-auto px-4 py-2 text-sm font-medium text-gray-600 hover:text-blue-600
                   border border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50
                   transition-colors flex items-center gap-1.5 mb-1"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 15V3" />
            </svg>
            Export All
          </button>
        </div>

        <!-- ============================================================ -->
        <!-- CSI TAB                                                       -->
        <!-- ============================================================ -->
        <div v-if="activeTab === 'csi'">

          <!-- Per-model detailed tables (collapsible) -->
          <div class="space-y-4 mb-8">
            <div v-for="model in results.models" :key="model">
              <div
                class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden"
              >
                <button
                  @click="toggleCollapse('csi-' + model)"
                  class="w-full px-5 py-3 bg-gray-50 border-b border-gray-100 flex items-center gap-3 hover:bg-gray-100 transition-colors"
                >
                  <svg
                    class="w-4 h-4 text-gray-400 transition-transform"
                    :class="{ 'rotate-90': !collapsed['csi-' + model] }"
                    fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"
                  >
                    <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
                  </svg>
                  <span class="text-sm font-bold text-gray-800">{{ model }}</span>
                  <span class="text-xs text-gray-400 ml-auto mr-2">CSI / POD / FAR</span>
                  <span
                    @click.stop="downloadModelCSI(model)"
                    class="p-1.5 rounded-lg text-gray-400 hover:text-blue-500 hover:bg-blue-50 transition-colors"
                    title="Download as CSV"
                  >
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 15V3" />
                    </svg>
                  </span>
                </button>
                <div v-if="!collapsed['csi-' + model]" class="p-4 space-y-4">
                  <div>
                    <h4 class="text-xs font-semibold text-gray-500 uppercase mb-2">CSI</h4>
                    <DataTable :data="results.csi[model]" />
                  </div>
                  <div>
                    <h4 class="text-xs font-semibold text-gray-500 uppercase mb-2">POD</h4>
                    <DataTable :data="results.pod[model]" />
                  </div>
                  <div>
                    <h4 class="text-xs font-semibold text-gray-500 uppercase mb-2">FAR</h4>
                    <DataTable :data="results.far[model]" />
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Overall Performance (ranked table) -->
          <div v-if="overallCsi" class="bg-white rounded-xl shadow-sm border border-gray-100 p-5 mb-8">
            <div class="flex items-center justify-between mb-3">
              <h3 class="text-sm font-bold text-gray-800">Overall Performance (Mean CSI by Threshold)</h3>
              <button
                @click="downloadOverallCSI"
                class="p-1.5 rounded-lg text-gray-400 hover:text-blue-500 hover:bg-blue-50 transition-colors"
                title="Download as CSV"
              >
                <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 15V3" />
                </svg>
              </button>
            </div>
            <div class="overflow-x-auto">
              <table class="min-w-full text-sm border-collapse">
                <thead>
                  <tr class="bg-gray-50">
                    <th class="px-3 py-2 text-left font-medium text-gray-600 border border-gray-200">Model</th>
                    <th
                      v-for="th in csiThresholds"
                      :key="th"
                      class="px-3 py-2 text-center font-medium text-gray-600 border border-gray-200"
                    >
                      {{ th }} mm/h
                    </th>
                    <th class="px-3 py-2 text-center font-semibold text-gray-700 border border-gray-200 bg-gray-100">Mean CSI</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="row in overallCsi" :key="row.model" class="hover:bg-gray-50">
                    <td class="px-3 py-2 font-medium text-gray-700 border border-gray-200 bg-gray-50">{{ row.model }}</td>
                    <td
                      v-for="th in csiThresholds"
                      :key="th"
                      class="px-3 py-2 text-center border border-gray-200"
                      :class="csiCellClass(row.thresholds[th])"
                    >
                      {{ formatVal(row.thresholds[th]) }}
                    </td>
                    <td
                      class="px-3 py-2 text-center font-semibold border border-gray-200"
                      :class="csiCellClass(row.mean)"
                    >
                      {{ formatVal(row.mean) }}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- CSI vs Lead Time charts (one per threshold) -->
          <div v-if="csiChartDatasets" class="space-y-6">
            <div v-for="th in csiThresholds" :key="th" class="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
              <div class="flex items-center justify-between mb-3">
                <h3 class="text-sm font-bold text-gray-800">CSI vs Lead Time — Threshold {{ th }} mm/h</h3>
                <button
                  @click="downloadCSIChart(th)"
                  class="p-1.5 rounded-lg text-gray-400 hover:text-blue-500 hover:bg-blue-50 transition-colors"
                  title="Download as PNG"
                >
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 15V3" />
                  </svg>
                </button>
              </div>
              <div class="h-[300px]">
                <Line :ref="el => { if (el) csiChartRefs[th] = el }" :data="csiChartData(th)" :options="csiChartOptions" />
              </div>
            </div>
          </div>
        </div>

        <!-- ============================================================ -->
        <!-- FSS TAB                                                       -->
        <!-- ============================================================ -->
        <div v-if="activeTab === 'fss'">
          <div v-if="results.fss && Object.keys(results.fss).length > 0" class="space-y-6">
            <div v-for="(data, threshold) in results.fss" :key="threshold" class="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
              <div class="flex items-center justify-between mb-3">
                <h3 class="text-sm font-bold text-gray-800">Threshold: {{ threshold }} mm/h</h3>
                <button
                  @click="downloadFSSTable(threshold)"
                  class="p-1.5 rounded-lg text-gray-400 hover:text-blue-500 hover:bg-blue-50 transition-colors"
                  title="Download as CSV"
                >
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 15V3" />
                  </svg>
                </button>
              </div>
              <div class="overflow-x-auto">
                <table class="min-w-full text-sm border-collapse">
                  <thead>
                    <tr class="bg-gray-50">
                      <th class="px-3 py-2 text-left font-medium text-gray-600 border border-gray-200">Window Size</th>
                      <th
                        v-for="model in fssModels(data)"
                        :key="model"
                        class="px-3 py-2 text-center font-medium text-gray-600 border border-gray-200"
                      >
                        {{ model }}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="ws in fssWindowSizes(data)" :key="ws" class="hover:bg-gray-50">
                      <td class="px-3 py-2 font-medium text-gray-700 border border-gray-200 bg-gray-50">{{ ws }} px</td>
                      <td
                        v-for="model in fssModels(data)"
                        :key="model"
                        class="px-3 py-2 text-center border border-gray-200"
                        :class="csiCellClass(data[model]?.[ws])"
                      >
                        {{ formatVal(data[model]?.[ws]) }}
                      </td>
                    </tr>
                    <!-- Average row -->
                    <tr class="bg-gray-50 font-semibold">
                      <td class="px-3 py-2 font-semibold text-gray-700 border border-gray-200">Average</td>
                      <td
                        v-for="model in fssModels(data)"
                        :key="model"
                        class="px-3 py-2 text-center border border-gray-200"
                        :class="csiCellClass(fssAverage(data, model))"
                      >
                        {{ formatVal(fssAverage(data, model)) }}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div v-else class="text-center py-12 text-gray-400 text-sm">No FSS data available.</div>
        </div>

        <!-- ============================================================ -->
        <!-- RMSE TAB                                                      -->
        <!-- ============================================================ -->
        <div v-if="activeTab === 'rmse'">
          <div v-if="results.regression && Object.keys(results.regression).length > 0">

            <!-- NMSE Table -->
            <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5 mb-6">
              <div class="flex items-center justify-between mb-3">
                <h3 class="text-sm font-bold text-gray-800">NMSE (Normalized Mean Square Error)</h3>
                <button
                  @click="downloadNMSETable"
                  class="p-1.5 rounded-lg text-gray-400 hover:text-blue-500 hover:bg-blue-50 transition-colors"
                  title="Download as CSV"
                >
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 15V3" />
                  </svg>
                </button>
              </div>
              <p class="text-xs text-gray-500 mb-3">Lower is better. Values closer to 0 indicate better prediction accuracy.</p>
              <div class="overflow-x-auto">
                <table class="min-w-full text-sm border-collapse">
                  <thead>
                    <tr class="bg-gray-50">
                      <th class="px-3 py-2 text-left font-medium text-gray-600 border border-gray-200">Model</th>
                      <th
                        v-for="lt in leadTimeLabels"
                        :key="lt"
                        class="px-3 py-2 text-center font-medium text-gray-600 border border-gray-200"
                      >
                        {{ lt }} min
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="model in regressionModels" :key="model" class="hover:bg-gray-50">
                      <td class="px-3 py-2 font-medium text-gray-700 border border-gray-200 bg-gray-50">{{ model }}</td>
                      <td
                        v-for="lt in leadTimeLabels"
                        :key="lt"
                        class="px-3 py-2 text-center border border-gray-200"
                        :class="nmseCellClass(results.regression[model]?.nmse?.[lt])"
                      >
                        {{ formatVal(results.regression[model]?.nmse?.[lt]) }}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Beta Table -->
            <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5 mb-6">
              <div class="flex items-center justify-between mb-3">
                <h3 class="text-sm font-bold text-gray-800">Beta Coefficient (Regression Slope)</h3>
                <button
                  @click="downloadBetaTable"
                  class="p-1.5 rounded-lg text-gray-400 hover:text-blue-500 hover:bg-blue-50 transition-colors"
                  title="Download as CSV"
                >
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 15V3" />
                  </svg>
                </button>
              </div>
              <p class="text-xs text-gray-500 mb-3">Ideal value is 1.0. Values closer to 1.0 indicate better calibration.</p>
              <div class="overflow-x-auto">
                <table class="min-w-full text-sm border-collapse">
                  <thead>
                    <tr class="bg-gray-50">
                      <th class="px-3 py-2 text-left font-medium text-gray-600 border border-gray-200">Model</th>
                      <th
                        v-for="lt in leadTimeLabels"
                        :key="lt"
                        class="px-3 py-2 text-center font-medium text-gray-600 border border-gray-200"
                      >
                        {{ lt }} min
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="model in regressionModels" :key="model" class="hover:bg-gray-50">
                      <td class="px-3 py-2 font-medium text-gray-700 border border-gray-200 bg-gray-50">{{ model }}</td>
                      <td
                        v-for="lt in leadTimeLabels"
                        :key="lt"
                        class="px-3 py-2 text-center border border-gray-200"
                        :class="betaCellClass(results.regression[model]?.beta?.[lt])"
                      >
                        {{ formatVal(results.regression[model]?.beta?.[lt]) }}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- NMSE vs Lead Time chart -->
            <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5 mb-6">
              <div class="flex items-center justify-between mb-3">
                <h3 class="text-sm font-bold text-gray-800">NMSE vs Lead Time</h3>
                <button
                  @click="downloadNMSEChart"
                  class="p-1.5 rounded-lg text-gray-400 hover:text-blue-500 hover:bg-blue-50 transition-colors"
                  title="Download as PNG"
                >
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 15V3" />
                  </svg>
                </button>
              </div>
              <div class="h-[300px]">
                <Line ref="nmseChartRef" :data="nmseChartData" :options="nmseChartOptions" />
              </div>
            </div>

            <!-- Beta vs Lead Time chart -->
            <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
              <div class="flex items-center justify-between mb-3">
                <h3 class="text-sm font-bold text-gray-800">Beta Coefficient vs Lead Time</h3>
                <button
                  @click="downloadBetaChart"
                  class="p-1.5 rounded-lg text-gray-400 hover:text-blue-500 hover:bg-blue-50 transition-colors"
                  title="Download as PNG"
                >
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 15V3" />
                  </svg>
                </button>
              </div>
              <div class="h-[300px]">
                <Line ref="betaChartRef" :data="betaChartData" :options="betaChartOptions" />
              </div>
            </div>
          </div>
          <div v-else class="text-center py-12 text-gray-400 text-sm">No regression data available.</div>
        </div>
      </div>

      <!-- Empty state -->
      <div v-if="!results && !loading && !computing && !metricsLoading && !error" class="text-center py-20">
        <svg class="mx-auto w-16 h-16 text-gray-300 mb-4" fill="none" stroke="currentColor" stroke-width="1" viewBox="0 0 24 24">
          <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
        <p class="text-gray-400 text-sm">Select models and a date range, then click <strong>Compute Metrics</strong></p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, reactive, watch, onBeforeUnmount } from 'vue'
import { VueDatePicker } from '@vuepic/vue-datepicker'
import '@vuepic/vue-datepicker/dist/main.css'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'
import { Line } from 'vue-chartjs'
import api from '../api.js'
import { useConfigStore } from '../stores/config.js'
import DataTable from '../components/DataTable.vue'

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)

const configStore = useConfigStore()

// ---------------------------------------------------------------------------
// Chart refs (for PNG export)
// ---------------------------------------------------------------------------
const csiChartRefs = reactive({})
const nmseChartRef = ref(null)
const betaChartRef = ref(null)

// ---------------------------------------------------------------------------
// Date/time state
// ---------------------------------------------------------------------------
const startDateTime = ref('')
const endDateTime = ref('')

// Parse date/time parts
function parseParts(isoStr) {
  if (!isoStr || !isoStr.includes('T')) return { date: '', hour: '00', minute: '00' }
  const [date, time] = isoStr.split('T')
  const [hour, minute] = (time || '00:00').split(':')
  return { date, hour: hour || '00', minute: minute || '00' }
}

const startDate = computed(() => parseParts(startDateTime.value).date || null)
const endDate = computed(() => parseParts(endDateTime.value).date || null)

const startTimeObj = computed(() => {
  const p = parseParts(startDateTime.value)
  return { hours: parseInt(p.hour) || 0, minutes: parseInt(p.minute) || 0, seconds: 0 }
})
const endTimeObj = computed(() => {
  const p = parseParts(endDateTime.value)
  return { hours: parseInt(p.hour) || 0, minutes: parseInt(p.minute) || 0, seconds: 0 }
})

function buildDT(date, hour, minute) {
  if (!date) return ''
  return `${date}T${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`
}

function onStartDateChange(val) {
  const ds = typeof val === 'string' ? val : ''
  if (ds) {
    const p = parseParts(startDateTime.value)
    startDateTime.value = buildDT(ds, p.hour, p.minute)
  }
}
function onStartTimeChange(val) {
  if (val && val.hours !== undefined) {
    const p = parseParts(startDateTime.value)
    startDateTime.value = buildDT(p.date || new Date().toISOString().split('T')[0], val.hours, val.minutes)
  }
}
function onEndDateChange(val) {
  const ds = typeof val === 'string' ? val : ''
  if (ds) {
    const p = parseParts(endDateTime.value)
    endDateTime.value = buildDT(ds, p.hour, p.minute)
  }
}
function onEndTimeChange(val) {
  if (val && val.hours !== undefined) {
    const p = parseParts(endDateTime.value)
    endDateTime.value = buildDT(p.date || new Date().toISOString().split('T')[0], val.hours, val.minutes)
  }
}

function formatDateDisplay(val) {
  if (!val?.includes('T')) return ''
  const [date, time] = val.split('T')
  const [y, m, d] = date.split('-')
  return `${d}/${m}/${y} ${time}`
}

// ---------------------------------------------------------------------------
// Model selection + availability
// ---------------------------------------------------------------------------
const selectedModels = ref([])
const availabilityMap = ref({})  // { model: { all_exist, missing_count, existing_count } }
let checkAbort = null

// Auto-check availability when dates or models change
watch([startDateTime, endDateTime, selectedModels], async () => {
  if (!startDateTime.value || !endDateTime.value || selectedModels.value.length === 0) {
    availabilityMap.value = {}
    return
  }
  if (!startDateTime.value.includes('T') || !endDateTime.value.includes('T')) return

  if (checkAbort) checkAbort.cancelled = true
  const thisCheck = { cancelled: false }
  checkAbort = thisCheck

  const results = await Promise.allSettled(
    selectedModels.value.map(model =>
      api.checkPredictions(model, startDateTime.value, endDateTime.value)
    )
  )

  if (thisCheck.cancelled) return

  const map = {}
  results.forEach((r, i) => {
    const model = selectedModels.value[i]
    if (r.status === 'fulfilled') {
      map[model] = r.value
    } else {
      map[model] = { all_exist: false, existing_count: 0, missing_count: -1 }
    }
  })
  availabilityMap.value = map
}, { deep: true })

const availabilitySummary = computed(() => {
  if (selectedModels.value.length === 0 || Object.keys(availabilityMap.value).length === 0) return ''
  const available = selectedModels.value.filter(m => availabilityMap.value[m]?.all_exist)
  const missing = selectedModels.value.filter(m => availabilityMap.value[m] && !availabilityMap.value[m].all_exist)
  if (missing.length === 0) return `All ${available.length} models have predictions`
  if (available.length === 0) return `No predictions found for selected models`
  return `${available.length} available, ${missing.length} missing: ${missing.join(', ')}`
})

const allSelectedAvailable = computed(() => {
  if (selectedModels.value.length === 0) return true
  return selectedModels.value.every(m => availabilityMap.value[m]?.all_exist)
})

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const loading = ref(false)
const computing = ref(false)
const metricsLoading = ref(false)
const results = ref(null)
const error = ref(null)
const activeTab = ref('csi')
const collapsed = reactive({})

const tabs = [
  { key: 'csi', label: 'CSI / POD / FAR' },
  { key: 'fss', label: 'FSS' },
  { key: 'rmse', label: 'RMSE' },
]

const canCompute = computed(() =>
  selectedModels.value.length > 0 &&
  startDateTime.value && startDateTime.value.length >= 16 &&
  endDateTime.value && endDateTime.value.length >= 16 &&
  !loading.value && !computing.value && !metricsLoading.value
)

const buttonLabel = computed(() => {
  if (metricsLoading.value) return 'Computing Metrics...'
  if (loading.value) return 'Loading...'
  if (computing.value) return 'Computing...'
  if (!allSelectedAvailable.value && selectedModels.value.length > 0) return 'Compute Predictions'
  return 'Compute Metrics'
})

function toggleCollapse(key) {
  collapsed[key] = !collapsed[key]
}

// ---------------------------------------------------------------------------
// Computing state (job submission)
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

async function computeMissing(missingModels) {
  computing.value = true
  for (const k of Object.keys(computeStatus)) delete computeStatus[k]
  const abort = { cancelled: false }
  pollAbort = abort

  for (const model of missingModels) {
    computeStatus[model] = { state: 'submitting', jobId: null }
  }

  const results = await Promise.allSettled(
    missingModels.map(m =>
      api.submitJob(m, startDateTime.value, endDateTime.value)
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
 * to find the prediction files. If not found, we declare it failed and
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
          failedChecks.delete(model)
        } else {
          // Job left queue — check if prediction files appeared
          const pred = await api.checkPredictions(model, startDateTime.value, endDateTime.value)
          if (pred.all_exist) {
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
  metricsLoading.value = false
  for (const k of Object.keys(computeStatus)) delete computeStatus[k]
  for (const k of Object.keys(expandedErrorLogs)) delete expandedErrorLogs[k]
}

onBeforeUnmount(() => {
  if (pollAbort) pollAbort.cancelled = true
  if (checkAbort) checkAbort.cancelled = true
})

// ---------------------------------------------------------------------------
// Main action
// ---------------------------------------------------------------------------
async function handleAction() {
  error.value = null
  results.value = null

  // Check which models are missing predictions
  const missing = selectedModels.value.filter(m => availabilityMap.value[m] && !availabilityMap.value[m].all_exist)

  if (missing.length > 0) {
    // Submit jobs for missing models, then compute metrics
    await computeMissing(missing)
    if (pollAbort?.cancelled) return

    // Re-check availability
    const recheck = await Promise.allSettled(
      selectedModels.value.map(model =>
        api.checkPredictions(model, startDateTime.value, endDateTime.value)
      )
    )
    const map = {}
    recheck.forEach((r, i) => {
      const model = selectedModels.value[i]
      if (r.status === 'fulfilled') map[model] = r.value
    })
    availabilityMap.value = map
  }

  // Now compute metrics
  await computeMetrics()
}

async function computeMetrics() {
  error.value = null
  metricsLoading.value = true
  results.value = null

  try {
    results.value = await api.computeMetrics(
      selectedModels.value,
      startDateTime.value,
      endDateTime.value
    )
    // Default-collapse per-model sections
    for (const model of results.value.models) {
      if (collapsed['csi-' + model] === undefined) {
        collapsed['csi-' + model] = true
      }
    }
  } catch (e) {
    error.value = `Failed to compute metrics: ${e.message}`
  } finally {
    metricsLoading.value = false
  }
}

// ---------------------------------------------------------------------------
// CSI helpers
// ---------------------------------------------------------------------------
const leadTimeLabels = ['5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60']

const csiThresholds = computed(() => {
  if (!results.value?.csi) return []
  const firstModel = Object.keys(results.value.csi)[0]
  if (!firstModel) return []
  const data = results.value.csi[firstModel]
  const firstCol = Object.keys(data)[0]
  return firstCol ? Object.keys(data[firstCol]) : []
})

// Overall CSI: for each model, average CSI across all lead times per threshold, then compute mean
const overallCsi = computed(() => {
  if (!results.value?.csi || !csiThresholds.value.length) return null

  const rows = []
  for (const model of results.value.models) {
    if (!results.value.csi[model]) continue
    const data = results.value.csi[model]
    const thresholdAvgs = {}
    const allVals = []
    for (const th of csiThresholds.value) {
      // Average across all lead times for this threshold
      const vals = Object.values(data).map(col => col[th]).filter(v => v != null)
      const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0
      thresholdAvgs[th] = avg
      allVals.push(avg)
    }
    rows.push({
      model,
      thresholds: thresholdAvgs,
      mean: allVals.length ? allVals.reduce((a, b) => a + b, 0) / allVals.length : 0,
    })
  }
  // Sort by mean CSI descending
  rows.sort((a, b) => b.mean - a.mean)
  return rows
})

function formatVal(val) {
  if (val === null || val === undefined) return '-'
  if (typeof val === 'number') return val.toFixed(3)
  return val
}

function csiCellClass(val) {
  if (val === null || val === undefined || typeof val !== 'number') return 'text-gray-300'
  if (val >= 0.7) return 'bg-green-100 text-green-800'
  if (val >= 0.4) return 'bg-yellow-50 text-yellow-700'
  if (val > 0) return 'bg-red-50 text-red-700'
  return 'text-gray-400'
}

function nmseCellClass(val) {
  if (val === null || val === undefined || typeof val !== 'number') return 'text-gray-300'
  if (val <= 0.3) return 'bg-green-100 text-green-800'
  if (val <= 0.7) return 'bg-yellow-50 text-yellow-700'
  return 'bg-red-50 text-red-700'
}

function betaCellClass(val) {
  if (val === null || val === undefined || typeof val !== 'number') return 'text-gray-300'
  const dist = Math.abs(val - 1.0)
  if (dist <= 0.1) return 'bg-green-100 text-green-800'
  if (dist <= 0.3) return 'bg-yellow-50 text-yellow-700'
  return 'bg-red-50 text-red-700'
}

// ---------------------------------------------------------------------------
// FSS helpers
// ---------------------------------------------------------------------------
function fssModels(data) {
  return Object.keys(data)
}

function fssWindowSizes(data) {
  const firstModel = Object.keys(data)[0]
  return firstModel ? Object.keys(data[firstModel]) : []
}

function fssAverage(data, model) {
  const vals = Object.values(data[model] || {}).filter(v => v != null && typeof v === 'number')
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null
}

// ---------------------------------------------------------------------------
// Regression helpers
// ---------------------------------------------------------------------------
const regressionModels = computed(() => {
  if (!results.value?.regression) return []
  return Object.keys(results.value.regression)
})

// ---------------------------------------------------------------------------
// Chart.js - model colors
// ---------------------------------------------------------------------------
const MODEL_COLORS = [
  '#3b82f6', // blue
  '#ef4444', // red
  '#10b981', // emerald
  '#f59e0b', // amber
  '#8b5cf6', // violet
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#84cc16', // lime
]

function modelColor(idx) {
  return MODEL_COLORS[idx % MODEL_COLORS.length]
}

// ---------------------------------------------------------------------------
// CSI Charts
// ---------------------------------------------------------------------------
const csiChartDatasets = computed(() => {
  if (!results.value?.csi) return null
  return true
})

function csiChartData(threshold) {
  const datasets = results.value.models.map((model, idx) => {
    const data = results.value.csi[model]
    if (!data) return null
    const values = leadTimeLabels.map(lt => {
      const col = data[lt]
      return col ? (col[threshold] ?? null) : null
    })
    return {
      label: model,
      data: values,
      borderColor: modelColor(idx),
      backgroundColor: modelColor(idx) + '20',
      tension: 0.3,
      pointRadius: 4,
      pointHoverRadius: 6,
    }
  }).filter(Boolean)

  return {
    labels: leadTimeLabels.map(lt => lt + ' min'),
    datasets,
  }
}

const csiChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  scales: {
    y: {
      beginAtZero: true,
      max: 1,
      title: { display: true, text: 'CSI' },
    },
    x: {
      title: { display: true, text: 'Lead Time' },
    },
  },
  plugins: {
    legend: { position: 'top' },
  },
}

// ---------------------------------------------------------------------------
// NMSE Chart
// ---------------------------------------------------------------------------
const nmseChartData = computed(() => {
  if (!results.value?.regression) return { labels: [], datasets: [] }

  const datasets = regressionModels.value.map((model, idx) => {
    const nmse = results.value.regression[model]?.nmse || {}
    const values = leadTimeLabels.map(lt => nmse[lt] ?? null)
    return {
      label: model,
      data: values,
      borderColor: modelColor(idx),
      backgroundColor: modelColor(idx) + '20',
      tension: 0.3,
      pointRadius: 4,
      pointHoverRadius: 6,
    }
  })

  return {
    labels: leadTimeLabels.map(lt => lt + ' min'),
    datasets,
  }
})

const nmseChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  scales: {
    y: {
      beginAtZero: true,
      title: { display: true, text: 'NMSE' },
    },
    x: {
      title: { display: true, text: 'Lead Time' },
    },
  },
  plugins: {
    legend: { position: 'top' },
  },
}

// ---------------------------------------------------------------------------
// Beta Chart (with reference line at beta=1)
// ---------------------------------------------------------------------------
const betaChartData = computed(() => {
  if (!results.value?.regression) return { labels: [], datasets: [] }

  const datasets = regressionModels.value.map((model, idx) => {
    const beta = results.value.regression[model]?.beta || {}
    const values = leadTimeLabels.map(lt => beta[lt] ?? null)
    return {
      label: model,
      data: values,
      borderColor: modelColor(idx),
      backgroundColor: modelColor(idx) + '20',
      tension: 0.3,
      pointRadius: 4,
      pointHoverRadius: 6,
    }
  })

  // Add reference line at beta = 1
  datasets.push({
    label: 'Ideal (beta=1)',
    data: leadTimeLabels.map(() => 1.0),
    borderColor: '#9ca3af',
    borderDash: [6, 4],
    pointRadius: 0,
    borderWidth: 2,
  })

  return {
    labels: leadTimeLabels.map(lt => lt + ' min'),
    datasets,
  }
})

const betaChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  scales: {
    y: {
      title: { display: true, text: 'Beta' },
    },
    x: {
      title: { display: true, text: 'Lead Time' },
    },
  },
  plugins: {
    legend: { position: 'top' },
  },
}

// ---------------------------------------------------------------------------
// Export / Download helpers
// ---------------------------------------------------------------------------
function filenameDateRange() {
  const fmt = (dt) => dt.replace('T', '_').replace(':', '')
  return `${fmt(startDateTime.value)}_${fmt(endDateTime.value)}`
}

function downloadFile(content, filename, mimeType = 'text/csv') {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

function downloadChartPNG(chartComp, filename) {
  if (!chartComp?.chart) return
  const url = chartComp.chart.toBase64Image()
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

// Per-model CSI/POD/FAR CSV
function downloadModelCSI(model) {
  if (!results.value?.csi?.[model]) return
  const range = filenameDateRange()
  let csv = ''
  for (const metric of ['csi', 'pod', 'far']) {
    const data = results.value[metric]?.[model]
    if (!data) continue
    csv += `${metric.toUpperCase()} - ${model}\n`
    const thresholds = Object.keys(Object.values(data)[0] || {})
    csv += `Lead Time,${thresholds.map(t => t + ' mm/h').join(',')}\n`
    for (const lt of leadTimeLabels) {
      const row = [lt + ' min']
      for (const th of thresholds) {
        row.push(formatVal(data[lt]?.[th]))
      }
      csv += row.join(',') + '\n'
    }
    csv += '\n'
  }
  downloadFile(csv, `CSI_POD_FAR_${model}_${range}.csv`)
}

// Overall CSI CSV
function downloadOverallCSI() {
  if (!overallCsi.value) return
  const range = filenameDateRange()
  let csv = 'Model,' + csiThresholds.value.map(t => t + ' mm/h').join(',') + ',Mean CSI\n'
  for (const row of overallCsi.value) {
    const vals = [row.model]
    for (const th of csiThresholds.value) {
      vals.push(formatVal(row.thresholds[th]))
    }
    vals.push(formatVal(row.mean))
    csv += vals.join(',') + '\n'
  }
  downloadFile(csv, `CSI_overall_${range}.csv`)
}

// CSI chart PNG
function downloadCSIChart(threshold) {
  const chartComp = csiChartRefs[threshold]
  if (!chartComp) return
  const range = filenameDateRange()
  downloadChartPNG(chartComp, `CSI_chart_${threshold}mmh_${range}.png`)
}

// FSS table CSV
function downloadFSSTable(threshold) {
  const data = results.value?.fss?.[threshold]
  if (!data) return
  const range = filenameDateRange()
  const models = fssModels(data)
  const windowSizes = fssWindowSizes(data)
  let csv = `FSS - Threshold ${threshold} mm/h\n`
  csv += 'Window Size,' + models.join(',') + '\n'
  for (const ws of windowSizes) {
    const row = [ws + ' px']
    for (const model of models) {
      row.push(formatVal(data[model]?.[ws]))
    }
    csv += row.join(',') + '\n'
  }
  const avgRow = ['Average']
  for (const model of models) {
    avgRow.push(formatVal(fssAverage(data, model)))
  }
  csv += avgRow.join(',') + '\n'
  downloadFile(csv, `FSS_${threshold}mmh_${range}.csv`)
}

// NMSE table CSV
function downloadNMSETable() {
  if (!results.value?.regression) return
  const range = filenameDateRange()
  let csv = 'Model,' + leadTimeLabels.map(lt => lt + ' min').join(',') + '\n'
  for (const model of regressionModels.value) {
    const row = [model]
    for (const lt of leadTimeLabels) {
      row.push(formatVal(results.value.regression[model]?.nmse?.[lt]))
    }
    csv += row.join(',') + '\n'
  }
  downloadFile(csv, `NMSE_${range}.csv`)
}

// Beta table CSV
function downloadBetaTable() {
  if (!results.value?.regression) return
  const range = filenameDateRange()
  let csv = 'Model,' + leadTimeLabels.map(lt => lt + ' min').join(',') + '\n'
  for (const model of regressionModels.value) {
    const row = [model]
    for (const lt of leadTimeLabels) {
      row.push(formatVal(results.value.regression[model]?.beta?.[lt]))
    }
    csv += row.join(',') + '\n'
  }
  downloadFile(csv, `Beta_${range}.csv`)
}

// NMSE chart PNG
function downloadNMSEChart() {
  const range = filenameDateRange()
  downloadChartPNG(nmseChartRef.value, `NMSE_chart_${range}.png`)
}

// Beta chart PNG
function downloadBetaChart() {
  const range = filenameDateRange()
  downloadChartPNG(betaChartRef.value, `Beta_chart_${range}.png`)
}

// Export All: combined CSV + all chart PNGs
async function exportAll() {
  if (!results.value) return
  const range = filenameDateRange()
  let csv = ''

  // CSI/POD/FAR per model
  if (results.value.csi) {
    for (const model of results.value.models) {
      for (const metric of ['csi', 'pod', 'far']) {
        const data = results.value[metric]?.[model]
        if (!data) continue
        csv += `${metric.toUpperCase()} - ${model}\n`
        const thresholds = Object.keys(Object.values(data)[0] || {})
        csv += `Lead Time,${thresholds.map(t => t + ' mm/h').join(',')}\n`
        for (const lt of leadTimeLabels) {
          const row = [lt + ' min']
          for (const th of thresholds) {
            row.push(formatVal(data[lt]?.[th]))
          }
          csv += row.join(',') + '\n'
        }
        csv += '\n'
      }
    }
  }

  // Overall CSI
  if (overallCsi.value) {
    csv += 'Overall CSI\n'
    csv += 'Model,' + csiThresholds.value.map(t => t + ' mm/h').join(',') + ',Mean CSI\n'
    for (const row of overallCsi.value) {
      const vals = [row.model]
      for (const th of csiThresholds.value) {
        vals.push(formatVal(row.thresholds[th]))
      }
      vals.push(formatVal(row.mean))
      csv += vals.join(',') + '\n'
    }
    csv += '\n'
  }

  // FSS
  if (results.value.fss) {
    for (const [threshold, data] of Object.entries(results.value.fss)) {
      const models = fssModels(data)
      const windowSizes = fssWindowSizes(data)
      csv += `FSS - Threshold ${threshold} mm/h\n`
      csv += 'Window Size,' + models.join(',') + '\n'
      for (const ws of windowSizes) {
        const row = [ws + ' px']
        for (const model of models) {
          row.push(formatVal(data[model]?.[ws]))
        }
        csv += row.join(',') + '\n'
      }
      const avgRow = ['Average']
      for (const model of models) {
        avgRow.push(formatVal(fssAverage(data, model)))
      }
      csv += avgRow.join(',') + '\n\n'
    }
  }

  // NMSE + Beta
  if (results.value.regression) {
    csv += 'NMSE\n'
    csv += 'Model,' + leadTimeLabels.map(lt => lt + ' min').join(',') + '\n'
    for (const model of regressionModels.value) {
      const row = [model]
      for (const lt of leadTimeLabels) {
        row.push(formatVal(results.value.regression[model]?.nmse?.[lt]))
      }
      csv += row.join(',') + '\n'
    }
    csv += '\n'

    csv += 'Beta\n'
    csv += 'Model,' + leadTimeLabels.map(lt => lt + ' min').join(',') + '\n'
    for (const model of regressionModels.value) {
      const row = [model]
      for (const lt of leadTimeLabels) {
        row.push(formatVal(results.value.regression[model]?.beta?.[lt]))
      }
      csv += row.join(',') + '\n'
    }
    csv += '\n'
  }

  downloadFile(csv, `Metrics_all_${range}.csv`)

  // Download chart PNGs with small delays to avoid browser blocking
  const chartDownloads = []
  if (csiChartDatasets.value) {
    for (const th of csiThresholds.value) {
      chartDownloads.push(() => downloadCSIChart(th))
    }
  }
  if (results.value.regression) {
    chartDownloads.push(() => downloadNMSEChart())
    chartDownloads.push(() => downloadBetaChart())
  }
  for (const fn of chartDownloads) {
    await new Promise(r => setTimeout(r, 150))
    fn()
  }
}
</script>

<style scoped>
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
</style>
