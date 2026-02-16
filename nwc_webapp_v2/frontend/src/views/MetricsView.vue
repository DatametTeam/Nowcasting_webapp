<!--
  MetricsView.vue — CSI / POD / FAR / FSS metrics analysis.

  This replaces csi_analysis.py in Streamlit. The workflow is:
  1. Select one or more models + date range
  2. Click "Compute Metrics" → backend runs compute_csi_for_models()
  3. Display results as tables (CSI by threshold × lead time)

  NOTE: The computation can be slow for large date ranges.
  The loading spinner keeps the page responsive while waiting.
-->
<template>
  <div class="p-6 max-w-6xl mx-auto">
    <h1 class="text-2xl font-bold text-gray-800 mb-6">Metrics Analysis</h1>

    <!-- Configuration -->
    <div class="bg-white rounded-lg shadow p-6 mb-6">
      <h2 class="text-lg font-semibold text-gray-700 mb-4">Configuration</h2>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <!-- Multi-model selector (checkboxes) -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Models</label>
          <div class="space-y-2">
            <!--
              v-model with checkboxes + array: Vue automatically adds/removes
              values from the selectedModels array when checkboxes are toggled.
              This is equivalent to st.multiselect() in Streamlit.
            -->
            <label
              v-for="model in configStore.models"
              :key="model"
              class="flex items-center gap-2 text-sm"
            >
              <input
                type="checkbox"
                :value="model"
                v-model="selectedModels"
                class="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              {{ model }}
            </label>
          </div>
        </div>

        <DateTimeInput v-model="startDateTime" label="Start Date/Time" />
        <DateTimeInput v-model="endDateTime" label="End Date/Time" />
      </div>

      <div class="flex gap-3 mt-5">
        <button
          @click="computeMetrics"
          :disabled="!canCompute"
          class="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md
                 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed
                 transition-colors"
        >
          {{ computing ? 'Computing...' : 'Compute Metrics' }}
        </button>
      </div>
    </div>

    <!-- Loading -->
    <div v-if="computing" class="bg-white rounded-lg shadow p-6 mb-6">
      <StatusBadge status="loading" text="Computing metrics... This may take a while for large date ranges." />
    </div>

    <!-- Results -->
    <div v-if="results" class="space-y-6">
      <!-- CSI Tables -->
      <MetricSection title="CSI (Critical Success Index)" :data="results.csi" :models="results.models" />
      <MetricSection title="POD (Probability of Detection)" :data="results.pod" :models="results.models" />
      <MetricSection title="FAR (False Alarm Ratio)" :data="results.far" :models="results.models" />

      <!-- FSS section (different structure) -->
      <div v-if="Object.keys(results.fss).length > 0" class="bg-white rounded-lg shadow p-6">
        <h2 class="text-lg font-semibold text-gray-700 mb-4">FSS (Fractions Skill Score)</h2>
        <div v-for="(data, threshold) in results.fss" :key="threshold" class="mb-4">
          <h3 class="text-sm font-medium text-gray-600 mb-2">Threshold: {{ threshold }} mm/h</h3>
          <DataTable :data="data" />
        </div>
      </div>
    </div>

    <!-- Error -->
    <div v-if="error" class="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
      <p class="text-sm text-red-700">{{ error }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import api from '../api.js'
import { useConfigStore } from '../stores/config.js'
import DateTimeInput from '../components/DateTimeInput.vue'
import StatusBadge from '../components/StatusBadge.vue'
import MetricSection from '../components/MetricSection.vue'
import DataTable from '../components/DataTable.vue'

const configStore = useConfigStore()

// Form state
const selectedModels = ref([])
const startDateTime = ref('')
const endDateTime = ref('')

// Computation state
const computing = ref(false)
const results = ref(null)
const error = ref(null)

const canCompute = computed(() => {
  return selectedModels.value.length > 0 && startDateTime.value && endDateTime.value && !computing.value
})

async function computeMetrics() {
  error.value = null
  computing.value = true
  results.value = null

  try {
    results.value = await api.computeMetrics(
      selectedModels.value,
      startDateTime.value,
      endDateTime.value
    )
  } catch (e) {
    error.value = `Failed to compute metrics: ${e.message}`
  } finally {
    computing.value = false
  }
}
</script>