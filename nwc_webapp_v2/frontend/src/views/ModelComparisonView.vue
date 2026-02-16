<!--
  ModelComparisonView.vue — Side-by-side model comparison.

  This replaces model_comparison.py in Streamlit. The workflow is:
  1. Select 2+ models + single datetime
  2. Check which predictions exist
  3. Display predictions side by side with a shared lead time slider

  In Streamlit, this page had a complex JavaScript hack for synchronized
  zoom/pan between maps. Here, we'll start with simple side-by-side images
  and can later add Leaflet-based synchronized maps.
-->
<template>
  <div class="p-6 max-w-7xl mx-auto">
    <h1 class="text-2xl font-bold text-gray-800 mb-6">Model Comparison</h1>

    <!-- Configuration -->
    <div class="bg-white rounded-lg shadow p-6 mb-6">
      <h2 class="text-lg font-semibold text-gray-700 mb-4">Configuration</h2>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- Multi-model selector (checkboxes) -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Models to Compare (select 2+)
          </label>
          <div class="space-y-2">
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

        <DateTimeInput v-model="selectedDateTime" label="Date/Time" />
      </div>

      <div class="flex gap-3 mt-5">
        <button
          @click="checkAndLoad"
          :disabled="!canCheck"
          class="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md
                 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed
                 transition-colors"
        >
          {{ checking ? 'Checking...' : 'Load Comparison' }}
        </button>
      </div>
    </div>

    <!-- Prediction availability -->
    <div v-if="checkResults.length > 0" class="bg-white rounded-lg shadow p-6 mb-6">
      <h2 class="text-lg font-semibold text-gray-700 mb-3">Data Availability</h2>
      <div class="flex flex-wrap gap-3">
        <StatusBadge
          v-for="result in checkResults"
          :key="result.model"
          :status="result.exists ? 'success' : 'error'"
          :text="`${result.model}: ${result.exists ? 'Available' : 'Not found'}`"
        />
      </div>
    </div>

    <!-- Lead time slider (shared across all models) -->
    <div v-if="showComparison" class="bg-white rounded-lg shadow p-6 mb-6">
      <div class="mb-4">
        <label class="block text-sm font-medium text-gray-700 mb-2">
          Lead Time: <span class="text-blue-600 font-bold">+{{ (leadTime + 1) * 5 }} min</span>
        </label>
        <input
          type="range"
          v-model.number="leadTime"
          min="0"
          max="11"
          step="1"
          class="w-full accent-blue-600"
        />
        <div class="flex justify-between text-xs text-gray-400 mt-1">
          <span>+5 min</span>
          <span>+30 min</span>
          <span>+60 min</span>
        </div>
      </div>

      <!-- Side-by-side grid: one column per model -->
      <div
        class="grid gap-4"
        :style="{ gridTemplateColumns: `repeat(${availableModels.length}, 1fr)` }"
      >
        <div v-for="model in availableModels" :key="model" class="text-center">
          <h3 class="text-sm font-semibold text-gray-600 mb-2">{{ model }}</h3>
          <img
            :src="api.figureUrl(model, selectedDateTime, leadTime, 'prediction')"
            :key="`${model}-${leadTime}`"
            class="rounded-lg shadow border border-gray-200 w-full"
            :alt="`${model} prediction`"
          />
        </div>
      </div>

      <!-- Groundtruth row -->
      <div class="mt-6 max-w-md mx-auto text-center">
        <h3 class="text-sm font-semibold text-gray-600 mb-2">Groundtruth</h3>
        <img
          :src="api.figureUrl(availableModels[0], selectedDateTime, leadTime, 'groundtruth')"
          :key="`gt-${leadTime}`"
          class="rounded-lg shadow border border-gray-200 w-full"
          alt="Groundtruth"
        />
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

const configStore = useConfigStore()

// Form state
const selectedModels = ref([])
const selectedDateTime = ref('')

// Check state
const checking = ref(false)
const checkResults = ref([])

// Comparison display
const showComparison = ref(false)
const leadTime = ref(5)

const error = ref(null)

const canCheck = computed(() => {
  return selectedModels.value.length >= 2 && selectedDateTime.value && !checking.value
})

// Models that have available predictions
const availableModels = computed(() => {
  return checkResults.value.filter(r => r.exists).map(r => r.model)
})

async function checkAndLoad() {
  error.value = null
  checking.value = true
  checkResults.value = []
  showComparison.value = false

  try {
    // Check all selected models in parallel
    const promises = selectedModels.value.map(model =>
      api.checkSinglePrediction(model, selectedDateTime.value)
    )
    checkResults.value = await Promise.all(promises)

    // Show comparison if at least 2 models have data
    if (availableModels.value.length >= 2) {
      showComparison.value = true
    } else if (availableModels.value.length < 2) {
      error.value = 'Need at least 2 models with available predictions for comparison.'
    }
  } catch (e) {
    error.value = `Check failed: ${e.message}`
  } finally {
    checking.value = false
  }
}
</script>