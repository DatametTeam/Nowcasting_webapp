<!--
  MetricSection.vue — Displays one metric (CSI/POD/FAR) for multiple models.

  USAGE:
    <MetricSection title="CSI" :data="results.csi" :models="['ConvLSTM', 'SPROG']" />

  The data structure from the backend is:
    { model_name: { column: { row: value } } }

  Each model gets its own table displayed in a tab-like interface.
-->
<template>
  <div class="bg-white rounded-lg shadow p-6">
    <h2 class="text-lg font-semibold text-gray-700 mb-4">{{ title }}</h2>

    <!-- Model tabs (if multiple models) -->
    <div v-if="models.length > 1" class="flex gap-1 mb-4 border-b border-gray-200">
      <button
        v-for="model in models"
        :key="model"
        @click="activeModel = model"
        class="px-4 py-2 text-sm font-medium border-b-2 transition-colors"
        :class="activeModel === model
          ? 'border-blue-500 text-blue-600'
          : 'border-transparent text-gray-500 hover:text-gray-700'"
      >
        {{ model }}
      </button>
    </div>

    <!-- Table for active model -->
    <div v-if="data[activeModel]">
      <DataTable :data="data[activeModel]" />
    </div>
    <div v-else class="text-sm text-gray-400">
      No data available for {{ activeModel }}
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import DataTable from './DataTable.vue'

const props = defineProps({
  title: { type: String, required: true },
  data: { type: Object, required: true },
  models: { type: Array, required: true },
})

const activeModel = ref(props.models[0] || '')

// If models list changes, reset to first model
watch(() => props.models, (newModels) => {
  if (newModels.length > 0 && !newModels.includes(activeModel.value)) {
    activeModel.value = newModels[0]
  }
})
</script>