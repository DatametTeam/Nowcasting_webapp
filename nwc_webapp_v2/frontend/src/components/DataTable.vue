<!--
  DataTable.vue — Renders a dict-of-dicts as an HTML table.

  USAGE:
    <DataTable :data="{ 'col1': { 'row1': 0.5, 'row2': 0.7 }, 'col2': { 'row1': 0.3 } }" />

  The backend returns DataFrames serialized as nested dicts (df.to_dict()).
  This component renders them as a clean HTML table with styling.

  Structure of the data:
    { column_name: { row_name: value, ... }, ... }

  This is the "columns" orientation from pandas DataFrame.to_dict().
-->
<template>
  <div class="overflow-x-auto">
    <table class="min-w-full text-sm border-collapse">
      <thead>
        <tr class="bg-gray-50">
          <th class="px-3 py-2 text-left font-medium text-gray-600 border border-gray-200"></th>
          <th
            v-for="col in columns"
            :key="col"
            class="px-3 py-2 text-center font-medium text-gray-600 border border-gray-200"
          >
            {{ col }}
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in rows" :key="row" class="hover:bg-gray-50">
          <td class="px-3 py-2 font-medium text-gray-700 border border-gray-200 bg-gray-50">
            {{ row }}
          </td>
          <td
            v-for="col in columns"
            :key="col"
            class="px-3 py-2 text-center border border-gray-200"
            :class="cellClass(getValue(col, row))"
          >
            {{ formatValue(getValue(col, row)) }}
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  data: { type: Object, required: true },
  invertColors: { type: Boolean, default: false },
})

// Extract column names (top-level keys)
const columns = computed(() => Object.keys(props.data))

// Extract row names (keys of the first column's data)
const rows = computed(() => {
  const firstCol = columns.value[0]
  return firstCol ? Object.keys(props.data[firstCol]) : []
})

function getValue(col, row) {
  return props.data[col]?.[row]
}

function formatValue(val) {
  if (val === null || val === undefined) return '-'
  if (typeof val === 'number') return val.toFixed(3)
  return val
}

// Color-code cells based on value (green=good, red=bad for metrics 0-1)
function cellClass(val) {
  if (typeof val !== 'number') return ''
  if (props.invertColors) {
    // Lower is better (FAR): green when low, red when high
    if (val <= 0.3) return 'bg-green-50 text-green-700'
    if (val <= 0.6) return 'bg-yellow-50 text-yellow-700'
    if (val < 1) return 'bg-red-50 text-red-700'
    return 'text-gray-400'
  }
  // Default: higher is better (CSI, POD)
  if (val >= 0.7) return 'bg-green-50 text-green-700'
  if (val >= 0.4) return 'bg-yellow-50 text-yellow-700'
  if (val > 0) return 'bg-red-50 text-red-700'
  return 'text-gray-400'
}
</script>