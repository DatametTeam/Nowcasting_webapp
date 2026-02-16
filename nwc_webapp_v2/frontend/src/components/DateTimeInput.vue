<!--
  DateTimeInput.vue — Date + Time picker with 5-minute intervals.

  USAGE:
    <DateTimeInput v-model="startDateTime" label="Start" />

  The value format is "YYYY-MM-DDTHH:MM" which matches what our API expects.

  Layout: a date input on top, and a time grid below showing every 5 minutes.
  The time buttons are arranged in a clear grid so users can quickly pick
  common radar times (00, 05, 10, 15, ..., 55).
-->
<template>
  <div>
    <label v-if="label" class="block text-sm font-medium text-gray-700 mb-1">
      {{ label }}
    </label>

    <!-- Date input -->
    <input
      type="date"
      :value="dateValue"
      @input="onDateChange($event.target.value)"
      class="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm
             shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
    />

    <!-- Hour + Minute selectors (side by side) -->
    <div class="flex gap-2 mt-2">
      <!-- Hour dropdown -->
      <div class="flex-1">
        <label class="block text-xs text-gray-500 mb-1">Hour</label>
        <select
          :value="hourValue"
          @change="onHourChange($event.target.value)"
          class="w-full rounded-md border border-gray-300 bg-white px-2 py-2 text-sm
                 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          <option value="" disabled>HH</option>
          <option v-for="h in hours" :key="h" :value="h">{{ h }}</option>
        </select>
      </div>

      <!-- Minute dropdown (5-min intervals only) -->
      <div class="flex-1">
        <label class="block text-xs text-gray-500 mb-1">Minute</label>
        <select
          :value="minuteValue"
          @change="onMinuteChange($event.target.value)"
          class="w-full rounded-md border border-gray-300 bg-white px-2 py-2 text-sm
                 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          <option value="" disabled>MM</option>
          <option v-for="m in minutes" :key="m" :value="m">{{ m }}</option>
        </select>
      </div>
    </div>

    <!-- Current selection preview -->
    <div v-if="modelValue" class="mt-1.5 text-xs text-blue-600 font-medium">
      {{ formatPreview(modelValue) }}
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  modelValue: { type: String, default: '' },  // "YYYY-MM-DDTHH:MM"
  label: { type: String, default: '' },
})

const emit = defineEmits(['update:modelValue'])

// Generate hour options: 00, 01, ..., 23
const hours = Array.from({ length: 24 }, (_, i) => String(i).padStart(2, '0'))

// Generate minute options: 00, 05, 10, 15, ..., 55 (5-minute intervals only)
const minutes = Array.from({ length: 12 }, (_, i) => String(i * 5).padStart(2, '0'))

// Parse the current value into date, hour, minute parts
const dateValue = computed(() => {
  if (!props.modelValue) return ''
  return props.modelValue.split('T')[0] || ''
})

const hourValue = computed(() => {
  if (!props.modelValue || !props.modelValue.includes('T')) return ''
  const time = props.modelValue.split('T')[1] || ''
  return time.split(':')[0] || ''
})

const minuteValue = computed(() => {
  if (!props.modelValue || !props.modelValue.includes('T')) return ''
  const time = props.modelValue.split('T')[1] || ''
  return time.split(':')[1] || ''
})

// Rebuild the full value when any part changes
function buildValue(date, hour, minute) {
  if (!date) return ''
  const h = hour || '00'
  const m = minute || '00'
  return `${date}T${h}:${m}`
}

function onDateChange(newDate) {
  emit('update:modelValue', buildValue(newDate, hourValue.value, minuteValue.value))
}

function onHourChange(newHour) {
  emit('update:modelValue', buildValue(dateValue.value, newHour, minuteValue.value))
}

function onMinuteChange(newMinute) {
  emit('update:modelValue', buildValue(dateValue.value, hourValue.value, newMinute))
}

// Format for the preview line
function formatPreview(value) {
  if (!value || !value.includes('T')) return ''
  const [date, time] = value.split('T')
  const [year, month, day] = date.split('-')
  return `${day}/${month}/${year} ${time}`
}
</script>