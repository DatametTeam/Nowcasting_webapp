<!--
  StatusBadge.vue — Small colored badge showing a status.

  USAGE:
    <StatusBadge status="success" text="All predictions exist" />
    <StatusBadge status="warning" text="3 missing" />
    <StatusBadge status="error" text="Failed" />
    <StatusBadge status="loading" text="Checking..." />
-->
<template>
  <span
    class="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium"
    :class="badgeClass"
  >
    <!-- Animated spinner for loading state -->
    <svg v-if="status === 'loading'" class="animate-spin h-3 w-3" viewBox="0 0 24 24">
      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
    <!-- Solid dot for non-loading states -->
    <span v-else class="h-1.5 w-1.5 rounded-full" :class="dotClass" />
    {{ text }}
  </span>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  status: { type: String, default: 'info' },  // success, warning, error, loading, info
  text: { type: String, default: '' },
})

// computed: a value that auto-recalculates when its dependencies change
// (like a @property in Python)
const badgeClass = computed(() => {
  const classes = {
    success: 'bg-green-100 text-green-700',
    warning: 'bg-yellow-100 text-yellow-700',
    error: 'bg-red-100 text-red-700',
    loading: 'bg-blue-100 text-blue-700',
    info: 'bg-gray-100 text-gray-700',
  }
  return classes[props.status] || classes.info
})

const dotClass = computed(() => {
  const classes = {
    success: 'bg-green-500',
    warning: 'bg-yellow-500',
    error: 'bg-red-500',
    info: 'bg-gray-500',
  }
  return classes[props.status] || classes.info
})
</script>