<!--
  ModelSelector.vue — Reusable model dropdown.

  USAGE:
    <ModelSelector v-model="selectedModel" />

  HOW v-model WORKS ON CUSTOM COMPONENTS:
  v-model is a shortcut for two things:
    1. Passing a value DOWN to the child (:modelValue="selectedModel")
    2. Listening for changes UP from the child (@update:modelValue="selectedModel = $event")

  So when the user picks a model in this dropdown, the parent's selectedModel
  variable updates automatically. This is like a Streamlit selectbox return value.
-->
<template>
  <div>
    <label v-if="label" class="block text-sm font-medium text-gray-700 mb-1">
      {{ label }}
    </label>
    <select
      :value="modelValue"
      @change="$emit('update:modelValue', $event.target.value)"
      class="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm
             shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
    >
      <option value="" disabled>Select a model...</option>
      <option v-for="model in models" :key="model" :value="model">
        {{ model }}
      </option>
    </select>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useConfigStore } from '../stores/config.js'

// Props: values passed in from the parent component
// defineProps is Vue's way of declaring what data this component accepts
const props = defineProps({
  modelValue: { type: String, default: '' },
  label: { type: String, default: 'Model' },
})

// Emit: events sent back to the parent (the "update" part of v-model)
defineEmits(['update:modelValue'])

// Get the model list from the global store
// BUG FIX: must use computed() so it reacts when the async fetchConfig() finishes.
// Before, `const models = configStore.models` copied the empty array ONCE at setup time,
// so the dropdown was always empty. computed() re-evaluates whenever the store changes.
const configStore = useConfigStore()
const models = computed(() => configStore.models)
</script>