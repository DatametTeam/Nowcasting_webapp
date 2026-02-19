<!--
  App.vue - The root component (like app.py in Streamlit).

  HOW VUE COMPONENTS WORK:
  A .vue file has 3 sections:
    <template>  → HTML (what the user sees)
    <script>    → JavaScript (logic, data, functions)
    <style>     → CSS (optional, for component-specific styles)

  This is different from Streamlit where Python code and UI are mixed together.
  Here, structure (HTML), logic (JS), and style (CSS) are separated but in one file.

  KEY VUE CONCEPTS:
  - {{ variable }}      → displays a variable's value (like f-string in Python)
  - v-for="item in list" → loops over a list (like for loop)
  - @click="doSomething" → calls a function when clicked (like st.button callback)
  - :class="{ active: isActive }" → conditional CSS class
  - <router-link to="/path"> → navigation link (like clicking a tab)
  - <router-view />     → displays the current page's component
-->
<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Top Navigation Bar — Leonardo red (#E4002B) -->
    <nav class="bg-[#C8102E] text-white shadow-lg">
      <div class="px-4">
        <div class="flex items-center h-14">
          <!-- App Title -->
          <div class="flex items-center gap-3 mr-4 lg:mr-8">
            <span class="text-lg sm:text-xl font-bold tracking-tight">LEOWeather</span>
          </div>

          <!-- Navigation Tabs -->
          <div class="flex flex-wrap gap-0.5 sm:gap-1">
            <router-link
              v-for="item in navItems"
              :key="item.path"
              :to="item.path"
              class="px-2 sm:px-4 py-1.5 sm:py-2 rounded-md text-xs sm:text-sm font-medium transition-colors duration-150"
              :class="isActive(item.path)
                ? 'bg-white/20 text-white'
                : 'text-white/70 hover:bg-white/10 hover:text-white'"
            >
              {{ item.label }}
            </router-link>
          </div>

          <!-- Right side: status + Leonardo logo -->
          <div class="ml-auto flex items-center gap-2 sm:gap-4">
            <div class="flex items-center gap-2">
              <div
                class="w-2 h-2 rounded-full"
                :class="backendConnected ? 'bg-green-300' : 'bg-white/40'"
                :title="backendConnected ? 'Backend connected' : 'Backend disconnected'"
              />
              <span class="text-xs text-white/60 hidden sm:inline">
                {{ environment }}
              </span>
            </div>
            <div class="w-px h-6 bg-white/30 hidden sm:block" />
            <img src="/ldo-logo.png" alt="Leonardo" class="h-5 brightness-0 invert hidden sm:block" />
          </div>
        </div>
      </div>
    </nav>

    <!-- Page Content -->
    <!--
      router-view: This is where the current page's component renders.
      When you click "Nowcasting" in the nav, this shows NowcastingView.
      When you click "Metrics", this shows MetricsView.

      In Streamlit, this would be like the content inside "with tab1:" / "with tab2:".
      But here, only ONE page is rendered at a time (more efficient).
    -->
    <!--
      keep-alive: tells Vue to KEEP components in memory when you navigate away,
      instead of destroying them. This means form values, results, and scroll
      position are preserved when you switch tabs — just like real desktop apps.
      Without this, navigating to another tab would lose all your selections.
    -->
    <main>
      <router-view v-slot="{ Component }">
        <keep-alive>
          <component :is="Component" />
        </keep-alive>
      </router-view>
    </main>
  </div>
</template>

<script setup>
/**
 * <script setup> is Vue 3's "Composition API" syntax.
 *
 * Think of it like the top of a Python file where you define variables and functions.
 * Everything defined here is automatically available in the <template> above.
 *
 * Key concepts:
 * - ref("value")     → creates a reactive variable (changes auto-update the UI)
 * - onMounted(fn)    → runs when the component first appears (like __init__)
 * - useRoute()       → gives access to the current URL/route
 */
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useConfigStore } from './stores/config.js'
import api from './api.js'

const route = useRoute()
const configStore = useConfigStore()

// Navigation items (same tabs as the Streamlit app)
const navItems = [
  { path: '/realtime', label: 'Real Time' },
  { path: '/comparison', label: 'Model Comparison' },
  { path: '/metrics', label: 'Metrics Analysis' },
]

// Reactive state
const backendConnected = ref(false)
const environment = ref('connecting...')

// Check if a nav item is the active route
function isActive(path) {
  return route.path === path
}

// On mount: check backend health and load config into the global store
onMounted(async () => {
  try {
    await api.health()
    backendConnected.value = true
    // Load config into the Pinia store — all pages can now access it
    await configStore.fetchConfig()
    environment.value = configStore.isHpc ? 'HPC Mode' : 'Local Mode'
  } catch (e) {
    backendConnected.value = false
    environment.value = 'disconnected'
  }
})
</script>