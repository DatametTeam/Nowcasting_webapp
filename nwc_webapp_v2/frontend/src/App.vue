<!--
  App.vue - The root component (like app.py in Streamlit).
-->
<template>
  <div class="min-h-dvh bg-gray-50">
    <!-- Top Navigation Bar — Leonardo red (#C8102E) -->
    <nav class="bg-[#C8102E] text-white shadow-lg">
      <div class="px-2 sm:px-4">
        <div class="flex items-center h-12 sm:h-14 gap-2 sm:gap-4">
          <!-- App Logo -->
          <div class="flex items-center flex-shrink-0">
            <img
              src="/leo-weather-logo-white-transp.png"
              alt="LEO Weather"
              class="h-7 sm:h-10 w-auto object-contain pr-1"
            />
          </div>

          <!-- Navigation Tabs — horizontally scrollable on mobile, no wrap -->
          <div
            class="flex flex-nowrap gap-0.5 sm:gap-1 overflow-x-auto no-scrollbar min-w-0 flex-1"
          >
            <router-link
              v-for="item in navItems"
              :key="item.path"
              :to="item.path"
              class="px-2 sm:px-4 py-1.5 sm:py-2 rounded-md text-xs sm:text-sm font-medium transition-colors duration-150 whitespace-nowrap flex-shrink-0"
              :class="isActive(item.path)
                ? 'bg-white/20 text-white'
                : 'text-white/70 hover:bg-white/10 hover:text-white'"
            >
              {{ item.label }}
            </router-link>
          </div>

          <!-- Right side: status + settings gear -->
          <div class="flex items-center gap-2 sm:gap-3 flex-shrink-0">
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
            <div class="w-px h-6 bg-white/30" />
            <!-- Settings gear button -->
            <button
              @click="settingsOpen = true"
              class="w-8 h-8 flex items-center justify-center rounded-md
                     text-white/70 hover:text-white hover:bg-white/10 transition-colors"
              title="Settings"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </button>
          </div>
        </div>
      </div>
    </nav>

    <!-- Settings modal -->
    <Teleport to="body">
      <div
        v-if="settingsOpen"
        class="fixed inset-0 z-[9000] flex items-center justify-center p-4"
      >
        <!-- Backdrop -->
        <div class="absolute inset-0 bg-black/60" @click="settingsOpen = false" />

        <!-- Panel -->
        <div class="relative bg-gray-900 border border-gray-700 rounded-xl shadow-2xl w-full max-w-sm text-white overflow-y-auto max-h-[90dvh]">
          <!-- Header -->
          <div class="flex items-center justify-between px-5 py-4 border-b border-gray-700">
            <h2 class="text-base font-semibold">Settings</h2>
            <button
              @click="settingsOpen = false"
              class="w-8 h-8 flex items-center justify-center rounded-full
                     text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path d="M6 18L18 6M6 6l12 12" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </button>
          </div>

          <div class="px-5 py-4 space-y-6">

            <!-- Time display -->
            <div>
              <p class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Time display</p>
              <div class="flex rounded-lg overflow-hidden border border-gray-600">
                <button
                  @click="settings.timeZone = 'local'"
                  class="flex-1 py-2 text-sm font-medium transition-colors"
                  :class="settings.timeZone === 'local'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'"
                >Local time</button>
                <button
                  @click="settings.timeZone = 'utc'"
                  class="flex-1 py-2 text-sm font-medium transition-colors border-l border-gray-600"
                  :class="settings.timeZone === 'utc'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'"
                >UTC</button>
              </div>
              <p class="text-[11px] text-gray-500 mt-1">
                {{ settings.timeZone === 'local' ? 'Timestamps shown in Europe/Rome (UTC+1/+2)' : 'Timestamps shown in UTC' }}
              </p>
            </div>

            <!-- Base map -->
            <div>
              <p class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Base map</p>
              <div class="grid grid-cols-2 gap-2">
                <button
                  v-for="layer in BASE_LAYERS"
                  :key="layer"
                  @click="settings.baseLayer = layer"
                  class="py-2 px-3 rounded-lg text-sm font-medium border transition-colors"
                  :class="settings.baseLayer === layer
                    ? 'bg-blue-600 border-blue-500 text-white'
                    : 'bg-gray-800 border-gray-600 text-gray-300 hover:bg-gray-700'"
                >{{ layer }}</button>
              </div>
            </div>

            <!-- Show colorbars -->
            <div class="flex items-center justify-between">
              <div>
                <p class="text-sm font-medium">Show colour bars</p>
                <p class="text-[11px] text-gray-500">Legend bars on radar maps</p>
              </div>
              <button
                @click="settings.showColorbars = !settings.showColorbars"
                class="relative w-11 h-6 rounded-full transition-colors flex-shrink-0"
                :class="settings.showColorbars ? 'bg-blue-600' : 'bg-gray-600'"
              >
                <span
                  class="absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform"
                  :class="settings.showColorbars ? 'translate-x-5' : 'translate-x-0'"
                />
              </button>
            </div>

            <!-- Default lookback -->
            <div>
              <p class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Default lookback window</p>
              <div class="flex gap-2">
                <button
                  v-for="h in [1, 2, 4, 6, 12]"
                  :key="h"
                  @click="settings.defaultLookback = h"
                  class="flex-1 py-1.5 rounded-lg text-sm font-medium border transition-colors"
                  :class="settings.defaultLookback === h
                    ? 'bg-blue-600 border-blue-500 text-white'
                    : 'bg-gray-800 border-gray-600 text-gray-300 hover:bg-gray-700'"
                >{{ h }}h</button>
              </div>
            </div>

            <!-- Default animation speed -->
            <div>
              <p class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Default animation speed</p>
              <div class="flex gap-2">
                <button
                  v-for="s in [0.5, 1, 2, 4]"
                  :key="s"
                  @click="settings.defaultSpeed = s"
                  class="flex-1 py-1.5 rounded-lg text-sm font-medium border transition-colors"
                  :class="settings.defaultSpeed === s
                    ? 'bg-blue-600 border-blue-500 text-white'
                    : 'bg-gray-800 border-gray-600 text-gray-300 hover:bg-gray-700'"
                >{{ s }}×</button>
              </div>
            </div>

          </div>
        </div>
      </div>
    </Teleport>

    <!-- Page Content -->
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
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useConfigStore } from './stores/config.js'
import { useSettingsStore } from './stores/settings.js'
import api from './api.js'

const route = useRoute()
const configStore = useConfigStore()
const settings = useSettingsStore()

const BASE_LAYERS = ['Dark', 'OpenStreetMap', 'Satellite', 'Terrain']

const settingsOpen = ref(false)

const ALL_TABS = [
  { key: 'realtime',    path: '/realtime',      label: 'Real Time' },
  { key: 'nowcasting',  path: '/nowcasting',   label: 'Nowcasting' },
  { key: 'wr10',        path: '/wr10',          label: 'WR10' },
  { key: 'wr10explorer',path: '/wr10-explorer', label: 'WR10 Explorer' },
  { key: 'explorer',    path: '/explorer',      label: 'Data Explorer' },
  { key: 'comparison', path: '/comparison', label: 'Model Comparison' },
  { key: 'metrics',    path: '/metrics',    label: 'Metrics Analysis' },
  { key: 'assessment', path: '/assessment', label: 'RT Assessment' },
]

const navItems = computed(() =>
  ALL_TABS.filter(tab => configStore.enabledTabs.includes(tab.key))
)

const backendConnected = ref(false)
const environment = ref('connecting...')

function isActive(path) {
  return route.path === path
}

onMounted(async () => {
  try {
    await api.health()
    backendConnected.value = true
    await configStore.fetchConfig()
    environment.value = configStore.isHpc ? 'HPC Mode' : (configStore.isServer ? 'Server Mode' : 'Local Mode')
  } catch (e) {
    backendConnected.value = false
    environment.value = 'disconnected'
  }
})
</script>

<style>
.no-scrollbar::-webkit-scrollbar { display: none; }
.no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
</style>
