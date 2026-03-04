/**
 * Pinia Store: Config
 *
 * WHAT IS PINIA?
 * Pinia is Vue's state management library. Think of it as a GLOBAL version
 * of st.session_state. Any component can read or modify this state, and
 * all components that use it will automatically update.
 *
 * WHY USE A STORE?
 * The model list and environment info are needed by MANY pages.
 * Instead of each page fetching /api/config separately, we fetch ONCE
 * and store it here. Every page reads from the same store.
 *
 * HOW IT WORKS:
 * - defineStore('name', { ... }) creates a store
 * - state: () => ({...}) defines the data (like st.session_state)
 * - actions: { ... } defines functions that modify the state
 * - In a component: const store = useConfigStore(); store.models
 */
import { defineStore } from 'pinia'
import api from '../api.js'

export const useConfigStore = defineStore('config', {
  // STATE: reactive data shared across all components
  state: () => ({
    models: [],
    environment: 'unknown',
    isHpc: false,
    sriFolder: '',
    gifStorage: '',
    csiThresholds: [],
    radarProducts: {},
    loaded: false,
    error: null,
  }),

  // ACTIONS: functions that modify state (like methods in a Python class)
  actions: {
    /**
     * Fetch config from the backend. Called once when the app loads.
     */
    async fetchConfig() {
      try {
        const data = await api.getConfig()
        this.models = data.models
        this.environment = data.environment
        this.isHpc = data.environment === 'hpc'
        this.sriFolder = data.sri_folder
        this.gifStorage = data.gif_storage
        this.csiThresholds = data.csi_thresholds
        this.radarProducts = data.radar_products || {}
        this.loaded = true
        this.error = null
      } catch (e) {
        this.error = e.message
        console.error('Failed to load config:', e)
      }
    },
  },
})