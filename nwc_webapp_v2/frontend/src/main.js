/**
 * Main entry point for the Vue application.
 *
 * HOW THIS WORKS:
 * This is like app.py in your Streamlit app - the starting point.
 *
 * It creates the Vue app, installs plugins (Router, Pinia), and mounts it
 * into the <div id="app"> in index.html.
 *
 * Key plugins:
 * - Router: handles page navigation (tabs become URL routes)
 * - Pinia: manages global state (replaces st.session_state)
 */
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router.js'

// Import global CSS (Tailwind + our custom styles)
import './assets/main.css'

// Create the Vue application
const app = createApp(App)

// Install plugins
app.use(createPinia())  // State management (like st.session_state but better)
app.use(router)         // Page routing (like st.tabs but with URLs)

// Mount the app into the HTML page
app.mount('#app')