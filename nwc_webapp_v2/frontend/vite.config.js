/**
 * Vite Configuration
 *
 * HOW THIS WORKS:
 * Vite is the "build tool" for the frontend. It does two things:
 *
 * 1. DEVELOPMENT: Runs a dev server with hot-reload (like streamlit's auto-reload
 *    but MUCH faster - changes appear instantly without page refresh)
 *
 * 2. PRODUCTION: Bundles all JS/CSS into optimized static files that FastAPI serves
 *
 * The "proxy" setting below is KEY for development:
 * - Vue dev server runs on port 5173
 * - FastAPI backend runs on port 8000
 * - When the frontend calls "/api/something", Vite forwards it to FastAPI
 * - This way the frontend doesn't need to know where the backend is
 */
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    vue(),
    tailwindcss(),
  ],

  server: {
    port: 5173,

    // Proxy API calls to FastAPI during development
    proxy: {
      '/api': {
        target: 'http://localhost:8000',  // FastAPI backend
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',    // WebSocket connections
        ws: true,
      },
    },
  },

  // Where to output built files (FastAPI will serve these)
  build: {
    outDir: '../backend/static',
    emptyOutDir: true,
  },
})