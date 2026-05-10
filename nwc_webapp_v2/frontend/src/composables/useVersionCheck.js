/**
 * useVersionCheck — auto-reload stale browser tabs after a redeploy.
 *
 * At build time, vite.config.js stamps VITE_APP_VERSION with the git commit
 * hash. This composable polls /api/health every 60 s and compares the
 * server's `git_hash` to the compiled-in value. When they differ, the page
 * reloads — the new index.html (served with no-cache) loads the new JS/CSS
 * bundles, so users always run the latest version without needing to manually
 * refresh.
 *
 * Skipped entirely in dev mode (when either hash is "dev") to avoid spurious
 * reloads during hot-module-replacement development.
 */

import { onMounted, onUnmounted } from 'vue'
import api from '../api.js'

const BUILD_VERSION = import.meta.env.VITE_APP_VERSION ?? 'dev'
// import.meta.env.DEV is true when running `vite dev` (dev server), false after `vite build`.
// Skip the check entirely in dev mode: the build hash is captured at Vite startup, so any
// commit made while Vite is running diverges from the backend's runtime hash and would
// trigger an endless reload loop before the page even finishes loading.
const IS_DEV_SERVER = import.meta.env.DEV
const CHECK_INTERVAL_MS = 60_000   // check every 60 s
const INITIAL_DELAY_MS  = 30_000   // first check after 30 s (avoid reload on first paint)

export function useVersionCheck() {
  let intervalId  = null
  let initialId   = null

  async function check() {
    if (IS_DEV_SERVER || BUILD_VERSION === 'dev') return   // skip in dev / when git unavailable
    try {
      const health = await api.health()
      const serverHash = health?.git_hash
      if (serverHash && serverHash !== 'dev' && serverHash !== BUILD_VERSION) {
        console.info(
          `[version] stale bundle detected (built=${BUILD_VERSION} server=${serverHash}) — reloading`
        )
        window.location.reload()
      }
    } catch {
      // backend unreachable — ignore, don't reload
    }
  }

  onMounted(() => {
    initialId  = setTimeout(() => { check(); initialId = null }, INITIAL_DELAY_MS)
    intervalId = setInterval(check, CHECK_INTERVAL_MS)
  })

  onUnmounted(() => {
    if (initialId)  clearTimeout(initialId)
    if (intervalId) clearInterval(intervalId)
  })
}
