/**
 * useFssWs — WebSocket client for FSS assessment update notifications.
 *
 * Connects to /api/fss/ws and calls onFssUpdated() whenever the backend
 * broadcasts an fss_updated message (cron script finished writing CSVs).
 *
 * Auto-reconnects with exponential back-off (1s → 2s → 4s … capped at 30s).
 */

import { ref, onUnmounted } from 'vue'

const BASE_DELAY_MS = 1_000
const MAX_DELAY_MS  = 30_000

export function useFssWs({ onFssUpdated } = {}) {
  const connected = ref(false)

  let ws         = null
  let retryDelay = BASE_DELAY_MS
  let retryTimer = null
  let stopped    = false

  function wsUrl() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    return `${proto}://${location.host}/api/fss/ws`
  }

  function connect() {
    if (stopped) return
    ws = new WebSocket(wsUrl())

    ws.onopen = () => {
      connected.value = true
      retryDelay = BASE_DELAY_MS
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg.type === 'fss_updated' && onFssUpdated) {
          onFssUpdated(msg)
        }
      } catch {
        // malformed frame — ignore
      }
    }

    ws.onclose = () => {
      connected.value = false
      ws = null
      if (!stopped) scheduleReconnect()
    }

    ws.onerror = () => {
      // onclose fires right after, handles reconnect
    }
  }

  function scheduleReconnect() {
    retryTimer = setTimeout(() => {
      retryTimer = null
      connect()
    }, retryDelay)
    retryDelay = Math.min(retryDelay * 2, MAX_DELAY_MS)
  }

  function stop() {
    stopped = true
    connected.value = false
    if (retryTimer) { clearTimeout(retryTimer); retryTimer = null }
    if (ws) { ws.onclose = null; ws.close(); ws = null }
  }

  connect()
  onUnmounted(stop)

  return { connected, stop }
}
