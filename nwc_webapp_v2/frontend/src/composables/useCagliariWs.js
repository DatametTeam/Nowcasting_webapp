/**
 * useCagliariWs — WebSocket client for Cagliari X-band new-data notifications.
 *
 * Connects to /api/cagliari/ws and calls onCagliariUpdate(data) whenever
 * the backend broadcasts a cagliari_update message (new file arrived).
 *
 * Auto-reconnects with exponential back-off (1s → 2s → 4s … capped at 30s).
 */

import { ref, onUnmounted } from 'vue'

const BASE_DELAY_MS = 1_000
const MAX_DELAY_MS  = 30_000

export function useCagliariWs({ onCagliariUpdate } = {}) {
  const connected = ref(false)

  let ws         = null
  let retryDelay = BASE_DELAY_MS
  let retryTimer = null
  let stopped    = false

  function wsUrl() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    return `${proto}://${location.host}/api/cagliari/ws`
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
        if (msg.type === 'cagliari_update' && onCagliariUpdate) {
          onCagliariUpdate(msg.data)
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
      // onclose fires right after, which handles reconnect
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
