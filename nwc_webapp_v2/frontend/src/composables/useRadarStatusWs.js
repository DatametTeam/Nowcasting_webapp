/**
 * useRadarStatusWs — WebSocket client for radar availability update notifications.
 *
 * Connects to /api/radar-status/ws and calls onRadarStatusUpdated() whenever
 * the backend broadcasts a radar_status_updated message (cron script finished
 * downloading a new SITES txt file from FTP).
 *
 * Auto-reconnects with exponential back-off (1s → 2s → 4s … capped at 30s).
 */

import { ref, onUnmounted } from 'vue'

const BASE_DELAY_MS = 1_000
const MAX_DELAY_MS  = 30_000

export function useRadarStatusWs({ onRadarStatusUpdated } = {}) {
  const connected = ref(false)

  let ws         = null
  let retryDelay = BASE_DELAY_MS
  let retryTimer = null
  let stopped    = false

  function wsUrl() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    return `${proto}://${location.host}/api/radar-status/ws`
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
        if (msg.type === 'radar_status_updated' && onRadarStatusUpdated) {
          onRadarStatusUpdated(msg)
        }
      } catch {
        // malformed frame — ignore
      }
    }

    ws.onclose = (e) => {
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
