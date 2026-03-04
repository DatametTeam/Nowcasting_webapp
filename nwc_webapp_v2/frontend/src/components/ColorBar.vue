<!--
  ColorBar.vue — Radar product colorbar legend.

  Accepts an optional `legend` prop:
    { label, unit, thresholds: number[], colors: string[] }

  If no prop is provided (e.g. in RealTimeView), falls back to the
  hardcoded SRI rain rate colormap (backward compat).

  Thresholds and colors come from /api/config/ → radar_products[product].
  The gradient is evenly spaced (one stop per color), but tick labels
  show the actual physical values (non-linear).
-->
<template>
  <div class="colorbar-container">
    <div class="colorbar-title">{{ unit }}</div>
    <div class="colorbar-body">
      <!-- Tick labels on the left -->
      <div class="colorbar-ticks">
        <span
          v-for="(tick, i) in ticks"
          :key="tick.value"
          class="colorbar-tick"
          :style="{ bottom: tick.position + '%' }"
        >
          {{ tick.value }}
        </span>
      </div>
      <!-- Gradient bar -->
      <div class="colorbar-gradient" :style="{ background: gradient }" />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

// ---- Default SRI (R legend) data — used when no `legend` prop is provided ----
const DEFAULT_THRESHOLDS = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100]
const DEFAULT_COLORS = [
  'rgb(100,100,100)',
  'rgb(0,120,200)',
  'rgb(0,200,250)',
  'rgb(0,150,0)',
  'rgb(0,250,0)',
  'rgb(250,250,0)',
  'rgb(250,150,0)',
  'rgb(250,0,0)',
  'rgb(180,0,0)',
  'rgb(220,100,250)',
]
const DEFAULT_UNIT = 'mm/h'

const props = defineProps({
  /**
   * Legend data from the config store.
   * Shape: { label: string, unit: string, thresholds: number[], colors: string[] }
   * If null/undefined, the hardcoded SRI defaults are used.
   */
  legend: {
    type: Object,
    default: null,
  },
})

const thresholds = computed(() =>
  props.legend?.thresholds?.length ? props.legend.thresholds : DEFAULT_THRESHOLDS
)
const colors = computed(() =>
  props.legend?.colors?.length ? props.legend.colors : DEFAULT_COLORS
)
const unit = computed(() => props.legend?.unit ?? DEFAULT_UNIT)

// Position of each tick as a % from bottom (evenly spaced in colormap space)
const ticks = computed(() =>
  thresholds.value.map((value, i) => ({
    value,
    position: (i / (thresholds.value.length - 1)) * 100,
  }))
)

// Build CSS gradient string from colors (bottom = lowest, top = highest)
const gradient = computed(() => {
  const n = colors.value.length
  const stops = colors.value.map((color, i) => {
    const pct = ((i / (n - 1)) * 100).toFixed(1)
    return `${color} ${pct}%`
  })
  return `linear-gradient(to top, ${stops.join(', ')})`
})
</script>

<style scoped>
.colorbar-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 8px 6px;
  background: rgba(0, 0, 0, 0.65);
  backdrop-filter: blur(8px);
  border-radius: 8px;
  user-select: none;
}

.colorbar-title {
  color: white;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.5px;
}

.colorbar-body {
  display: flex;
  align-items: stretch;
  height: 180px;
  gap: 4px;
}

.colorbar-ticks {
  position: relative;
  width: 28px;
  display: flex;
  flex-direction: column;
}

.colorbar-tick {
  position: absolute;
  right: 0;
  transform: translateY(50%);
  color: white;
  font-size: 10px;
  font-variant-numeric: tabular-nums;
  line-height: 1;
  text-align: right;
}

.colorbar-gradient {
  width: 14px;
  border-radius: 3px;
}
</style>
