<!--
  ColorBar.vue — Radar product colorbar legend.

  Accepts an optional `legend` prop with colormap data from /api/config/:
    { label, unit, thresholds: number[], colors: string[] }

  Accepts an optional `productName` prop (e.g. "SRI", "VMI") displayed as a
  vertical label on the LEFT of the gradient (used in DataExplorerView stacking).

  Falls back to hardcoded SRI defaults when no prop is provided (RealTimeView
  backward compat — no changes needed there).
-->
<template>
  <div class="colorbar-container">
    <div class="colorbar-body">
      <!-- Left label: product name (vertical) -->
      <div v-if="productName" class="colorbar-label">
        {{ productName }}
      </div>
      <!-- Gradient bar -->
      <div class="colorbar-gradient" :style="{ background: gradient }" />
      <!-- Tick labels on the right -->
      <div class="colorbar-ticks">
        <span
          v-for="tick in ticks"
          :key="tick.value"
          class="colorbar-tick"
          :style="{ bottom: tick.position + '%' }"
        >
          {{ tick.label }}
        </span>
      </div>
    </div>
    <!-- Unit label below -->
    <div class="colorbar-unit">{{ unit }}</div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const DEFAULT_THRESHOLDS = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100]
const DEFAULT_COLORS = [
  'rgb(100,100,100)', 'rgb(0,120,200)', 'rgb(0,200,250)', 'rgb(0,150,0)',
  'rgb(0,250,0)', 'rgb(250,250,0)', 'rgb(250,150,0)', 'rgb(250,0,0)',
  'rgb(180,0,0)', 'rgb(220,100,250)',
]
const DEFAULT_UNIT = 'mm/h'

const props = defineProps({
  legend: { type: Object, default: null },
  /** Short product name shown as a vertical label on the left (e.g. "SRI"). */
  productName: { type: String, default: null },
})

const thresholds = computed(() =>
  props.legend?.thresholds?.length ? props.legend.thresholds : DEFAULT_THRESHOLDS
)
const colors = computed(() =>
  props.legend?.colors?.length ? props.legend.colors : DEFAULT_COLORS
)
const unit = computed(() => props.legend?.unit ?? DEFAULT_UNIT)

// Format tick labels compactly so wide numbers don't fatten the colorbar:
// 12000 → 12k, 1500 → 1.5k. Smaller values stay as-is.
function formatTick(v) {
  if (typeof v !== 'number' || !isFinite(v)) return String(v)
  const abs = Math.abs(v)
  if (abs >= 1000) {
    const k = v / 1000
    return (Number.isInteger(k) ? k.toFixed(0) : k.toFixed(1)) + 'k'
  }
  return String(v)
}

const ticks = computed(() =>
  thresholds.value.map((value, i) => ({
    value,
    label: formatTick(value),
    position: (i / (thresholds.value.length - 1)) * 100,
  }))
)

const gradient = computed(() => {
  const n = colors.value.length
  const stops = colors.value.map((color, i) => `${color} ${((i / (n - 1)) * 100).toFixed(1)}%`)
  return `linear-gradient(to top, ${stops.join(', ')})`
})
</script>

<style scoped>
.colorbar-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 3px;
  padding: 6px 5px;
  background: rgba(0, 0, 0, 0.65);
  backdrop-filter: blur(8px);
  border-radius: 8px;
  user-select: none;
}

.colorbar-body {
  display: flex;
  align-items: stretch;
  height: 200px;
  gap: 3px;
}

/* Mobile: smaller colorbar so it doesn't dominate the screen */
@media (max-width: 640px) {
  .colorbar-container {
    padding: 3px 2px;
    gap: 1px;
  }
  .colorbar-body {
    height: 90px;
    gap: 2px;
  }
  .colorbar-label {
    width: 8px;
    font-size: 7px;
    letter-spacing: 0.3px;
  }
  .colorbar-gradient {
    width: 7px;
  }
  /* Tick column narrower since labels are now compact (12k vs 12000) */
  .colorbar-ticks {
    width: 14px;
  }
  .colorbar-tick {
    font-size: 7px;
    left: 1px;
  }
  .colorbar-unit {
    font-size: 7px;
  }
}

/* Vertical product label (e.g. "SRI") */
.colorbar-label {
  width: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(255, 255, 255, 0.85);
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 1px;
  writing-mode: vertical-rl;
  transform: rotate(180deg);
  white-space: nowrap;
}

.colorbar-gradient {
  width: 12px;
  border-radius: 2px;
  flex-shrink: 0;
}

.colorbar-ticks {
  position: relative;
  width: 26px;
}

.colorbar-tick {
  position: absolute;
  left: 2px;
  transform: translateY(50%);
  color: white;
  font-size: 9px;
  font-variant-numeric: tabular-nums;
  line-height: 1;
}

.colorbar-unit {
  color: rgba(255, 255, 255, 0.7);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.5px;
}
</style>
