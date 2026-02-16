<!--
  ColorBar.vue — Precipitation colorbar legend for the radar map.

  Displays a vertical gradient colorbar matching the server-side colormap
  used for rendering radar overlays. The colors and thresholds come from
  the legend file (src/nwc_webapp/resources/legends/R/legend.txt).

  COLORMAP EXPLANATION:
  The backend uses a LinearSegmentedColormap with 10 color stops,
  mapped non-linearly via CustomNorm to thresholds:
    [0, 1, 2, 5, 10, 20, 30, 50, 75, 100] mm/h

  In the colormap, these 10 colors are evenly spaced (0% to 100%),
  but the threshold values are non-linear. So the gradient is evenly
  spaced but the tick labels show the actual mm/h values.
-->
<template>
  <div class="colorbar-container">
    <div class="colorbar-title">mm/h</div>
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
      <div class="colorbar-gradient" />
    </div>
  </div>
</template>

<script setup>
// Thresholds from legend.txt: [0, 1, 2, 5, 10, 20, 30, 50, 75, 100]
// Colors are evenly spaced in colormap space (10 stops → 0%, 11.1%, 22.2%, ...)
const thresholds = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100]

const ticks = thresholds.map((value, i) => ({
  value,
  // Position as percentage (evenly spaced in colormap space)
  position: (i / (thresholds.length - 1)) * 100,
}))
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
  height: 220px;
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
  /*
    Colors from legend.txt (RGB values), evenly spaced in gradient.
    Bottom = low values (gray), Top = high values (purple).

    The 10 colors match the 10 thresholds:
      0 mm/h:   rgb(100, 100, 100)  — gray
      1 mm/h:   rgb(0, 120, 200)    — blue
      2 mm/h:   rgb(0, 200, 250)    — cyan
      5 mm/h:   rgb(0, 150, 0)      — dark green
      10 mm/h:  rgb(0, 250, 0)      — bright green
      20 mm/h:  rgb(250, 250, 0)    — yellow
      30 mm/h:  rgb(250, 150, 0)    — orange
      50 mm/h:  rgb(250, 0, 0)      — red
      75 mm/h:  rgb(180, 0, 0)      — dark red
      100 mm/h: rgb(220, 100, 250)  — purple
  */
  background: linear-gradient(
    to top,
    rgb(100, 100, 100) 0%,
    rgb(0, 120, 200) 11.1%,
    rgb(0, 200, 250) 22.2%,
    rgb(0, 150, 0) 33.3%,
    rgb(0, 250, 0) 44.4%,
    rgb(250, 250, 0) 55.6%,
    rgb(250, 150, 0) 66.7%,
    rgb(250, 0, 0) 77.8%,
    rgb(180, 0, 0) 88.9%,
    rgb(220, 100, 250) 100%
  );
}
</style>