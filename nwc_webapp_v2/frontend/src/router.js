/**
 * Vue Router - Client-side page navigation.
 *
 * HOW THIS WORKS:
 * In Streamlit, you have st.tabs(["Tab1", "Tab2"]) and switching tabs
 * re-runs the entire script.
 *
 * In Vue Router, each "tab" is a URL route:
 *   /realtime    → RealTimeView component
 *   /nowcasting  → NowcastingView component
 *   /prediction  → PredictionByDateView component
 *   etc.
 *
 * Benefits:
 * - Browser back/forward buttons work
 * - You can bookmark a specific tab
 * - Switching tabs is INSTANT (no server round-trip)
 * - Each page keeps its state when you navigate away and come back
 */
import { createRouter, createWebHistory } from 'vue-router'

import LiveView from './views/LiveView.vue'
import RealTimeView from './views/RealTimeView.vue'
import DataExplorerView from './views/DataExplorerView.vue'
import ModelComparisonView from './views/ModelComparisonView.vue'
import MetricsView from './views/MetricsView.vue'

const routes = [
  { path: '/', redirect: '/realtime' },

  {
    path: '/realtime',
    name: 'Real Time',
    component: LiveView,
  },
  {
    path: '/nowcasting',
    name: 'Nowcasting',
    component: RealTimeView,
  },
  {
    path: '/explorer',
    name: 'Data Explorer',
    component: DataExplorerView,
  },
  {
    path: '/comparison',
    name: 'Model Comparison',
    component: ModelComparisonView,
  },
  {
    path: '/metrics',
    name: 'Metrics Analysis',
    component: MetricsView,
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router