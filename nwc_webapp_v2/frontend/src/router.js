/**
 * Vue Router - Client-side page navigation.
 *
 * HOW THIS WORKS:
 * In Streamlit, you have st.tabs(["Tab1", "Tab2"]) and switching tabs
 * re-runs the entire script.
 *
 * In Vue Router, each "tab" is a URL route:
 *   /realtime    → RealTimeView component (live multi-product radar)
 *   /nowcasting  → NowcastingView component (model predictions)
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

import RealTimeView from './views/RealTimeView.vue'
import NowcastingView from './views/NowcastingView.vue'
import DataExplorerView from './views/DataExplorerView.vue'
import ModelComparisonView from './views/ModelComparisonView.vue'
import MetricsView from './views/MetricsView.vue'
import WR10View from './views/WR10View.vue'
import WR10ExplorerView from './views/WR10ExplorerView.vue'
import CagliariView from './views/CagliariView.vue'
import TorchiaroloView from './views/TorchiaroloView.vue'
import FssAssessmentView from './views/FssAssessmentView.vue'

const routes = [
  { path: '/', redirect: '/realtime' },

  {
    path: '/realtime',
    name: 'Real Time',
    component: RealTimeView,
  },
  {
    path: '/wr10',
    name: 'WR10',
    component: WR10View,
  },
  {
    path: '/wr10-explorer',
    name: 'WR10 Explorer',
    component: WR10ExplorerView,
  },
  {
    path: '/cagliari',
    name: 'Cagliari X-band',
    component: CagliariView,
  },
  {
    path: '/torchiarolo',
    name: 'Torchiarolo',
    component: TorchiaroloView,
  },
  {
    path: '/nowcasting',
    name: 'Nowcasting',
    component: NowcastingView,
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
  {
    path: '/assessment',
    name: 'RT Assessment',
    component: FssAssessmentView,
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router