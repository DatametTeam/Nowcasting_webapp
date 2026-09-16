/**
 * basemaps.js — Shared base map definitions for every Leaflet map in the app.
 *
 * Leaflet layers can only live on one map at a time, so this module exports
 * a factory (createBaseLayers) that builds a fresh set of layers per map
 * instead of shared layer objects.
 */
import L from 'leaflet'

export const DEFAULT_BASE_LAYER = 'Esri Dark'

// Order here is the order shown in the Settings modal and the layer control
export const BASE_LAYER_NAMES = ['Esri Dark', 'Stadia Dark', 'OpenStreetMap', 'Satellite', 'Terrain']

// Base maps with a dark background: the map container gets the
// `dark-basemap` class, which switches icons/overlays to their light variants
export const DARK_BASE_LAYERS = new Set(['Esri Dark', 'Stadia Dark', 'Satellite'])

// Pane for label layers, above the radar overlays (overlayPane = 400) but
// below markers (600), so city names stay readable over precipitation
const LABELS_PANE = 'basemapLabels'

// Stadia serves tiles without a key on localhost; on a server it needs either
// the domain registered at client.stadiamaps.com or an API key set at build time
const STADIA_API_KEY = import.meta.env.VITE_STADIA_API_KEY
const STADIA_KEY_PARAM = STADIA_API_KEY ? `?api_key=${STADIA_API_KEY}` : ''

/** Map a saved setting to a valid layer name (old 'Dark' was CARTO, now gone). */
export function resolveBaseLayerName(name) {
  return BASE_LAYER_NAMES.includes(name) ? name : DEFAULT_BASE_LAYER
}

/** Build one fresh set of base layers for the given map, keyed by name. */
export function createBaseLayers(map) {
  if (!map.getPane(LABELS_PANE)) {
    const pane = map.createPane(LABELS_PANE)
    pane.style.zIndex = 450
    pane.style.pointerEvents = 'none'
  }

  const esriAttribution = 'Esri, HERE, Garmin, &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

  return {
    // Esri World Dark Gray Canvas: a label-free base plus a separate
    // reference layer (city names, borders) drawn in the labels pane
    'Esri Dark': L.layerGroup([
      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
        attribution: esriAttribution,
        maxNativeZoom: 16,
        maxZoom: 20,
      }),
      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
        pane: LABELS_PANE,
        maxNativeZoom: 16,
        maxZoom: 20,
      }),
    ]),
    'Stadia Dark': L.tileLayer(`https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png${STADIA_KEY_PARAM}`, {
      attribution: '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      maxZoom: 20,
    }),
    'OpenStreetMap': L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      maxZoom: 19,
    }),
    'Satellite': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: '&copy; <a href="https://www.esri.com/">Esri</a>',
      maxZoom: 18,
    }),
    'Terrain': L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://opentopomap.org">OpenTopoMap</a>',
      maxZoom: 17,
    }),
  }
}
