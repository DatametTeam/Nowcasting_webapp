/**
 * basemaps.js — Shared base map definitions for every Leaflet map in the app.
 *
 * Leaflet layers can only live on one map at a time, so this module exports
 * a factory (createBaseLayers) that builds a fresh set of layers per map
 * instead of shared layer objects.
 */
import L from 'leaflet'

export const DEFAULT_BASE_LAYER = 'Stamen Toner Dark'

// Order here is the order shown in the Settings modal and the layer control
export const BASE_LAYER_NAMES = ['Stamen Toner Dark', 'OpenStreetMap', 'Satellite', 'Terrain']

// Base maps with a dark background: the map container gets the
// `dark-basemap` class, which switches icons/overlays to their light variants
export const DARK_BASE_LAYERS = new Set(['Stamen Toner Dark', 'Satellite'])

// Stadia serves tiles without a key on localhost; on a server it needs either
// the domain registered at client.stadiamaps.com or an API key set at build time
const STADIA_API_KEY = import.meta.env.VITE_STADIA_API_KEY
const STADIA_KEY_PARAM = STADIA_API_KEY ? `?api_key=${STADIA_API_KEY}` : ''

/** Map a saved setting to a valid layer name (old dark maps fall back to the default). */
export function resolveBaseLayerName(name) {
  return BASE_LAYER_NAMES.includes(name) ? name : DEFAULT_BASE_LAYER
}

/** Build one fresh set of base layers for a map, keyed by name. */
export function createBaseLayers() {
  return {
    'Stamen Toner Dark': L.tileLayer(`https://tiles.stadiamaps.com/tiles/stamen_toner_dark/{z}/{x}/{y}{r}.png${STADIA_KEY_PARAM}`, {
      attribution: '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> &copy; <a href="https://stamen.com/">Stamen Design</a> &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
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
