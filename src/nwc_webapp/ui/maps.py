"""
Map creation and rendering utilities.
"""
import base64
import io
import streamlit as st
import streamlit.components.v1 as components
import folium
import folium.plugins
import branca.colormap as cm
from streamlit_folium import st_folium
from PIL import Image
import numpy as np

from nwc_webapp.utils import cmap, norm
from nwc_webapp.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

def create_map():
    """
    Create a basic Folium map with multiple tile layers.

    Returns:
        Folium Map object with Gray Canvas, Satellite, and OSM layers
    """
    map = folium.Map(
        location=[42.5, 12.5],
        zoom_start=5,
        control_scale=False,
        tiles='Esri.WorldGrayCanvas',
        name="WorldGray",
    )

    folium.TileLayer(
        tiles='Esri.WorldImagery',
        name="Satellite",
        control=True
    ).add_to(map)

    folium.TileLayer(
        tiles='OpenStreetMap.Mapnik',
        name="OSM",
        control=True
    ).add_to(map)

    folium.LayerControl().add_to(map)

    return map


def create_only_map(rgba_img, prediction: bool = False):
    """
    Create a Folium map with optional prediction overlay.

    Args:
        rgba_img: RGBA image data for prediction overlay
        prediction: If True, add prediction overlay to map
    """
    logger.info("creating map")
    if st.session_state.selected_model and st.session_state.selected_time:
        if "display_prediction" in st.session_state:
            if st.session_state["display_prediction"]:
                # 3 --> nuova predizione da caricare, si aggiorna il centro
                center = st.session_state["center"]
                zoom = st.session_state["zoom"]

                st.session_state["old_center"] = center
                st.session_state["old_zoom"] = zoom

                st.session_state["display_prediction"] = False
            else:
                center = st.session_state["old_center"]
                zoom = st.session_state["old_zoom"]
        elif "old_center" in st.session_state and "old_zoom" in st.session_state:
            center = st.session_state["old_center"]
            zoom = st.session_state["old_zoom"]
        elif "center" in st.session_state and "zoom" in st.session_state:
            # 1 --> direttamente sull'overlay
            center = st.session_state["center"]
            zoom = st.session_state["zoom"]

            # 2 --> salvataggio come valori precedenti
            st.session_state["old_center"] = center
            st.session_state["old_zoom"] = zoom
        else:
            center = {'lat': 42.0, 'lng': 12.5}  # Rome coordinates
            zoom = 6
    else:
        if ("old_center" in st.session_state and "old_zoom" in st.session_state
                and st.session_state["old_center"] and st.session_state["old_zoom"]):
            center = st.session_state["old_center"]
            zoom = st.session_state["old_zoom"]
        else:
            center = {'lat': 42.0, 'lng': 12.5}  # Rome coordinates
            zoom = 6

    # Create map with OSM as the only base layer, centered on Rome
    map = folium.Map(
        location=[center['lat'], center['lng']],
        zoom_start=zoom,
        control_scale=False,
        tiles='OpenStreetMap',
        attr='OpenStreetMap'
    )

    if prediction:
        # ricreazione totale della mappa + predizione
        folium.raster_layers.ImageOverlay(
            image=rgba_img,
            bounds=[[35.0623, 4.51987], [47.5730, 20.4801]],
            mercator_project=False,
            origin="lower",
            name="NWC_pred"
            # opacity=0.5
        ).add_to(map)

        data_min = 0  # Minimum value in your data
        data_max = 100  # Maximum value in your data

        # Use all data values for color interpolation
        all_data_values = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100]
        normalized_values = norm(all_data_values)

        # But only display a subset of tick labels to avoid overlap
        tick_labels = [0, 5, 10, 20, 30, 50, 75, 100]

        colormap = cm.LinearColormap(
            colors=[cmap(n) for n in normalized_values],
            index=all_data_values,
            vmin=data_min,
            vmax=data_max,
            tick_labels=tick_labels
        )

        colormap.caption = "Precipitation (mm/h)"
        # Add colorbar - by default it goes to top-right, we'll move it with CSS
        colormap.add_to(map)

        # Add custom CSS to reposition and resize colorbar to bottom-left
        style_element = folium.Element("""
            <style>
                /* Move legend/colormap to bottom left and make it more compact */
                .legend {
                    position: fixed !important;
                    bottom: 15px !important;
                    left: 1px !important;
                    top: auto !important;
                    right: auto !important;
                    z-index: 1000 !important;
                    max-height: 250px !important;
                    max-width: 200px !important;
                }
                /* Make the SVG inside the legend more compact */
                .legend svg {
                    max-height: 200px !important;
                }
            </style>
        """)
        # colormap.width = '50%'
        map.get_root().header.add_child(style_element)

    # Render map using full column width
    st_map = st_folium(
        map,
        use_container_width=True,
        height=700,
        returned_objects=["center", "zoom"]
    )
    st.session_state["st_map"] = st_map
    if st_map and "center" in st_map.keys():
        st.session_state["center"] = st_map["center"]
        st.session_state["zoom"] = st_map["zoom"]


def create_animated_map(rgba_images_dict):
    """
    Create a Folium map with animated prediction overlays.

    Args:
        rgba_images_dict: Dictionary mapping time_option (e.g., "+5min") -> RGBA image array

    Returns:
        None (renders map to Streamlit)
    """
    # Determine map center and zoom from session state
    if st.session_state.selected_model:
        if "display_prediction" in st.session_state:
            if st.session_state["display_prediction"]:
                center = st.session_state.get("center", {'lat': 42.0, 'lng': 12.5})
                zoom = st.session_state.get("zoom", 6)
                st.session_state["old_center"] = center
                st.session_state["old_zoom"] = zoom
                st.session_state["display_prediction"] = False
            else:
                center = st.session_state.get("old_center", {'lat': 42.0, 'lng': 12.5})
                zoom = st.session_state.get("old_zoom", 6)
        elif "old_center" in st.session_state and "old_zoom" in st.session_state:
            center = st.session_state["old_center"]
            zoom = st.session_state["old_zoom"]
        elif "center" in st.session_state and "zoom" in st.session_state:
            center = st.session_state["center"]
            zoom = st.session_state["zoom"]
            st.session_state["old_center"] = center
            st.session_state["old_zoom"] = zoom
        else:
            center = {'lat': 42.0, 'lng': 12.5}
            zoom = 6
    else:
        if ("old_center" in st.session_state and "old_zoom" in st.session_state
                and st.session_state["old_center"] and st.session_state["old_zoom"]):
            center = st.session_state["old_center"]
            zoom = st.session_state["old_zoom"]
        else:
            center = {'lat': 42.0, 'lng': 12.5}
            zoom = 6

    # Create base map
    map_obj = folium.Map(
        location=[center['lat'], center['lng']],
        zoom_start=zoom,
        control_scale=False,
        tiles='OpenStreetMap',
        attr='OpenStreetMap'
    )

    # Add geocoder
    folium.plugins.Geocoder(
        collapsed=False,
        position='topright',
        placeholder='Search for a location...',
        add_marker=True
    ).add_to(map_obj)

    # Add all prediction layers with opacity control
    timestamps = list(rgba_images_dict.keys())
    for idx, (timestamp, rgba_img) in enumerate(rgba_images_dict.items()):
        layer = folium.raster_layers.ImageOverlay(
            image=rgba_img,
            bounds=[[35.0623, 4.51987], [47.5730, 20.4801]],
            mercator_project=False,
            origin="lower",
            name=f"pred_{timestamp}",
            opacity=1.0 if idx == 0 else 0.0  # Only first layer visible
        )
        layer.add_to(map_obj)

    # Add colorbar
    data_min = 0
    data_max = 100
    all_data_values = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100]
    normalized_values = norm(all_data_values)
    tick_labels = [0, 5, 10, 20, 30, 50, 75, 100]

    colormap = cm.LinearColormap(
        colors=[cmap(n) for n in normalized_values],
        index=all_data_values,
        vmin=data_min,
        vmax=data_max,
        tick_labels=tick_labels
    )
    colormap.caption = "Precipitation (mm/h)"
    colormap.add_to(map_obj)

    # Inject custom CSS for colorbar positioning
    style_element = folium.Element("""
        <style>
            .legend {
                position: fixed !important;
                bottom: 15px !important;
                left: 1px !important;
                top: auto !important;
                right: auto !important;
                z-index: 1000 !important;
                max-height: 250px !important;
                max-width: 200px !important;
            }
            .legend svg {
                max-height: 200px !important;
            }

            /* Animation controls styling - COMPACT */
            #animationControls {
                position: absolute;
                top: 8px;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(255, 255, 255, 0.95);
                padding: 6px 10px;
                border-radius: 5px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.3);
                z-index: 1000;
                display: flex;
                align-items: center;
                gap: 8px;
                font-family: Arial, sans-serif;
                font-size: 11px;
            }

            #playBtn {
                padding: 4px 10px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
                font-weight: bold;
                font-size: 11px;
            }

            #playBtn:hover {
                background: #45a049;
            }

            #playBtn.playing {
                background: #f44336;
            }

            #playBtn.playing:hover {
                background: #da190b;
            }

            #timeSlider {
                width: 180px;
                cursor: pointer;
                height: 4px;
            }

            #timeLabel {
                font-weight: bold;
                min-width: 50px;
                text-align: center;
                font-size: 11px;
            }

            #speedControl {
                display: flex;
                align-items: center;
                gap: 4px;
                font-size: 10px;
            }

            #speedSelect {
                padding: 2px 4px;
                border-radius: 3px;
                border: 1px solid #ccc;
                cursor: pointer;
                font-size: 10px;
            }
        </style>
    """)
    map_obj.get_root().header.add_child(style_element)

    # JavaScript for animation logic - using IIFE and proper event binding
    timestamps_js = str(timestamps)  # Convert to JS array format
    animation_script = folium.Element(f"""
        <div id="animationControls">
            <button id="playBtn">Play</button>
            <input type="range" id="timeSlider" min="0" max="{len(timestamps) - 1}" value="0" step="1">
            <span id="timeLabel">{timestamps[0]}</span>
            <div id="speedControl">
                <label>Speed:</label>
                <select id="speedSelect">
                    <option value="1000">Slow</option>
                    <option value="500" selected>Normal</option>
                    <option value="250">Fast</option>
                    <option value="100">Very Fast</option>
                </select>
            </div>
        </div>

        <script>
            (function() {{
                console.log('Animation script starting...');

                let currentFrame = 0;
                let isPlaying = false;
                let animationInterval = null;
                let animationSpeed = 500;
                const timestamps = {timestamps_js};
                const layers = {{}};
                let mapInstance = null;

                function initAnimation() {{
                    console.log('Initializing animation...');

                    // Find map - try multiple methods
                    const mapDiv = document.querySelector('.folium-map');
                    if (!mapDiv) {{
                        console.error('Map div not found');
                        setTimeout(initAnimation, 100);
                        return;
                    }}

                    const mapId = mapDiv.id;
                    console.log('Found map div with ID:', mapId);

                    // Try to get map instance
                    mapInstance = window[mapId];
                    if (!mapInstance || !mapInstance.eachLayer) {{
                        console.log('Map instance not ready, retrying...');
                        setTimeout(initAnimation, 100);
                        return;
                    }}

                    console.log('Map instance found!');

                    // Collect all prediction layers
                    mapInstance.eachLayer(function(layer) {{
                        if (layer.options && layer.options.name && layer.options.name.startsWith('pred_')) {{
                            const timestamp = layer.options.name.replace('pred_', '');
                            layers[timestamp] = layer;
                        }}
                    }});

                    console.log('Found layers:', Object.keys(layers));

                    if (Object.keys(layers).length === 0) {{
                        console.error('No prediction layers found!');
                        return;
                    }}

                    // Attach event listeners
                    const playBtn = document.getElementById('playBtn');
                    const slider = document.getElementById('timeSlider');
                    const speedSelect = document.getElementById('speedSelect');

                    if (!playBtn || !slider || !speedSelect) {{
                        console.error('Control elements not found');
                        return;
                    }}

                    playBtn.addEventListener('click', function() {{
                        console.log('Play button clicked');
                        togglePlay();
                    }});

                    slider.addEventListener('input', function(e) {{
                        console.log('Slider moved to:', e.target.value);
                        currentFrame = parseInt(e.target.value);
                        showFrame(currentFrame);
                        if (isPlaying) {{
                            togglePlay();
                        }}
                    }});

                    speedSelect.addEventListener('change', function(e) {{
                        console.log('Speed changed to:', e.target.value);
                        updateSpeed();
                    }});

                    console.log('Event listeners attached!');

                    // Show initial frame
                    showFrame(0);
                }}

                function showFrame(frameIndex) {{
                    console.log('Showing frame:', frameIndex, 'timestamp:', timestamps[frameIndex]);

                    if (!mapInstance) {{
                        console.error('Map instance not available');
                        return;
                    }}

                    // Hide all layers
                    for (const timestamp in layers) {{
                        layers[timestamp].setOpacity(0);
                    }}

                    // Show current frame
                    const timestamp = timestamps[frameIndex];
                    if (layers[timestamp]) {{
                        layers[timestamp].setOpacity(1);
                        console.log('Set opacity for layer:', timestamp);
                    }} else {{
                        console.error('Layer not found for timestamp:', timestamp);
                    }}

                    // Update UI
                    document.getElementById('timeSlider').value = frameIndex;
                    document.getElementById('timeLabel').textContent = timestamp;
                }}

                function togglePlay() {{
                    isPlaying = !isPlaying;
                    const playBtn = document.getElementById('playBtn');

                    if (isPlaying) {{
                        console.log('Starting animation');
                        playBtn.textContent = 'Pause';
                        playBtn.classList.add('playing');
                        animationInterval = setInterval(function() {{
                            currentFrame = (currentFrame + 1) % timestamps.length;
                            showFrame(currentFrame);
                        }}, animationSpeed);
                    }} else {{
                        console.log('Stopping animation');
                        playBtn.textContent = 'Play';
                        playBtn.classList.remove('playing');
                        clearInterval(animationInterval);
                    }}
                }}

                function updateSpeed() {{
                    const speedSelect = document.getElementById('speedSelect');
                    animationSpeed = parseInt(speedSelect.value);
                    console.log('Animation speed updated to:', animationSpeed);

                    // Restart animation if playing
                    if (isPlaying) {{
                        clearInterval(animationInterval);
                        animationInterval = setInterval(function() {{
                            currentFrame = (currentFrame + 1) % timestamps.length;
                            showFrame(currentFrame);
                        }}, animationSpeed);
                    }}
                }}

                // Start initialization when DOM is ready
                if (document.readyState === 'loading') {{
                    document.addEventListener('DOMContentLoaded', initAnimation);
                }} else {{
                    initAnimation();
                }}
            }})();
        </script>
    """)

    map_obj.get_root().html.add_child(animation_script)

    # Render map with stable key to reduce reloads
    # Key changes only when model changes, not on zoom/pan
    map_key = f"animated_map_{st.session_state.get('selected_model', 'default')}"

    st_map = st_folium(
        map_obj,
        use_container_width=True,
        height=700,
        returned_objects=[],  # Don't track zoom/center to avoid constant reloads
        key=map_key
    )

    # Note: We sacrifice real-time zoom/center tracking to avoid reloads
    # The map will remember its position within a session anyway via browser


def create_animated_map_html(rgba_images_dict, latest_file=None):
    """
    Create an animated map using custom HTML component (bypasses iframe restrictions).

    Args:
        rgba_images_dict: Dictionary mapping time_option -> RGBA image array
        latest_file: Latest SRI filename to extract base datetime
    """
    # Parse base datetime from latest_file
    base_dt = None
    if latest_file:
        try:
            from datetime import datetime, timedelta
            from pathlib import Path
            filename_clean = Path(latest_file).stem
            base_dt = datetime.strptime(filename_clean, "%d-%m-%Y-%H-%M")
        except:
            pass

    # Convert RGBA images to base64 data URLs with formatted timestamps
    timestamps = list(rgba_images_dict.keys())
    image_data_urls = []

    logger.info(f"Converting {len(rgba_images_dict)} images to base64...")
    for timestamp, rgba_img in rgba_images_dict.items():
        # Convert numpy array to PIL Image
        img_pil = Image.fromarray((rgba_img * 255).astype(np.uint8))

        # Convert to base64
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        data_url = f"data:image/png;base64,{img_base64}"

        # Create formatted label like "HH:MM (±Xmin)"
        display_label = timestamp
        if base_dt:
            try:
                # Extract minutes offset from timestamp (e.g., "+30min" -> 30, "-30min" -> -30)
                minutes_offset = int(timestamp.replace("min", "").replace("+", "").replace("-", ""))
                if timestamp.startswith("-"):
                    minutes_offset = -minutes_offset

                # Calculate actual datetime
                actual_dt = base_dt + timedelta(minutes=minutes_offset)
                # Show only HH:MM and offset
                display_label = f"{actual_dt.strftime('%H:%M')} ({timestamp})"
            except:
                pass

        image_data_urls.append({
            'timestamp': timestamp,  # Keep original for layer matching
            'display_label': display_label,
            'data_url': data_url
        })

    # Build JavaScript array of images with display labels
    images_js = "[\n"
    for img_data in image_data_urls:
        images_js += f"  {{timestamp: '{img_data['timestamp']}', label: '{img_data['display_label']}', url: '{img_data['data_url']}'}},\n"
    images_js += "]"

    # Get first display label for initial display
    first_label = image_data_urls[0]['display_label'] if image_data_urls else timestamps[0]

    # Generate colorbar gradient
    data_values = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100]
    colors = []
    for val in data_values:
        normalized = norm(val)
        rgba = cmap(normalized)
        rgb = f"rgb({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)})"
        colors.append(rgb)

    # Create gradient stops
    gradient_stops = ""
    for i, color in enumerate(colors):
        percent = (i / (len(colors) - 1)) * 100
        gradient_stops += f"{color} {percent}%, "
    gradient_stops = gradient_stops.rstrip(", ")

    # Create HTML with Leaflet map and animation controls
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <link rel="stylesheet" href="https://unpkg.com/leaflet-control-geocoder@2.4.0/dist/Control.Geocoder.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script src="https://unpkg.com/leaflet-control-geocoder@2.4.0/dist/Control.Geocoder.js"></script>
        <style>
            body {{ margin: 0; padding: 0; }}
            #map {{ width: 100%; height: 700px; }}

            /* Animation controls - fully responsive, scales with zoom */
            #animationControls {{
                position: fixed;
                top: 12px;  /* Moved down 2px */
                left: 60px;  /* Space for zoom controls */
                right: 10px;
                background: rgba(255, 255, 255, 0.95);
                padding: 1vh 1.2vw;  /* Responsive padding */
                border-radius: 0.6vh;
                box-shadow: 0 0.3vh 0.8vh rgba(0,0,0,0.3);
                z-index: 1000;
                display: flex;
                flex-direction: row;
                align-items: center;
                gap: 1.2vw;
                font-family: Arial, sans-serif;
                height: 8.5vh;  /* Taller */
                box-sizing: border-box;
            }}

            #playBtn {{
                padding: 0 1.5vw;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 0.4vh;
                cursor: pointer;
                font-weight: bold;
                font-size: 2.6vh;  /* Even bigger text */
                height: 70%;  /* 70% of bar height */
                min-width: 7vw;
                flex-shrink: 0;
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            #playBtn:hover {{ background: #45a049; }}
            #playBtn.playing {{ background: #f44336; }}
            #playBtn.playing:hover {{ background: #da190b; }}

            #timeSlider {{
                width: 30vw;  /* Fixed smaller width */
                cursor: pointer;
                height: 0.4vh;  /* Even smaller slider */
                margin: 0 0.8vw;
                flex-shrink: 0;
            }}

            #timeLabel {{
                font-weight: bold;
                flex: 1;  /* Take remaining space */
                min-width: 20vw;
                text-align: center;
                font-size: 2.8vh;  /* Even bigger text */
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}

            #speedSelect {{
                padding: 0 1vw;
                border-radius: 0.4vh;
                border: 1px solid #ccc;
                cursor: pointer;
                font-size: 2.4vh;  /* Even bigger text */
                font-weight: 500;
                height: 70%;  /* 70% of bar height, matching button */
                min-width: 7vw;
                background: white;
                flex-shrink: 0;
                box-sizing: border-box;
            }}

            /* Colorbar legend - horizontal, bottom left, compact */
            .colorbar-legend {{
                position: absolute;
                bottom: 30px;
                left: 10px;
                background: rgba(255, 255, 255, 0.9);
                padding: 4px 6px;
                border-radius: 3px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.3);
                z-index: 1000;
                font-family: Arial, sans-serif;
            }}

            .colorbar-legend .title {{
                font-weight: bold;
                margin-bottom: 2px;
                text-align: center;
                font-size: 8px;
            }}

            .colorbar-legend .gradient-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
            }}

            .colorbar-legend .gradient {{
                width: 200px;
                height: 6px;
                background: linear-gradient(to right, {gradient_stops});
                border: 1px solid #999;
            }}

            .colorbar-legend .labels {{
                margin-top: 2px;
                width: 200px;
                display: flex;
                flex-direction: row;
                justify-content: space-between;
                font-size: 7px;
                line-height: 1;
            }}

            /* Geocoder positioning - top right below control bar */
            .leaflet-control-geocoder {{
                position: fixed !important;
                top: 75px !important;  /* Closer to control bar */
                right: 10px !important;
                bottom: auto !important;
                left: auto !important;
                margin: 0 !important;
                max-width: 300px !important;
                z-index: 2000 !important;  /* Higher than control bar */
                display: flex !important;
                flex-direction: row !important;  /* Single row */
                align-items: center !important;
                visibility: visible !important;
                opacity: 1 !important;
                background: white !important;
                border-radius: 4px !important;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3) !important;
                padding: 2px 4px !important;
            }}

            .leaflet-control-geocoder-form {{
                margin: 0 !important;
                display: flex !important;
                flex-direction: row !important;  /* Single row */
                align-items: center !important;
                flex: 1 !important;
            }}

            .leaflet-control-geocoder-form input {{
                font-size: 13px !important;
                padding: 6px 10px !important;
                width: 220px !important;
                border: 1px solid #ccc !important;
                border-radius: 4px !important;
                margin-left: 4px !important;
            }}

            .leaflet-control-geocoder-icon {{
                display: inline-block !important;
                flex-shrink: 0 !important;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>

        <!-- Animation Controls -->
        <div id="animationControls">
            <button id="playBtn">Play</button>
            <input type="range" id="timeSlider" min="0" max="{len(timestamps) - 1}" value="0" step="1">
            <span id="timeLabel">{first_label}</span>
            <select id="speedSelect">
                <option value="1000">Slow</option>
                <option value="500" selected>Normal</option>
                <option value="250">Fast</option>
                <option value="100">Very Fast</option>
            </select>
        </div>

        <!-- Colorbar Legend -->
        <div class="colorbar-legend">
            <div class="title">Precip. (mm/h)</div>
            <div class="gradient-container">
                <div class="gradient"></div>
                <div class="labels">
                    <div>0</div>
                    <div>5</div>
                    <div>10</div>
                    <div>20</div>
                    <div>30</div>
                    <div>50</div>
                    <div>75</div>
                    <div>100</div>
                </div>
            </div>
        </div>

        <script>
            console.log('Initializing animated map...');

            // Initialize map
            const map = L.map('map').setView([42.0, 12.5], 6);

            // Add OpenStreetMap tiles
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© OpenStreetMap contributors'
            }}).addTo(map);

            // Add geocoder control (search only, no permanent markers)
            const geocoder = L.Control.geocoder({{
                collapsed: false,
                position: 'topright',
                placeholder: 'Search for a location...',
                errorMessage: 'Location not found',
                geocoder: L.Control.Geocoder.nominatim(),
                defaultMarkGeocode: false
            }}).addTo(map);

            // Pan to location without adding permanent marker
            geocoder.on('markgeocode', function(e) {{
                const latlng = e.geocode.center;
                map.setView(latlng, 10);
            }});

            // Image data
            const images = {images_js};
            // Bounds: [[south, west], [north, east]] - same as Folium
            // With origin="lower" in Folium, Y-axis starts from bottom
            // In Leaflet, we need to swap lat bounds to match
            const bounds = [[35.0623, 4.51987], [47.5730, 20.4801]];

            // Create image overlays
            const layers = [];
            images.forEach((img, idx) => {{
                const layer = L.imageOverlay(img.url, bounds, {{
                    opacity: idx === 0 ? 1 : 0
                }});
                layer.addTo(map);
                layers.push({{
                    layer: layer,
                    timestamp: img.timestamp,
                    label: img.label  // Store display label
                }});
            }});

            console.log('Loaded', layers.length, 'image layers');

            // Animation controls
            let currentFrame = 0;
            let isPlaying = false;
            let animationInterval = null;
            let animationSpeed = 500;

            const playBtn = document.getElementById('playBtn');
            const slider = document.getElementById('timeSlider');
            const timeLabel = document.getElementById('timeLabel');
            const speedSelect = document.getElementById('speedSelect');

            function showFrame(frameIndex) {{
                // Hide all layers
                layers.forEach(l => l.layer.setOpacity(0));

                // Show current frame
                if (frameIndex >= 0 && frameIndex < layers.length) {{
                    layers[frameIndex].layer.setOpacity(1);
                    slider.value = frameIndex;
                    timeLabel.textContent = layers[frameIndex].label;  // Use display label
                    console.log('Showing frame:', frameIndex, layers[frameIndex].label);
                }}
            }}

            function togglePlay() {{
                isPlaying = !isPlaying;

                if (isPlaying) {{
                    playBtn.textContent = 'Pause';
                    playBtn.classList.add('playing');
                    animationInterval = setInterval(() => {{
                        currentFrame = (currentFrame + 1) % layers.length;
                        showFrame(currentFrame);
                    }}, animationSpeed);
                    console.log('Animation started');
                }} else {{
                    playBtn.textContent = 'Play';
                    playBtn.classList.remove('playing');
                    clearInterval(animationInterval);
                    console.log('Animation stopped');
                }}
            }}

            function updateSpeed() {{
                animationSpeed = parseInt(speedSelect.value);
                console.log('Speed updated to:', animationSpeed);

                if (isPlaying) {{
                    clearInterval(animationInterval);
                    animationInterval = setInterval(() => {{
                        currentFrame = (currentFrame + 1) % layers.length;
                        showFrame(currentFrame);
                    }}, animationSpeed);
                }}
            }}

            // Event listeners
            playBtn.addEventListener('click', togglePlay);
            slider.addEventListener('input', (e) => {{
                currentFrame = parseInt(e.target.value);
                showFrame(currentFrame);
                if (isPlaying) togglePlay();
            }});
            speedSelect.addEventListener('change', updateSpeed);

            // Show initial frame
            showFrame(0);
            console.log('Animation controls ready!');

            // Auto-start animation
            togglePlay();
            console.log('Animation auto-started!');
        </script>
    </body>
    </html>
    """

    # Render using Streamlit components
    components.html(html_content, height=720, scrolling=False)