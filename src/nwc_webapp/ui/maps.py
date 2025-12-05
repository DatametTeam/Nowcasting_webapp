"""
Map creation and rendering utilities.
"""

import base64
import io
from datetime import datetime, timedelta
from pathlib import Path

import branca.colormap as cm
import folium
import folium.plugins
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit_folium import st_folium

from nwc_webapp.logging_config import setup_logger
from nwc_webapp.rendering.colormaps import cmap, norm

# Set up logger
logger = setup_logger(__name__)

# Template directory
TEMPLATE_DIR = Path(__file__).parent / "templates"


def load_template(template_name: str) -> str:
    """
    Load an HTML/JS template file.

    Args:
        template_name: Name of the template file (e.g., 'legend_css.html')

    Returns:
        Template content as string
    """
    template_path = TEMPLATE_DIR / template_name
    if not template_path.exists():
        logger.error(f"Template not found: {template_path}")
        return ""

    with open(template_path, 'r') as f:
        return f.read()


def parse_radar_positions():
    """
    Parse radar positions from the radar_positions.txt file.
    Format: RADARNAME = lon lat

    Returns:
        list: List of dictionaries with 'name', 'lon', and 'lat' keys
    """
    radar_file = Path(__file__).parent.parent / "resources" / "legends" / "radar_positions.txt"
    radars = []

    try:
        with open(radar_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Parse line format: "RADARNAME = lon lat"
                if '=' in line:
                    parts = line.split('=')
                    name = parts[0].strip()
                    coords = parts[1].strip().split()

                    if len(coords) >= 2:
                        try:
                            lon = float(coords[0])
                            lat = float(coords[1])
                            radars.append({
                                'name': name,
                                'lon': lon,
                                'lat': lat
                            })
                        except ValueError:
                            logger.warning(f"Could not parse coordinates for radar: {name}")

    except FileNotFoundError:
        logger.error(f"Radar positions file not found: {radar_file}")
    except Exception as e:
        logger.error(f"Error parsing radar positions: {e}")

    logger.info(f"Loaded {len(radars)} radar positions")
    return radars


def get_radar_icon_base64():
    """
    Convert radar icon to base64 for embedding in HTML.

    Returns:
        str: Base64 encoded PNG image
    """
    icon_path = Path(__file__).parent.parent.parent / "imgs" / "radar.png"

    try:
        with open(icon_path, 'rb') as f:
            icon_data = f.read()
            icon_base64 = base64.b64encode(icon_data).decode()
            return icon_base64
    except FileNotFoundError:
        logger.error(f"Radar icon not found: {icon_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading radar icon: {e}")
        return None


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
        tiles="Esri.WorldGrayCanvas",
        name="WorldGray",
    )

    folium.TileLayer(tiles="Esri.WorldImagery", name="Satellite", control=True).add_to(map)

    folium.TileLayer(tiles="OpenStreetMap.Mapnik", name="OSM", control=True).add_to(map)

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

    # Load radar positions and icon for markers
    radars = parse_radar_positions()
    radar_icon_base64 = get_radar_icon_base64()

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
            center = {"lat": 42.0, "lng": 12.5}  # Rome coordinates
            zoom = 6
    else:
        if (
            "old_center" in st.session_state
            and "old_zoom" in st.session_state
            and st.session_state["old_center"]
            and st.session_state["old_zoom"]
        ):
            center = st.session_state["old_center"]
            zoom = st.session_state["old_zoom"]
        else:
            center = {"lat": 42.0, "lng": 12.5}  # Rome coordinates
            zoom = 6

    # Create map with OSM as the only base layer, centered on Rome
    map = folium.Map(
        location=[center["lat"], center["lng"]],
        zoom_start=zoom,
        control_scale=False,
        tiles="OpenStreetMap",
        attr="OpenStreetMap",
    )

    # Add radar markers to the map
    if radar_icon_base64 and radars:
        for radar in radars:
            # Create custom icon using base64 encoded image
            icon = folium.CustomIcon(
                icon_image=f"data:image/png;base64,{radar_icon_base64}",
                icon_size=(24, 24),  # Size of the icon (not too big, not too small)
            )

            # Add marker with tooltip
            folium.Marker(
                location=[radar['lat'], radar['lon']],
                icon=icon,
                tooltip=radar['name']  # Tooltip shows radar name on hover
            ).add_to(map)

    if prediction:
        # ricreazione totale della mappa + predizione
        folium.raster_layers.ImageOverlay(
            image=rgba_img,
            bounds=[[35.0623, 4.51987], [47.5730, 20.4801]],
            mercator_project=False,
            origin="lower",
            name="NWC_pred",
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
            tick_labels=tick_labels,
        )

        colormap.caption = "Precipitation (mm/h)"
        # Add colorbar - by default it goes to top-right, we'll move it with CSS
        colormap.add_to(map)

        # Add custom CSS to reposition and resize colorbar to bottom-left
        style_element = folium.Element(load_template('legend_css.html'))
        # colormap.width = '50%'
        map.get_root().header.add_child(style_element)

    # Render map using full column width
    st_map = st_folium(map, width='stretch', height=700, returned_objects=["center", "zoom"])
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
                center = st.session_state.get("center", {"lat": 42.0, "lng": 12.5})
                zoom = st.session_state.get("zoom", 6)
                st.session_state["old_center"] = center
                st.session_state["old_zoom"] = zoom
                st.session_state["display_prediction"] = False
            else:
                center = st.session_state.get("old_center", {"lat": 42.0, "lng": 12.5})
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
            center = {"lat": 42.0, "lng": 12.5}
            zoom = 6
    else:
        if (
            "old_center" in st.session_state
            and "old_zoom" in st.session_state
            and st.session_state["old_center"]
            and st.session_state["old_zoom"]
        ):
            center = st.session_state["old_center"]
            zoom = st.session_state["old_zoom"]
        else:
            center = {"lat": 42.0, "lng": 12.5}
            zoom = 6

    # Create base map
    map_obj = folium.Map(
        location=[center["lat"], center["lng"]],
        zoom_start=zoom,
        control_scale=False,
        tiles="OpenStreetMap",
        attr="OpenStreetMap",
    )

    # Add geocoder
    folium.plugins.Geocoder(
        collapsed=False, position="topright", placeholder="Search for a location...", add_marker=True
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
            opacity=1.0 if idx == 0 else 0.0,  # Only first layer visible
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
        tick_labels=tick_labels,
    )
    colormap.caption = "Precipitation (mm/h)"
    colormap.add_to(map_obj)

    # Inject custom CSS for colorbar positioning
    style_element = folium.Element(
        """
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
    """
    )
    map_obj.get_root().header.add_child(style_element)

    # JavaScript for animation logic - using IIFE and proper event binding
    timestamps_js = str(timestamps)  # Convert to JS array format
    animation_script = folium.Element(
        f"""
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
    """
    )

    map_obj.get_root().html.add_child(animation_script)

    # Render map with stable key to reduce reloads
    # Key changes only when model changes, not on zoom/pan
    map_key = f"animated_map_{st.session_state.get('selected_model', 'default')}"

    st_map = st_folium(
        map_obj,
        width='stretch',
        height=700,
        returned_objects=[],  # Don't track zoom/center to avoid constant reloads
        key=map_key,
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
    # Parse radar positions and get icon
    radars = parse_radar_positions()
    radar_icon_base64 = get_radar_icon_base64()

    # Parse base datetime from latest_file
    base_dt = None
    if latest_file:
        try:
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
        img_pil.save(buffer, format="PNG")
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

        image_data_urls.append(
            {
                "timestamp": timestamp,  # Keep original for layer matching
                "display_label": display_label,
                "data_url": data_url,
            }
        )

    # Build JavaScript array of images with display labels
    images_js = "[\n"
    for img_data in image_data_urls:
        images_js += f"  {{timestamp: '{img_data['timestamp']}', label: '{img_data['display_label']}', url: '{img_data['data_url']}'}},\n"
    images_js += "]"

    # Build JavaScript array of radar positions
    radars_js = "[\n"
    for radar in radars:
        radars_js += f"  {{name: '{radar['name']}', lat: {radar['lat']}, lon: {radar['lon']}}},\n"
    radars_js += "]"

    # Get first display label for initial display
    first_label = image_data_urls[0]["display_label"] if image_data_urls else timestamps[0]

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

    # Load HTML template
    html_template = load_template('animated_map_full.html')

    # Replace template variables
    html_content = html_template.replace('{{max_index}}', str(len(timestamps) - 1))
    html_content = html_content.replace('{{first_label}}', first_label)
    html_content = html_content.replace('{{gradient_stops}}', gradient_stops)
    html_content = html_content.replace('{{radars_js}}', radars_js)
    html_content = html_content.replace('{{radar_icon_base64}}', radar_icon_base64)
    html_content = html_content.replace('{{images_js}}', images_js)


    # Render using Streamlit components
    components.html(html_content, height=720, scrolling=False)
