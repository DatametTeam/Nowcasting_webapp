"""
Map creation and rendering utilities.
"""
import streamlit as st
import folium
import folium.plugins
import branca.colormap as cm
from streamlit_folium import st_folium

from nwc_webapp.utils import cmap, norm


def create_only_map(rgba_img, prediction: bool = False):
    """
    Create a Folium map with optional prediction overlay.

    Args:
        rgba_img: RGBA image data for prediction overlay
        prediction: If True, add prediction overlay to map
    """
    print("creating map")
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

    # Add geocoder (search bar) for location search in upper right
    folium.plugins.Geocoder(
        collapsed=False,
        position='topright',
        placeholder='Search for a location...',
        add_marker=True
    ).add_to(map)

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