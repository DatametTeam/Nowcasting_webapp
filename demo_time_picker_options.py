"""
Demo script showing different methods for displaying prediction availability.
Run with: streamlit run demo_time_picker_options.py
"""

import streamlit as st
from datetime import datetime, time, timedelta

# ============================================================================
# MOCK DATA - Simulating which predictions are available
# ============================================================================

def get_mock_available_times(model_name, date):
    """
    Mock function that returns which times have predictions available.
    In real implementation, this would scan the prediction directory.
    """
    # Simulate some random availability
    if model_name == "ConvLSTM":
        return {
            "08:00", "08:05", "08:10", "08:20", "08:30",
            "09:00", "09:15", "09:30",
            "10:00", "10:10", "10:20", "10:30", "10:40",
            "12:00", "12:30",
            "14:00", "14:05", "14:10", "14:15", "14:20", "14:25"
        }
    elif model_name == "ED_ConvLSTM":
        return {
            "08:00", "08:15", "08:30",
            "09:00", "09:30",
            "10:00", "10:30",
            "14:00", "14:30"
        }
    else:
        return {"08:00", "09:00", "10:00"}


def generate_time_slots(start_hour=0, end_hour=23, interval_minutes=5):
    """Generate all possible time slots for a day"""
    slots = []
    current = datetime.combine(datetime.today(), time(start_hour, 0))
    end = datetime.combine(datetime.today(), time(end_hour, 55))

    while current <= end:
        slots.append(current.strftime("%H:%M"))
        current += timedelta(minutes=interval_minutes)

    return slots


# ============================================================================
# OPTION 1: EMOJI INDICATORS IN SELECTBOX
# ============================================================================

def option1_emoji_selectbox():
    st.subheader("Option 1: Emoji Indicators in Selectbox")
    st.caption("✅ Easiest to implement | ✅ Clear visual feedback | ⚠️ Long list to scroll")

    col1, col2 = st.columns(2)

    with col1:
        selected_model = st.selectbox(
            "Model",
            ["ConvLSTM", "ED_ConvLSTM", "DynamicUnet"],
            key="opt1_model"
        )

    with col2:
        selected_date = st.date_input("Date", key="opt1_date")

    # Get available times for this model
    available_times = get_mock_available_times(selected_model, selected_date)
    all_times = generate_time_slots(8, 16, 5)  # 8 AM to 4 PM for demo

    # Create formatted options with emoji
    time_options = []
    for t in all_times:
        if t in available_times:
            time_options.append(f"🟢 {t}")
        else:
            time_options.append(f"⚪ {t}")

    selected = st.selectbox(
        "Select Time (🟢 = available, ⚪ = not available)",
        time_options,
        key="opt1_time"
    )

    # Extract actual time from selection
    actual_time = selected.split(" ")[1]

    if "🟢" in selected:
        st.success(f"✅ Prediction available for {selected_model} at {actual_time}")
    else:
        st.info(f"ℹ️ No prediction for {selected_model} at {actual_time}")


# ============================================================================
# OPTION 2: CUSTOM HTML TIME GRID
# ============================================================================

def option2_html_grid():
    st.subheader("Option 2: Custom HTML Time Grid")
    st.caption("✅ Most visual | ✅ See all at once | ⚠️ More complex | ⚠️ Needs JavaScript")

    col1, col2 = st.columns(2)

    with col1:
        selected_model = st.selectbox(
            "Model",
            ["ConvLSTM", "ED_ConvLSTM", "DynamicUnet"],
            key="opt2_model"
        )

    with col2:
        selected_date = st.date_input("Date", key="opt2_date")

    # Get available times
    available_times = get_mock_available_times(selected_model, selected_date)
    all_times = generate_time_slots(8, 16, 5)

    # Create HTML grid
    html = """
    <style>
        .time-grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 5px;
            margin: 20px 0;
        }
        .time-slot {
            padding: 8px;
            text-align: center;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            border: 2px solid transparent;
        }
        .time-slot.available {
            background-color: #28a745;
            color: white;
        }
        .time-slot.unavailable {
            background-color: #e0e0e0;
            color: #666;
        }
        .time-slot:hover {
            border-color: #007bff;
            transform: scale(1.05);
        }
        .time-slot.selected {
            border-color: #ff6b6b;
            box-shadow: 0 0 10px rgba(255, 107, 107, 0.5);
        }
    </style>
    <div class="time-grid">
    """

    for t in all_times:
        status = "available" if t in available_times else "unavailable"
        html += f'<div class="time-slot {status}">{t}</div>\n'

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)

    st.info("💡 In a real implementation, clicking a time slot would update the selection using JavaScript + Streamlit session state")

    # Fallback selector
    selected_time = st.selectbox("Or select from dropdown:", all_times, key="opt2_time_fallback")

    if selected_time in available_times:
        st.success(f"✅ Prediction available for {selected_model} at {selected_time}")
    else:
        st.info(f"ℹ️ No prediction for {selected_model} at {selected_time}")


# ============================================================================
# OPTION 3: AVAILABILITY HEATMAP/TIMELINE
# ============================================================================

def option3_heatmap():
    st.subheader("Option 3: Native Time Picker + Availability Heatmap")
    st.caption("✅ Best of both worlds | ✅ Native picker | ⚠️ Takes vertical space")

    col1, col2 = st.columns(2)

    with col1:
        selected_model = st.selectbox(
            "Model",
            ["ConvLSTM", "ED_ConvLSTM", "DynamicUnet"],
            key="opt3_model"
        )

    with col2:
        selected_date = st.date_input("Date", key="opt3_date")

    # Native time picker
    selected_time = st.time_input(
        "Select Time",
        value=time(10, 0),
        step=timedelta(minutes=5),
        key="opt3_time"
    )

    st.markdown("**Prediction Availability Timeline:**")

    # Get available times
    available_times = get_mock_available_times(selected_model, selected_date)

    # Create horizontal timeline visualization
    hours = list(range(8, 17))

    timeline_html = """
    <style>
        .timeline {
            display: flex;
            align-items: center;
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .hour-block {
            flex: 1;
            text-align: center;
            position: relative;
        }
        .hour-label {
            font-size: 11px;
            font-weight: bold;
            color: #666;
            margin-bottom: 5px;
        }
        .availability-bar {
            height: 30px;
            border-radius: 4px;
            display: flex;
            gap: 2px;
        }
        .time-tick {
            flex: 1;
            height: 100%;
        }
        .time-tick.available {
            background-color: #28a745;
        }
        .time-tick.unavailable {
            background-color: #ddd;
        }
    </style>
    <div class="timeline">
    """

    for hour in hours:
        timeline_html += f'<div class="hour-block">'
        timeline_html += f'<div class="hour-label">{hour:02d}:00</div>'
        timeline_html += '<div class="availability-bar">'

        # 12 ticks per hour (5-minute intervals)
        for minute in range(0, 60, 5):
            time_str = f"{hour:02d}:{minute:02d}"
            status = "available" if time_str in available_times else "unavailable"
            timeline_html += f'<div class="time-tick {status}" title="{time_str}"></div>'

        timeline_html += '</div></div>'

    timeline_html += '</div>'
    timeline_html += '<div style="text-align: center; margin-top: 5px; font-size: 12px; color: #666;">🟩 = Available | ⬜ = Not available</div>'

    st.markdown(timeline_html, unsafe_allow_html=True)

    # Check selected time
    selected_time_str = selected_time.strftime("%H:%M")
    if selected_time_str in available_times:
        st.success(f"✅ Prediction available for {selected_model} at {selected_time_str}")
    else:
        st.info(f"ℹ️ No prediction for {selected_model} at {selected_time_str}")


# ============================================================================
# OPTION 4: REAL-TIME BADGE INDICATOR
# ============================================================================

def option4_badge():
    st.subheader("Option 4: Native Picker + Real-time Status Badge")
    st.caption("✅ Minimal changes | ✅ Clean UI | ⚠️ No preview of other times")

    col1, col2 = st.columns(2)

    with col1:
        selected_model = st.selectbox(
            "Model",
            ["ConvLSTM", "ED_ConvLSTM", "DynamicUnet"],
            key="opt4_model"
        )

    with col2:
        selected_date = st.date_input("Date", key="opt4_date")

    # Native time picker
    selected_time = st.time_input(
        "Select Time",
        value=time(10, 0),
        step=timedelta(minutes=5),
        key="opt4_time"
    )

    # Get available times
    available_times = get_mock_available_times(selected_model, selected_date)
    selected_time_str = selected_time.strftime("%H:%M")

    # Real-time status badge
    if selected_time_str in available_times:
        st.success(f"✅ Prediction available for **{selected_model}** at **{selected_time_str}**")

        # Show additional info about other models
        other_models = ["ConvLSTM", "ED_ConvLSTM", "DynamicUnet"]
        other_models.remove(selected_model)

        other_available = []
        for model in other_models:
            if selected_time_str in get_mock_available_times(model, selected_date):
                other_available.append(model)

        if other_available:
            st.info(f"ℹ️ Also available for: {', '.join(other_available)}")
    else:
        st.warning(f"⚠️ No prediction for **{selected_model}** at **{selected_time_str}**")
        st.info("💡 Click 'Check/Compute Prediction' to generate it")

        # Show which models DO have it
        available_models = []
        for model in ["ConvLSTM", "ED_ConvLSTM", "DynamicUnet"]:
            if selected_time_str in get_mock_available_times(model, selected_date):
                available_models.append(model)

        if available_models:
            st.info(f"ℹ️ Available for other models: {', '.join(available_models)}")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(layout="wide", page_title="Time Picker Options Demo")

    st.title("🕐 Time Picker Options Demo")
    st.markdown("---")
    st.markdown("""
    This demo shows **4 different approaches** to indicate prediction availability in the time picker.

    **Legend:**
    - 🟢 Green = Prediction available
    - ⚪ White/Gray = No prediction
    - Different models have different availability patterns (try switching models!)
    """)
    st.markdown("---")

    # Show all 4 options
    option1_emoji_selectbox()
    st.markdown("---")

    option2_html_grid()
    st.markdown("---")

    option3_heatmap()
    st.markdown("---")

    option4_badge()
    st.markdown("---")

    # Summary
    st.header("📊 Comparison Summary")

    comparison_data = """
    | Feature | Option 1<br>Emoji Selectbox | Option 2<br>HTML Grid | Option 3<br>Heatmap | Option 4<br>Badge |
    |---------|---------------------------|---------------------|-------------------|-----------------|
    | **Ease of Implementation** | ⭐⭐⭐⭐⭐ Very Easy | ⭐⭐⭐ Medium | ⭐⭐⭐⭐ Easy | ⭐⭐⭐⭐⭐ Very Easy |
    | **Visual Feedback** | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Moderate |
    | **Performance** | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Best |
    | **Screen Space** | ⭐⭐⭐ Compact | ⭐⭐⭐⭐ Medium | ⭐⭐ Takes space | ⭐⭐⭐⭐⭐ Very compact |
    | **User Experience** | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very good | ⭐⭐⭐ Good |
    | **See all times at once** | ✅ Yes (scrolling) | ✅ Yes | ✅ Yes | ❌ No |
    | **Native Streamlit** | ✅ Yes | ⚠️ HTML/CSS | ✅ Yes | ✅ Yes |
    """

    st.markdown(comparison_data)

    st.markdown("---")
    st.success("💡 **Recommendation**: Start with Option 1 (easiest) or Option 3 (best UX)")


if __name__ == "__main__":
    main()