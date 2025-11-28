"""
Demo: Calendar-based Prediction Availability Browser (5-minute intervals)
Shows a monthly calendar with daily aggregated status, and detailed 5-minute intervals when clicking a date.

Run with: streamlit run demo_calendar_availability.py
"""

import calendar
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================================
# MOCK DATA GENERATOR (5-MINUTE INTERVALS)
# ============================================================================

def generate_mock_availability_data():
    """
    Generate mock availability data for 5-minute intervals.
    Returns dict with structure:
        {
            date_str: {
                time_str: {
                    'groundtruth': bool,
                    'models': {
                        'ConvLSTM': bool,
                        ...
                    }
                }
            }
        }
    """
    data = {}
    models = ['ConvLSTM', 'ED_ConvLSTM', 'DynamicUnet', 'PySTEPS']

    # Generate data for January 2025
    start_date = datetime(2025, 1, 1)

    for day_offset in range(31):  # January has 31 days
        current_date = start_date + timedelta(days=day_offset)
        date_str = current_date.strftime('%Y-%m-%d')

        data[date_str] = {}

        # Generate 5-minute intervals for each day (288 intervals = 24h * 12 per hour)
        for hour in range(24):
            for minute in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
                time_str = f"{hour:02d}:{minute:02d}"

                # Simulate different scenarios based on day and time
                day = current_date.day

                # Some days have no groundtruth at all
                if day in [3, 4, 12, 25]:
                    data[date_str][time_str] = {
                        'groundtruth': False,
                        'models': {model: False for model in models}
                    }

                # Some time slots missing groundtruth (e.g., night hours)
                elif hour < 6 or hour > 22:
                    data[date_str][time_str] = {
                        'groundtruth': False,
                        'models': {model: False for model in models}
                    }

                # Some days have groundtruth but no predictions
                elif day in [7, 14, 21]:
                    data[date_str][time_str] = {
                        'groundtruth': True,
                        'models': {model: False for model in models}
                    }

                # Some time slots have partial predictions
                elif day in [5, 6, 10, 15, 20, 28, 29]:
                    data[date_str][time_str] = {
                        'groundtruth': True,
                        'models': {
                            'ConvLSTM': (hour + minute) % 2 == 0,
                            'ED_ConvLSTM': (hour + minute) % 3 == 0,
                            'DynamicUnet': (hour + minute) % 5 == 0,
                            'PySTEPS': hour % 2 == 0,
                        }
                    }

                # Rest have complete data
                else:
                    data[date_str][time_str] = {
                        'groundtruth': True,
                        'models': {model: True for model in models}
                    }

    return data


def aggregate_daily_status(date_data, models):
    """
    Aggregate all 5-minute intervals for a day to get overall daily status.

    Returns:
        - 'red': No groundtruth for any interval
        - 'blue': Has groundtruth but no predictions for any interval
        - 'yellow': Some predictions missing
        - 'green': All predictions available for all intervals
        - 'gray': No data
    """
    if not date_data:
        return 'gray'

    has_any_groundtruth = False
    has_any_predictions = False
    all_complete = True

    for time_str, interval_data in date_data.items():
        if interval_data['groundtruth']:
            has_any_groundtruth = True

            # Check if this interval has all model predictions
            interval_complete = all(interval_data['models'].get(model, False) for model in models)

            if interval_complete:
                has_any_predictions = True
            else:
                all_complete = False

            # Check if this interval has any predictions
            if any(interval_data['models'].get(model, False) for model in models):
                has_any_predictions = True

    if not has_any_groundtruth:
        return 'red'
    elif not has_any_predictions:
        return 'blue'
    elif all_complete:
        return 'green'
    else:
        return 'yellow'


def get_interval_status(interval_data, models):
    """Get status for a single 5-minute interval"""
    if not interval_data['groundtruth']:
        return 'red'

    available_count = sum(1 for model in models if interval_data['models'].get(model, False))

    if available_count == 0:
        return 'blue'
    elif available_count == len(models):
        return 'green'
    else:
        return 'yellow'


def create_tooltip_content(interval_data, models, time_str):
    """Generate HTML content for tooltip"""
    tooltip = f"<strong>{time_str}</strong><br>"

    # Groundtruth status
    if interval_data['groundtruth']:
        tooltip += "✅ Groundtruth: Available<br>"
    else:
        tooltip += "❌ Groundtruth: Missing<br>"

    tooltip += "<br><strong>Predictions:</strong><br>"

    # Model statuses
    for model in models:
        status = interval_data['models'].get(model, False)
        icon = "✅" if status else "❌"
        tooltip += f"{icon} {model}<br>"

    return tooltip


# ============================================================================
# CALENDAR RENDERING (MONTH VIEW)
# ============================================================================

def render_calendar(year, month):
    """Render a simple calendar for reference"""

    cal = calendar.monthcalendar(year, month)
    month_name = calendar.month_name[month]

    css = """
    <style>
        .calendar-container {
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .calendar-header {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #333;
        }

        .calendar-grid {
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 8px;
            max-width: 800px;
            margin: 0 auto;
        }

        .calendar-day-header {
            text-align: center;
            font-weight: bold;
            padding: 12px;
            color: #666;
            font-size: 14px;
        }

        .calendar-day {
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            background: #f8f9fa;
            border: 2px solid #e0e0e0;
            color: #333;
        }

        .calendar-day.empty {
            background: transparent;
            border: none;
        }
    </style>
    """

    html = css
    html += '<div class="calendar-container">'
    html += f'<div class="calendar-header">{month_name} {year}</div>'
    html += '<div class="calendar-grid">'

    # Day headers
    for day_name in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        html += f'<div class="calendar-day-header">{day_name}</div>'

    # Calendar days (plain, no colors)
    for week_idx, week in enumerate(cal):
        for day in week:
            if day == 0:
                html += '<div class="calendar-day empty"></div>'
            else:
                html += f'<div class="calendar-day">{day}</div>'

    html += '</div>'  # Close grid
    html += '</div>'  # Close container

    return html


# ============================================================================
# DAY HEATMAP RENDERING (5-MINUTE INTERVALS)
# ============================================================================

def render_day_heatmap(date_str, date_data, models):
    """Render a compact heatmap for a single day showing all 288 5-minute intervals"""

    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    formatted_date = date_obj.strftime('%B %d, %Y')

    css = """
    <style>
        .heatmap-container {
            margin: 20px auto;
            padding: 30px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            max-width: 1200px;
        }

        .heatmap-header {
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 30px;
            color: #333;
        }

        .heatmap-wrapper {
            display: flex;
            gap: 10px;
        }

        .hour-labels {
            display: flex;
            flex-direction: column;
            justify-content: space-around;
            padding-top: 30px;
        }

        .hour-label {
            font-size: 9px;
            color: #666;
            font-weight: 600;
            height: 15px;
            display: flex;
            align-items: center;
        }

        .heatmap-main {
            flex: 1;
        }

        .minute-labels {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 1px;
            margin-bottom: 5px;
            text-align: center;
        }

        .minute-label {
            font-size: 8px;
            color: #999;
        }

        .heatmap-grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            grid-template-rows: repeat(24, 1fr);
            gap: 1px;
            background: #f5f5f5;
            padding: 3px;
            border-radius: 6px;
        }

        .heatmap-cell {
            position: relative;
            aspect-ratio: 1;
            border-radius: 2px;
            cursor: pointer;
            transition: all 0.1s ease;
            border: 1px solid transparent;
        }

        .heatmap-cell:hover {
            transform: scale(2.5);
            z-index: 1000;
            box-shadow: 0 4px 16px rgba(0,0,0,0.5);
            border: 2px solid #333;
        }

        /* Status colors - more vibrant for heatmap */
        .heatmap-cell.red {
            background: #f44336;
        }

        .heatmap-cell.blue {
            background: #2196f3;
        }

        .heatmap-cell.yellow {
            background: #ffc107;
        }

        .heatmap-cell.green {
            background: #4caf50;
        }

        /* Tooltip - smaller and more compact */
        .heatmap-cell .tooltip {
            visibility: hidden;
            position: fixed;
            background-color: #333;
            color: white;
            text-align: left;
            border-radius: 6px;
            padding: 8px 10px;
            z-index: 10000;
            white-space: nowrap;
            font-size: 11px;
            line-height: 1.4;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            opacity: 0;
            transition: opacity 0.15s, visibility 0.15s;
            pointer-events: none;
        }

        .heatmap-cell:hover .tooltip {
            visibility: visible;
            opacity: 1;
        }

        .legend {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 30px;
            flex-wrap: wrap;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 14px;
        }

        .legend-box {
            width: 24px;
            height: 24px;
            border-radius: 4px;
        }

        .legend-box.red { background: #f44336; }
        .legend-box.blue { background: #2196f3; }
        .legend-box.yellow { background: #ffc107; }
        .legend-box.green { background: #4caf50; }
    </style>
    """

    html = css
    html += '<div class="heatmap-container">'
    html += f'<div class="heatmap-header">{formatted_date} - 5-Minute Intervals</div>'

    html += '<div class="heatmap-wrapper">'

    # Hour labels on the left
    html += '<div class="hour-labels">'
    for hour in range(24):
        html += f'<div class="hour-label">{hour:02d}h</div>'
    html += '</div>'

    # Main heatmap
    html += '<div class="heatmap-main">'

    # Minute labels at the top
    html += '<div class="minute-labels">'
    for minute in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
        html += f'<div class="minute-label">:{minute:02d}</div>'
    html += '</div>'

    # Heatmap grid (24 rows × 12 columns = 288 cells)
    html += '<div class="heatmap-grid">'

    for hour in range(24):
        for minute in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
            time_str = f"{hour:02d}:{minute:02d}"
            interval_data = date_data.get(time_str, {
                'groundtruth': False,
                'models': {model: False for model in models}
            })

            status = get_interval_status(interval_data, models)
            tooltip_content = create_tooltip_content(interval_data, models, time_str)

            html += f'<div class="heatmap-cell {status}"><div class="tooltip">{tooltip_content}</div></div>'

    html += '</div>'  # Close heatmap-grid
    html += '</div>'  # Close heatmap-main
    html += '</div>'  # Close heatmap-wrapper

    # Legend
    html += '''
    <div class="legend">
        <div class="legend-item">
            <div class="legend-box red"></div>
            <span>No Groundtruth</span>
        </div>
        <div class="legend-item">
            <div class="legend-box blue"></div>
            <span>GT Only (No Predictions)</span>
        </div>
        <div class="legend-item">
            <div class="legend-box yellow"></div>
            <span>Some Predictions Missing</span>
        </div>
        <div class="legend-item">
            <div class="legend-box green"></div>
            <span>All Predictions Available</span>
        </div>
    </div>
    '''

    html += '</div>'  # Close heatmap-container

    return html


# ============================================================================
# STATISTICS PANEL
# ============================================================================

def render_statistics(availability_data, models, year, month):
    """Render statistics about prediction availability"""

    # Filter data for the selected month
    month_data = {
        date_str: data
        for date_str, data in availability_data.items()
        if date_str.startswith(f"{year}-{month:02d}")
    }

    total_days = len(month_data)

    # Count by daily aggregated status
    status_counts = {'red': 0, 'blue': 0, 'yellow': 0, 'green': 0}

    for date_str, date_data in month_data.items():
        status = aggregate_daily_status(date_data, models)
        status_counts[status] = status_counts.get(status, 0) + 1

    # Model-specific stats (count days where model has at least some data)
    model_stats = {}
    for model in models:
        days_with_data = 0
        for date_data in month_data.values():
            has_any_data = any(
                interval['groundtruth'] and interval['models'].get(model, False)
                for interval in date_data.values()
            )
            if has_any_data:
                days_with_data += 1
        model_stats[model] = days_with_data

    # Render
    st.markdown("### 📊 Monthly Statistics (Daily Aggregated)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Complete Days", status_counts['green'],
                  delta=f"{status_counts['green']/total_days*100:.1f}%")

    with col2:
        st.metric("Partial Days", status_counts['yellow'],
                  delta=f"{status_counts['yellow']/total_days*100:.1f}%")

    with col3:
        st.metric("GT Only", status_counts['blue'],
                  delta=f"{status_counts['blue']/total_days*100:.1f}%")

    with col4:
        st.metric("No GT", status_counts['red'],
                  delta=f"{status_counts['red']/total_days*100:.1f}%")

    st.markdown("---")
    st.markdown("### 🤖 Model Availability (Days with Data)")

    model_cols = st.columns(len(models))
    for idx, model in enumerate(models):
        with model_cols[idx]:
            count = model_stats[model]
            percentage = count / total_days * 100
            st.metric(model, f"{count}/{total_days}", delta=f"{percentage:.1f}%")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(layout="wide", page_title="Prediction Availability Calendar")

    # Initialize session state
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'calendar'  # 'calendar' or 'day'
    if 'selected_date' not in st.session_state:
        st.session_state.selected_date = None

    st.title("📅 Prediction Availability Browser (5-Minute Intervals)")
    st.markdown("""
    Browse prediction availability at 5-minute intervals.
    **Click any date** to see detailed 5-minute intervals for that day.
    """)

    # Controls in main page
    st.markdown("### ⚙️ Settings")

    col1, col2, col3, col4 = st.columns([1, 1, 3, 1])

    with col1:
        selected_month = st.selectbox(
            "Month",
            range(1, 13),
            format_func=lambda x: calendar.month_name[x],
            index=0  # January
        )

    with col2:
        selected_year = st.selectbox(
            "Year",
            [2024, 2025],
            index=1  # 2025
        )

    with col3:
        st.markdown("**Select Models:**")
        all_models = ['ConvLSTM', 'ED_ConvLSTM', 'DynamicUnet', 'PySTEPS']
        model_cols = st.columns(len(all_models))
        selected_models = []

        for idx, model in enumerate(all_models):
            with model_cols[idx]:
                if st.checkbox(model, value=True, key=f"model_{model}"):
                    selected_models.append(model)

        if not selected_models:
            st.warning("⚠️ Select at least one model")
            selected_models = all_models

    with col4:
        if st.session_state.view_mode == 'day':
            if st.button("← Back to Calendar", use_container_width=True):
                st.session_state.view_mode = 'calendar'
                st.session_state.selected_date = None
                st.rerun()

    st.markdown("---")

    # Generate mock data
    availability_data = generate_mock_availability_data()

    # VIEW MODE: CALENDAR (Month overview)
    if st.session_state.view_mode == 'calendar':
        # Render statistics
        render_statistics(availability_data, selected_models, selected_year, selected_month)

        st.markdown("---")

        # Show simple calendar for reference
        col_cal, col_pick = st.columns([2, 1])

        with col_cal:
            st.markdown("### 📅 Calendar Reference")
            calendar_html = render_calendar(selected_year, selected_month)
            components.html(calendar_html, height=500, scrolling=False)

        with col_pick:
            st.markdown("### 🔍 Select a Date")
            st.markdown("Pick a date to view its 5-minute interval heatmap:")

            selected_date = st.date_input(
                "Date",
                value=datetime(selected_year, selected_month, 1),
                min_value=datetime(selected_year, selected_month, 1),
                max_value=datetime(selected_year, selected_month, calendar.monthrange(selected_year, selected_month)[1])
            )

            if st.button("View Heatmap", type="primary", use_container_width=True):
                st.session_state.selected_date = selected_date.strftime('%Y-%m-%d')
                st.session_state.view_mode = 'day'
                st.rerun()

        # Additional info
        st.markdown("---")
        st.markdown("### 💡 How to Use")
        st.markdown("""
        - The **monthly calendar** shows aggregated daily status
        - **Select a date** and click "View 5-Minute Intervals" to see detailed timeline
        - Each day has **288 5-minute intervals** (24 hours × 12 per hour)
        - Colors indicate data availability:
          - 🔴 **Red**: No groundtruth data
          - 🔵 **Blue**: Groundtruth only (no predictions)
          - 🟡 **Yellow**: Some predictions missing
          - 🟢 **Green**: All predictions available
        """)

    # VIEW MODE: DAY (5-minute intervals)
    elif st.session_state.view_mode == 'day':
        date_str = st.session_state.selected_date
        date_data = availability_data.get(date_str, {})

        st.markdown(f"### 🔥 Heatmap: {date_str}")
        st.markdown("**288 5-minute intervals** displayed as a compact heatmap. Hover over any cell to see details.")

        # Render day heatmap
        heatmap_html = render_day_heatmap(date_str, date_data, selected_models)
        components.html(heatmap_html, height=900, scrolling=False)

        # Stats for this specific day
        st.markdown("---")
        st.markdown("### 📊 Daily Statistics")

        total_intervals = len(date_data)
        status_counts = {'red': 0, 'blue': 0, 'yellow': 0, 'green': 0}

        for interval_data in date_data.values():
            status = get_interval_status(interval_data, selected_models)
            status_counts[status] = status_counts.get(status, 0) + 1

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Complete Intervals", status_counts['green'],
                      delta=f"{status_counts['green']/total_intervals*100:.1f}%")
        with col2:
            st.metric("Partial Intervals", status_counts['yellow'],
                      delta=f"{status_counts['yellow']/total_intervals*100:.1f}%")
        with col3:
            st.metric("GT Only", status_counts['blue'],
                      delta=f"{status_counts['blue']/total_intervals*100:.1f}%")
        with col4:
            st.metric("No GT", status_counts['red'],
                      delta=f"{status_counts['red']/total_intervals*100:.1f}%")


if __name__ == "__main__":
    main()