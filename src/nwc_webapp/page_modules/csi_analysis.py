"""
CSI Analysis page for model evaluation.
"""
import time
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from nwc_webapp.config.config import get_config
from nwc_webapp.config.environment import is_hpc
from nwc_webapp.logging_config import setup_logger
from nwc_webapp.page_modules.csi_utils import compute_csi_for_models
from nwc_webapp.page_modules.nowcasting_utils import (
    check_missing_predictions,
    submit_date_range_prediction_job,
)

# Set up logger
logger = setup_logger(__name__)


def round_to_5_minutes(dt: datetime) -> datetime:
    """Round datetime to nearest 5 minutes."""
    minute = (dt.minute // 5) * 5
    return dt.replace(minute=minute, second=0, microsecond=0)


def get_model_prediction_status(model_name: str, start_dt: datetime, end_dt: datetime) -> dict:
    """
    Check if model has all predictions in the specified interval.

    Returns:
        dict with keys: 'has_all', 'missing_count', 'existing_count', 'status'
    """
    missing, existing = check_missing_predictions(model_name, start_dt, end_dt, verbose=False)

    has_all = len(missing) == 0
    total = len(missing) + len(existing)

    return {
        'has_all': has_all,
        'missing_count': len(missing),
        'existing_count': len(existing),
        'total_count': total,
        'status': '✅' if has_all else '❌'
    }


def show_csi_analysis_page(model_list):
    """
    Display CSI Analysis page.

    Args:
        model_list: List of available models
    """
    st.title("CSI Analysis")

    # Initialize session state for job monitoring
    if "csi_computing_models" not in st.session_state:
        st.session_state["csi_computing_models"] = {}  # {model: job_id}
    if "csi_submitted_models" not in st.session_state:
        st.session_state["csi_submitted_models"] = set()
    if "csi_failed_models" not in st.session_state:
        st.session_state["csi_failed_models"] = set()
    if "csi_results" not in st.session_state:
        st.session_state["csi_results"] = None
    if "csi_cached_params" not in st.session_state:
        st.session_state["csi_cached_params"] = None

    # Date and time selection with persistence
    now = round_to_5_minutes(datetime.now())

    # Initialize session state for date/time persistence (only on first load)
    if "csi_start_date" not in st.session_state:
        st.session_state["csi_start_date"] = now.date()
    if "csi_start_time" not in st.session_state:
        st.session_state["csi_start_time"] = now.time()
    if "csi_end_date" not in st.session_state:
        st.session_state["csi_end_date"] = now.date()
    if "csi_end_time" not in st.session_state:
        st.session_state["csi_end_time"] = now.time()

    col1, col2 = st.columns(2)
    with col1:
        st.date_input(
            "Start Date",
            max_value=now.date(),
            format="DD/MM/YYYY",
            key="csi_start_date"
        )
        st.time_input(
            "Start Time",
            step=300,  # 5 minutes in seconds
            key="csi_start_time"
        )

    with col2:
        st.date_input(
            "End Date",
            max_value=now.date(),
            format="DD/MM/YYYY",
            key="csi_end_date"
        )
        st.time_input(
            "End Time",
            step=300,  # 5 minutes in seconds
            key="csi_end_time"
        )

    # Read the current values from session state
    start_date = st.session_state["csi_start_date"]
    start_time = st.session_state["csi_start_time"]
    end_date = st.session_state["csi_end_date"]
    end_time = st.session_state["csi_end_time"]

    # Combine date and time
    start_datetime = datetime.combine(start_date, start_time)
    end_datetime = datetime.combine(end_date, end_time)

    # Validate date range
    if start_datetime > end_datetime:
        st.error("❌ Start date/time must be before or equal to end date/time!")
        return

    if end_datetime > now:
        st.error("❌ End date/time cannot be in the future!")
        return

    # Check if date range changed (clear results if so)
    current_params = (start_datetime, end_datetime)
    if st.session_state["csi_cached_params"] != current_params:
        st.session_state["csi_cached_params"] = current_params
        st.session_state["csi_results"] = None
        logger.info(f"Date range changed - clearing cached CSI results")

    st.markdown("---")

    # Filter out "Test" model
    available_models = [m for m in model_list if m.upper() != "TEST"]

    # Check prediction status for all models
    st.subheader("📊 Model Status for Selected Interval")

    with st.spinner("Checking prediction status..."):
        model_status = {}
        for model in available_models:
            status = get_model_prediction_status(model, start_datetime, end_datetime)
            model_status[model] = status

    # Display model list with checkboxes and status
    st.markdown("**Select models:**")

    # Select All / Deselect All buttons
    col_all, col_none = st.columns([1, 1])
    with col_all:
        if st.button("✓ Select All (with predictions)", key="csi_select_all", width='stretch'):
            # Select only models with all predictions
            for model in available_models:
                if model_status[model]['has_all']:
                    st.session_state[f"csi_model_{model}"] = True
                else:
                    st.session_state[f"csi_model_{model}"] = False
            st.rerun()

    with col_none:
        if st.button("✗ Deselect All", key="csi_deselect_all", width='stretch'):
            for model in available_models:
                if f"csi_model_{model}" in st.session_state:
                    st.session_state[f"csi_model_{model}"] = False
            st.rerun()

    st.markdown("")

    # Model list with status and checkboxes
    selected_models = []
    models_with_predictions = []
    models_without_predictions = []

    for model in available_models:
        status = model_status[model]['status']
        info = model_status[model]

        # Get job status if computing
        job_status_text = ""
        if model in st.session_state["csi_computing_models"]:
            # Check PBS job status
            try:
                from nwc_webapp.services.pbs import get_model_job_status, is_pbs_available

                if is_pbs_available():
                    job_status = get_model_job_status(model)
                    if job_status == "Q":
                        job_status_text = " 📋 Queue"
                    elif job_status == "R":
                        job_status_text = " ⚙️ Computing"
                    elif job_status is None:
                        # Job finished - check if successful
                        del st.session_state["csi_computing_models"][model]
                        # Recheck status
                        info = get_model_prediction_status(model, start_datetime, end_datetime)
                        model_status[model] = info
                        status = info['status']
            except Exception as e:
                logger.error(f"Error checking job status for {model}: {e}")

        elif model in st.session_state["csi_failed_models"]:
            job_status_text = " ❌ Failed"

        # Build status line
        status_line = f"{status} **{model}**"
        if info['has_all']:
            status_line += f" - {info['existing_count']} predictions"
        else:
            status_line += f" - Missing {info['missing_count']}/{info['total_count']}"

        status_line += job_status_text

        # Checkbox
        checked = st.checkbox(
            status_line,
            key=f"csi_model_{model}",
            label_visibility="visible"
        )

        if checked:
            selected_models.append(model)
            if info['has_all']:
                models_with_predictions.append(model)
            else:
                models_without_predictions.append(model)

    st.markdown("---")

    # Action buttons
    col_pred, col_csi = st.columns(2)

    with col_pred:
        can_compute_predictions = len(selected_models) > 0
        if st.button(
            "🔄 Compute/Recompute Predictions",
            disabled=not can_compute_predictions,
            width='stretch',
            type="secondary"
        ):
            # Submit jobs for ALL selected models
            # For models with predictions: delete first, then recompute
            # For models without predictions: just compute

            models_to_compute = selected_models.copy()

            # Delete existing predictions for models that have them
            if len(models_with_predictions) > 0:
                st.info(f"🗑️ Deleting existing predictions for {len(models_with_predictions)} model(s)...")
                from nwc_webapp.page_modules.nowcasting_utils import delete_predictions_in_range

                for model in models_with_predictions:
                    deleted = delete_predictions_in_range(model, start_datetime, end_datetime)
                    logger.info(f"Deleted {deleted} prediction(s) for {model}")

            st.info(f"📝 Submitting jobs for {len(models_to_compute)} model(s)...")

            # Store selected models for CSI computation after jobs finish
            all_selected_models = selected_models.copy()

            submission_results = {}
            for model in models_to_compute:
                logger.info(f"Submitting job for {model} ({start_datetime} to {end_datetime})")
                job_id = submit_date_range_prediction_job(model, start_datetime, end_datetime)

                if job_id:
                    st.session_state["csi_computing_models"][model] = job_id
                    st.session_state["csi_submitted_models"].add(model)
                    submission_results[model] = job_id
                    logger.info(f"✅ Submitted {model}: {job_id}")
                else:
                    st.session_state["csi_failed_models"].add(model)
                    logger.error(f"❌ Failed to submit {model}")

            if submission_results:
                st.success(f"✅ Submitted {len(submission_results)} job(s) successfully!")

                # Check if this is HPC or local mode
                from nwc_webapp.services.pbs import is_pbs_available

                if not is_pbs_available():
                    # Local mode: predictions already created, skip monitoring
                    logger.info("🖥️  Local mode detected - predictions created instantly")
                    time.sleep(2)  # Brief pause for filesystem sync
                    st.rerun()
                else:
                    # HPC mode: Monitor jobs until completion
                    st.markdown("---")
                    st.subheader("📊 Job Progress")

                    # Add CSS for animated dots
                    st.markdown(
                        """
                        <style>
                        .queue-text::after, .running-text::after {
                            content: '';
                            animation: dots 1.5s steps(4, end) infinite;
                        }
                        @keyframes dots {
                            0%, 24% { content: ''; }
                            25%, 49% { content: '.'; }
                            50%, 74% { content: '..'; }
                            75%, 100% { content: '...'; }
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Create placeholders for each model
                    model_placeholders = {}
                    for model in submission_results.keys():
                        model_placeholders[model] = st.empty()

                    # Monitor all jobs
                    from nwc_webapp.services.pbs import get_model_job_status

                    max_iterations = 1800  # 1 hour max
                    iteration = 0
                    models_last_status = {model: None for model in submission_results.keys()}
                    models_none_count = {model: 0 for model in submission_results.keys()}  # Track consecutive None checks
                    completed_models = set()
                    failed_models = set()

                    while iteration < max_iterations and len(completed_models) + len(failed_models) < len(submission_results):
                        # Check ALL models simultaneously every 2 seconds
                        current_statuses = {}

                        for model in submission_results.keys():
                            if model in completed_models or model in failed_models:
                                continue

                            try:
                                current_status = get_model_job_status(model)
                                current_statuses[model] = current_status

                            except Exception as e:
                                logger.error(f"Error checking job status for {model}: {e}")
                                current_statuses[model] = None

                        # Update displays for all models
                        for model, current_status in current_statuses.items():
                            if model in completed_models or model in failed_models:
                                continue

                            # Update display if status changed
                            if current_status != models_last_status[model]:
                                if current_status == "Q":
                                    model_placeholders[model].markdown(
                                        f"**{model}**: ⏳ <span class='queue-text'>In queue</span>",
                                        unsafe_allow_html=True
                                    )
                                    models_none_count[model] = 0
                                elif current_status == "R":
                                    model_placeholders[model].markdown(
                                        f"**{model}**: ⚙️ <span class='running-text'>Running</span>",
                                        unsafe_allow_html=True
                                    )
                                    models_none_count[model] = 0

                            # Check if job disappeared (None status after being in queue/running)
                            if current_status is None and models_last_status[model] is not None:
                                models_none_count[model] += 1
                                logger.debug(f"[{model}] Status None (consecutive: {models_none_count[model]}/3)")

                                if models_none_count[model] >= 3:
                                    # Job completed or failed - check predictions
                                    logger.info(f"[{model}] Job disappeared from queue - verifying results")
                                    model_placeholders[model].info(f"**{model}**: ⏳ Verifying results...")
                                    time.sleep(3)  # Wait for filesystem sync

                                    status_check = get_model_prediction_status(model, start_datetime, end_datetime)

                                    if status_check['has_all']:
                                        model_placeholders[model].success(f"**{model}**: ✅ Prediction ready!")
                                        completed_models.add(model)
                                        if model in st.session_state["csi_computing_models"]:
                                            del st.session_state["csi_computing_models"][model]
                                    else:
                                        model_placeholders[model].error(f"**{model}**: ❌ FAILED - predictions not found")
                                        failed_models.add(model)
                                        st.session_state["csi_failed_models"].add(model)
                                        if model in st.session_state["csi_computing_models"]:
                                            del st.session_state["csi_computing_models"][model]
                            elif current_status is None:
                                # Still None - increment counter
                                models_none_count[model] += 1
                            else:
                                # Reset counter if status is not None
                                models_none_count[model] = 0

                            # Update last status
                            if current_status is not None:
                                models_last_status[model] = current_status

                        time.sleep(2)
                        iteration += 1

                    # Summary
                    if len(completed_models) > 0:
                        st.success(f"✅ {len(completed_models)} model(s) completed successfully!")

                        # Auto-compute CSI for ALL selected models (completed + already existing)
                        # Check which models now have all predictions
                        models_for_csi = []
                        for model in all_selected_models:
                            if model not in failed_models:
                                # Verify predictions exist
                                status_check = get_model_prediction_status(model, start_datetime, end_datetime)
                                if status_check['has_all']:
                                    models_for_csi.append(model)

                        if len(models_for_csi) > 0:
                            st.info(f"🔄 Auto-computing CSI/POD/FAR for {len(models_for_csi)} model(s)...")

                            with st.spinner(f"Computing CSI/POD/FAR for {len(models_for_csi)} model(s)..."):
                                try:
                                    # Compute CSI, POD, FAR for all models with predictions
                                    csi_results, pod_results, far_results = compute_csi_for_models(
                                        models=models_for_csi,
                                        start_dt=start_datetime,
                                        end_dt=end_datetime
                                    )

                                    if csi_results is not None and pod_results is not None and far_results is not None:
                                        # Store results in session state
                                        st.session_state["csi_results"] = csi_results
                                        st.session_state["pod_results"] = pod_results
                                        st.session_state["far_results"] = far_results
                                        st.session_state["csi_result_models"] = models_for_csi
                                        st.session_state["csi_result_interval"] = (start_datetime, end_datetime)
                                        logger.info(f"✅ Auto-computed CSI/POD/FAR for {len(models_for_csi)} model(s)")
                                    else:
                                        logger.error("Failed to auto-compute CSI/POD/FAR")

                                except Exception as e:
                                    logger.error(f"Error auto-computing CSI/POD/FAR: {e}")
                                    import traceback
                                    logger.error(traceback.format_exc())

                    if len(failed_models) > 0:
                        st.error(f"❌ {len(failed_models)} model(s) failed")

                    # Wait a moment before rerunning
                    time.sleep(2)
                    st.rerun()
            else:
                st.error("❌ Failed to submit any jobs")
                st.rerun()

    with col_csi:
        can_compute_csi = len(models_with_predictions) > 0
        if st.button(
            "📊 Compute CSI for Selected Models",
            disabled=not can_compute_csi,
            width='stretch',
            type="primary"
        ):
            # Compute CSI/POD/FAR
            st.info(f"Computing CSI/POD/FAR for: {', '.join(models_with_predictions)}")

            with st.spinner(f"Computing CSI/POD/FAR for {len(models_with_predictions)} model(s)..."):
                try:
                    # Compute CSI, POD, FAR
                    csi_results, pod_results, far_results = compute_csi_for_models(
                        models=models_with_predictions,
                        start_dt=start_datetime,
                        end_dt=end_datetime
                    )

                    if csi_results is not None and pod_results is not None and far_results is not None:
                        # Store results in session state
                        st.session_state["csi_results"] = csi_results
                        st.session_state["pod_results"] = pod_results
                        st.session_state["far_results"] = far_results
                        st.session_state["csi_result_models"] = models_with_predictions
                        st.session_state["csi_result_interval"] = (start_datetime, end_datetime)

                        st.success(f"✅ CSI/POD/FAR computation completed for {len(models_with_predictions)} model(s)!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to compute CSI/POD/FAR. Check logs for details.")

                except Exception as e:
                    st.error(f"❌ Error computing CSI/POD/FAR: {e}")
                    logger.error(f"Error computing CSI/POD/FAR: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

    # Display CSI results if available
    if st.session_state["csi_results"] is not None:
        st.markdown("---")
        st.subheader("📈 CSI Results")

        results_dict = st.session_state["csi_results"]  # Dict[str, DataFrame]
        result_models = st.session_state.get("csi_result_models", [])
        result_interval = st.session_state.get("csi_result_interval", (start_datetime, end_datetime))

        # Display info about results
        start_str = result_interval[0].strftime("%d/%m/%Y %H:%M")
        end_str = result_interval[1].strftime("%d/%m/%Y %H:%M")
        st.info(f"**Models**: {', '.join(result_models)}  \n**Interval**: {start_str} to {end_str}")

        # Show tables per model with expandable sections
        st.markdown("**📊 CSI by Lead Time (per model):**")

        with st.expander("📋 Show detailed tables", expanded=False):
            for model in result_models:
                st.markdown(f"**{model}:**")
                model_df = results_dict[model]
                # Style: row-wise gradient (best lead time per threshold in green, worst in red)
                # axis=1 means compute gradient within each row (across columns/lead times)
                # No vmin/vmax - each table uses its own min/max for independent coloring
                styled = model_df.style.format("{:.3f}").background_gradient(cmap='RdYlGn', axis=1)
                st.dataframe(styled, width='stretch')
                st.markdown("")

        # Compute average CSI per threshold (averaged across lead times)
        st.markdown("**📈 Overall Model Performance by Threshold:**")

        # Build DataFrame: rows=models, columns=thresholds
        model_threshold_avg = {}
        thresholds = results_dict[result_models[0]].index.tolist()

        for model in result_models:
            model_df = results_dict[model]
            # For each threshold, compute mean across all lead times (columns)
            model_threshold_avg[model] = model_df.mean(axis=1)  # Mean across columns (lead times)

        # Create DataFrame: rows=models, columns=thresholds
        performance_df = pd.DataFrame(model_threshold_avg).T  # Transpose so models are rows
        performance_df.columns.name = "Threshold (mm/h)"
        performance_df.index.name = "Model"

        # Sort by mean CSI (best to worst)
        performance_df['Mean CSI'] = performance_df.mean(axis=1)
        performance_df = performance_df.sort_values('Mean CSI', ascending=False)

        # Style and display with gradient per threshold (column-wise: axis=0)
        # For each threshold (column), best model = green, worst = red
        # No vmin/vmax - each column uses its own min/max for independent coloring
        styled = performance_df.style.format("{:.3f}").background_gradient(cmap='RdYlGn', axis=0)
        st.dataframe(styled, width='stretch')

        # Compute overall average CSI per model (across all thresholds and lead times)
        model_overall_avg = {}
        for model in result_models:
            model_overall_avg[model] = results_dict[model].values.mean()

        avg_series = pd.Series(model_overall_avg).sort_values(ascending=False)

        # Display summary statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            best_model = avg_series.idxmax()
            best_score = avg_series.max()
            st.metric("Best Model", best_model)
            st.caption(f"Average CSI: {best_score:.3f}")

        with col2:
            worst_model = avg_series.idxmin()
            worst_score = avg_series.min()
            st.metric("Worst Model", worst_model)
            st.caption(f"Average CSI: {worst_score:.3f}")

        with col3:
            overall_avg = avg_series.mean()
            st.metric("Overall Average", f"{overall_avg:.3f}")
            st.caption("Across all models")

        # Visualization - CSI vs Lead Time plots
        st.markdown("---")
        st.subheader("📊 CSI vs Lead Time")

        try:
            import matplotlib.pyplot as plt

            # Get thresholds from first model's DataFrame
            thresholds = results_dict[result_models[0]].index.tolist()

            # Get POD and FAR results from session state
            pod_dict = st.session_state.get("pod_results", {})
            far_dict = st.session_state.get("far_results", {})

            # Display CSI plots and Fit Diagrams side by side for each threshold
            for threshold in thresholds:
                st.markdown(f"### 📊 Threshold: {threshold} mm/h")

                # Create two columns: CSI plot on left, Fit diagram on right
                col_csi, col_fit = st.columns(2)

                with col_csi:
                    st.markdown("**CSI vs Lead Time**")

                    # Create CSI plot for this threshold
                    fig_csi, ax_csi = plt.subplots(figsize=(6, 4))

                    # Plot each model as a line
                    for model in result_models:
                        model_df = results_dict[model]
                        lead_times = [int(col) for col in model_df.columns]  # Convert "5", "10", ... to integers
                        csi_values = model_df.loc[threshold].values

                        ax_csi.plot(lead_times, csi_values, marker='o', label=model, linewidth=2, markersize=4)

                    ax_csi.set_xlabel("Lead Time (minutes)", fontsize=9)
                    ax_csi.set_ylabel("CSI Score", fontsize=9)
                    ax_csi.set_ylim([0, 1])
                    ax_csi.set_xlim([0, 65])
                    ax_csi.grid(True, alpha=0.3)
                    ax_csi.legend(fontsize=7, loc='best')
                    ax_csi.set_xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])

                    plt.tight_layout()
                    st.pyplot(fig_csi, use_container_width=True)
                    plt.close(fig_csi)

                with col_fit:
                    st.markdown("**Performance Fit Diagram (POD vs FAR)**")

                    if pod_dict and far_dict:
                        from nwc_webapp.visualization.fit_diagram import create_performance_fit_diagram

                        # Extract averaged POD, FAR, CSI values for this threshold (across all lead times)
                        pod_values = []
                        far_values = []
                        csi_values = []

                        for model in result_models:
                            # Average across all lead times for this threshold
                            pod_avg = pod_dict[model].loc[threshold].mean()
                            far_avg = far_dict[model].loc[threshold].mean()
                            csi_avg = results_dict[model].loc[threshold].mean()

                            pod_values.append(pod_avg)
                            far_values.append(far_avg)
                            csi_values.append(csi_avg)

                        # Create fit diagram
                        try:
                            fig_fit = create_performance_fit_diagram(
                                pod_values=pod_values,
                                far_values=far_values,
                                csi_values=csi_values,
                                model_names=result_models,
                                threshold=threshold
                            )

                            st.pyplot(fig_fit, use_container_width=True)
                            plt.close(fig_fit)

                        except Exception as e:
                            logger.error(f"Error creating fit diagram for threshold {threshold}: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            st.warning(f"⚠️ Could not generate fit diagram")
                    else:
                        st.warning("⚠️ POD/FAR data not available")

                st.markdown("---")  # Separator between thresholds

            # Bar chart comparing overall model performance
            st.markdown("---")
            st.markdown("**Overall Model Comparison:**")
            fig_avg, ax_avg = plt.subplots(figsize=(8, 4))

            avg_sorted = avg_series.sort_values(ascending=False)
            bars = avg_sorted.plot(kind='bar', ax=ax_avg, color='skyblue', edgecolor='black')

            ax_avg.set_title("Average CSI Score by Model (All Thresholds & Lead Times)", fontsize=12, fontweight='bold')
            ax_avg.set_xlabel("Model", fontsize=10)
            ax_avg.set_ylabel("Average CSI Score", fontsize=10)
            ax_avg.set_ylim([0, 1])
            ax_avg.grid(True, alpha=0.3, axis='y')
            ax_avg.axhline(y=overall_avg, color='red', linestyle='--', linewidth=2, label=f'Overall Avg: {overall_avg:.3f}')
            ax_avg.legend()
            ax_avg.tick_params(axis='x', rotation=45)

            # Add value labels
            for container in ax_avg.containers:
                ax_avg.bar_label(container, fmt='%.3f')

            plt.tight_layout()
            st.pyplot(fig_avg, width='stretch')
            plt.close()

        except Exception as e:
            logger.error(f"Error creating plots: {e}")
            import traceback
            logger.error(traceback.format_exc())
            st.warning("⚠️ Could not generate plots")