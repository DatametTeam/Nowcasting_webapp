import io
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from functools import partial
from multiprocessing import Manager, Process
from pathlib import Path

import imageio
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image

from nwc_webapp.logging_config import setup_logger
from nwc_webapp.rendering.figures import compute_figure_gpd

# Set up logger
logger = setup_logger(__name__)


def create_diff_dict_in_parallel(diff_array, sidebar_args, save_on_disk=False):
    home_dir = Path.home()
    diff_img_out_dir = Path(f"{home_dir}/demo_sole/data/output/imgs/{sidebar_args['model_name']}/diff")
    start_date = sidebar_args["start_date"]
    start_time = sidebar_args["start_time"]
    combined_start = datetime.combine(start_date, start_time)
    logger.debug(f"Starting from: {combined_start}")
    time_step = timedelta(minutes=5)

    max_workers = os.cpu_count()

    progress_container = st.container()
    with progress_container:
        diff_progress = st.progress(0)
        diff_status = st.text("Processing DIFF")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        diff_results = []

        for i, result in enumerate(
            executor.map(
                partial(
                    save_figure,
                    base_date=combined_start,
                    time_step=time_step,
                    out_dir=diff_img_out_dir,
                    save_on_disk=save_on_disk,
                    name="diff",
                ),
                diff_array[:, 0],
                range(diff_array.shape[0]),
            )
        ):
            diff_results.append(result)
            diff_progress.progress((i + 1) / len(diff_array[:, 0]))
            diff_status.text(f"Processed {i + 1}/{len(diff_array[:, 0])} Diff images")

    diff_figures = {key: value for result in diff_results for key, value in result.items()}
    return diff_figures


def create_fig_dict_in_parallel(gt_data, pred_data, sidebar_args, save_on_disk=False):
    home_dir = home_dir = Path.home()
    gt_img_out_dir = Path(f"{home_dir}/demo_sole/data/output/imgs/{sidebar_args['model_name']}/gt")
    pred_img_out_dir = Path(f"{home_dir}/demo_sole/data/output/imgs/{sidebar_args['model_name']}/pred")

    start_date = sidebar_args["start_date"]
    start_time = sidebar_args["start_time"]
    combined_start = datetime.combine(start_date, start_time)
    logger.debug(f"Starting from: {combined_start}")
    time_step = timedelta(minutes=5)

    max_workers = os.cpu_count()

    progress1_container = st.container()
    with progress1_container:
        gt_progress = st.progress(0)
        gt_status = st.text("Processing GT")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        gt_results = []
        for i, result in enumerate(
            executor.map(
                partial(
                    save_figure,
                    base_date=combined_start,
                    time_step=time_step,
                    out_dir=gt_img_out_dir,
                    save_on_disk=save_on_disk,
                    name="",
                ),
                gt_data[:, 0],
                range(gt_data.shape[0]),
            )
        ):
            gt_results.append(result)
            gt_progress.progress((i + 1) / len(gt_data[:, 0]))
            gt_status.text(f"Processed {i + 1}/{len(gt_data[:, 0])} GT images")
    # st.write(f"Time for GT data: {time.time() - timea} seconds")

    progress_container = st.container()
    with progress_container:
        pred_progress = st.progress(0)
        pred_status = st.text("Processing Predictions")

    # salvataggio figure PRD
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        pred_results = []
        total_sequences = pred_data.shape[0]

        for i, result in enumerate(
            executor.map(
                partial(
                    save_prediction_sequence,
                    base_date=combined_start,
                    time_step=time_step,
                    out_dir=pred_img_out_dir,
                    save_on_disk=save_on_disk,
                ),
                pred_data,
                range(total_sequences),
            )
        ):
            pred_results.append(result)
            pred_progress.progress((i + 1) / total_sequences)
            pred_status.text(f"Processed {i + 1}/{total_sequences} prediction sequences")

    # Aggregate results into dictionaries
    gt_figures = {key: fig for result in gt_results for key, fig in result.items()}
    pred_figures = {key: value for result in pred_results for key, value in result.items()}

    return gt_figures, pred_figures


def save_figure(data_slice, index, base_date, time_step, out_dir, save_on_disk, name=""):
    actual_date = base_date + index * time_step
    fig = compute_figure_gpd(data_slice, actual_date.strftime("%d/%m/%Y %H:%M"), name=name)

    if save_on_disk:
        file_name = f"{actual_date.strftime('%d%m%Y_%H%M')}.png"
        file_path = os.path.join(out_dir, file_name)
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return {actual_date.strftime("%d%m%Y_%H%M"): fig}
    else:
        plt.close(fig)
        return {actual_date.strftime("%d%m%Y_%H%M"): fig}


def save_prediction_sequence(data_series, element_index, base_date, time_step, out_dir, save_on_disk):
    actual_date = base_date + element_index * time_step
    folder_name = actual_date.strftime("%d%m%Y_%H%M")
    results = {}

    for i, sequence in enumerate(data_series):  # Iterate over sequence_len
        timestamp_offset = (i + 1) * 5  # 5 minutes per time step
        if timestamp_offset not in [30, 60]:
            continue  # Skip if timestamp_offset is not 30 or 60

        fig = compute_figure_gpd(
            sequence, ("PRED @ " + (actual_date + timedelta(minutes=timestamp_offset)).strftime("%d/%m/%Y %H:%M"))
        )  # Compute figure

        if save_on_disk:
            folder_path = os.path.join(out_dir, "pred", folder_name)
            os.makedirs(folder_path, exist_ok=True)
            file_name = f"{folder_name}_+{timestamp_offset}min.png"
            file_path = os.path.join(folder_path, file_name)
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        # Add to the nested dictionary
        results[f"+{timestamp_offset}min"] = fig

    # print(f"Finished processing: result = {results.keys()}")
    return {folder_name: results}


def create_single_gif_for_parallel(
    queue,
    start_pos,
    figures_dict,
    window_size,
    sorted_keys,
    process_idx,
    sidebar_args,
    save_on_disk=True,
    fps_gif=3,
    name="gt",
):
    """
    Create a single GIF and send progress updates through a queue.
    """
    buf = io.BytesIO()
    frames = []
    logger.debug(f"Creating GIF from position {start_pos} to {start_pos + window_size}")
    window_keys = sorted_keys[start_pos : start_pos + window_size]
    logger.debug(f"Window keys: {len(window_keys)} frames")

    for i, key in enumerate(window_keys):
        fig = figures_dict[key]
        buf_tracked = io.BytesIO()
        fig.savefig(buf_tracked, format="png", bbox_inches="tight", pad_inches=0)
        buf_tracked.seek(0)
        img = Image.open(buf_tracked)
        frames.append(np.array(img))

        # Send progress update through queue
        progress = (i + 1) / len(window_keys)
        queue.put(("progress", process_idx, progress))

    imageio.mimsave(buf, frames, format="GIF", fps=fps_gif, loop=0)
    buf.seek(0)
    # Send completed GIF through queue
    queue.put(("complete", process_idx, buf.getvalue()))

    if save_on_disk:
        from nwc_webapp.config.config import get_config

        config = get_config()

        # Use new path structure
        if name == "gt":
            save_dir = config.gif_storage / "groundtruth"
        elif name == "diff":
            save_dir = config.gif_storage / "difference" / sidebar_args["model_name"]
        else:
            save_dir = config.gif_storage / "prediction" / sidebar_args["model_name"]

        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{window_keys[0]}_{window_keys[-1]}"

        # Add suffix based on start position to match naming convention
        if "gif_suffix" in sidebar_args:
            file_name += sidebar_args["gif_suffix"]

        save_path = save_dir / f"{file_name}.gif"
        imageio.mimsave(save_path, frames, format="GIF", fps=fps_gif, loop=0)
        logger.info(f"GIF saved @ {save_path}")


def create_sliding_window_gifs(
    figures_dict, sidebar_args, start_positions=[0, 6, 12], save_on_disk=True, fps_gif=3, name="gt"
):
    """
    Create multiple GIFs in parallel with progress tracking.

    GIF suffixes are added based on start_position:
    - Position 0: no suffix (full sequence)
    - Position 6: _+30m suffix
    - Position 12: _+60m suffix
    """
    window_size = len(figures_dict) - 12
    sorted_keys = sorted(figures_dict.keys())

    # Define suffixes for each start position
    suffix_map = {0: "", 6: "_+30m", 12: "_+60m"}

    # Create progress bars and text containers
    progress_bars = []
    progress_texts = []

    with st.container():
        for i in range(len(start_positions)):
            progress_bars.append(st.progress(0))
            progress_texts.append(st.empty())

    # Setup multiprocessing queue and processes
    queue = Manager().Queue()
    processes = []
    for idx, start_pos in enumerate(start_positions):
        # Create a copy of sidebar_args with the gif_suffix added
        args_with_suffix = sidebar_args.copy()
        args_with_suffix["gif_suffix"] = suffix_map.get(start_pos, "")

        p = Process(
            target=create_single_gif_for_parallel,
            args=(
                queue,
                start_pos,
                figures_dict,
                window_size,
                sorted_keys,
                idx,
                args_with_suffix,
                save_on_disk,
                fps_gif,
                name,
            ),
        )
        processes.append(p)

    # Start all processes
    for p in processes:
        p.start()

    # Initialize storage for completed GIFs
    completed_gifs = [None] * len(start_positions)
    completed_count = 0

    # Monitor queue for updates
    while completed_count < len(start_positions):
        try:
            msg_type, idx, data = queue.get(timeout=1)

            if msg_type == "progress":
                total_steps = window_size
                current_step = int(data * total_steps)
                progress_bars[idx].progress(data)
                progress_texts[idx].text(f"GIF {idx + 1}: {current_step}/{total_steps} complete")

            elif msg_type == "complete":
                completed_gifs[idx] = io.BytesIO(data)
                completed_count += 1
                progress_texts[idx].text(f"GIF {idx + 1}: Complete!")

        except:
            # Check if all processes are still alive
            if not any(p.is_alive() for p in processes):
                break

    # Clean up processes
    for p in processes:
        p.join()

    # Clear progress indicators
    for bar in progress_bars:
        bar.empty()
    for text in progress_texts:
        text.empty()

    return completed_gifs


def create_single_gif(
    queue, figures, gif_type, process_idx, start_key, end_key, sidebar_args, fps_gif=3, save_on_disk=True
):
    buf = io.BytesIO()
    frames = []

    for idx, fig in enumerate(figures):
        try:
            buf_tracked = io.BytesIO()
            fig.savefig(buf_tracked, format="png", bbox_inches="tight", pad_inches=0)
            buf_tracked.seek(0)
            img = Image.open(buf_tracked)
            frames.append(np.array(img))

            # Update progress
            progress = (idx + 1) / len(figures)
            queue.put(("progress", process_idx, progress))

        except Exception as e:
            logger.info(f"Error processing figure for {gif_type}, index {idx}: {e}")
            continue

    if not frames:
        raise ValueError(f"No frames generated for {gif_type}.")

    # Create the GIF
    imageio.mimsave(buf, frames, format="GIF", fps=fps_gif, loop=0)
    buf.seek(0)
    queue.put(("complete", process_idx, buf.getvalue()))

    if save_on_disk:
        from nwc_webapp.config.config import get_config

        config = get_config()

        # Use new path structure for predictions
        save_dir = config.gif_storage / "prediction" / sidebar_args["model_name"]
        save_dir.mkdir(parents=True, exist_ok=True)

        # Convert gif_type "+30 mins" -> "_+30m", "+60 mins" -> "_+60m"
        suffix = gif_type.replace(" mins", "m").replace(" ", "")
        file_name = f"{start_key}_{end_key}{suffix}"

        save_path = save_dir / f"{file_name}.gif"
        imageio.mimsave(save_path, frames, format="GIF", fps=fps_gif, loop=0)
        logger.info(f"GIF saved @ {save_path}")


def create_single_gif_from_dict(figures_dict, save_path, sidebar_args, fps_gif=3, show_progress=True):
    """
    Create a single GIF from all figures in a dictionary.

    Args:
        figures_dict: Dictionary where keys are timestamps and values are matplotlib figures
        save_path: Path where to save the GIF file
        sidebar_args: Sidebar arguments (unused, kept for compatibility)
        fps_gif: Frames per second for the GIF
        show_progress: Whether to show progress in Streamlit UI

    Returns:
        Path to the saved GIF file, or None if failed
    """
    if not figures_dict:
        logger.warning("Empty figures dictionary - cannot create GIF")
        return None

    try:
        # Sort keys to ensure correct temporal order
        sorted_keys = sorted(figures_dict.keys())
        frames = []

        # Create progress indicator if requested
        if show_progress:
            progress_bar = st.progress(0)
            progress_text = st.empty()

        # Convert each figure to a frame
        for idx, key in enumerate(sorted_keys):
            fig = figures_dict[key]
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            img = Image.open(buf)
            frames.append(np.array(img))
            buf.close()

            # Update progress
            if show_progress:
                progress = (idx + 1) / len(sorted_keys)
                progress_bar.progress(progress)
                progress_text.text(f"Processing frame {idx + 1}/{len(sorted_keys)}")

        # Ensure the parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the GIF
        imageio.mimsave(save_path, frames, format="GIF", fps=fps_gif, loop=0)
        logger.info(f"GIF saved @ {save_path}")

        # Clear progress indicators
        if show_progress:
            progress_bar.empty()
            progress_text.empty()

        return save_path

    except Exception as e:
        logger.error(f"Error creating GIF: {e}")
        return None


def create_sliding_window_gifs_for_predictions(prediction_dict, sidebar_args, save_on_disk=True, fps_gif=3):
    """
    Create GIFs for the predictions dictionary. Each GIF corresponds to:
    - +30 mins: Figures from the "+30mins" key in the nested dictionary.
    - +60 mins: Figures from the "+60mins" key in the nested dictionary.

    Args:
        prediction_dict: Dictionary where each key is a timestamp, and the value is a nested dictionary
                         with "+30mins" and "+60mins" keys containing figures.
        sidebar_args: sidebar_args.

    Returns:
        Tuple of BytesIO buffers containing the two GIFs (+30 mins, +60 mins).
    """

    sorted_keys = sorted(prediction_dict.keys())

    # Select all except the last 12 keys
    keys_except_last_12 = sorted_keys[:-12]  # TODO: perchè?

    # Filter the dictionary
    prediction_dict = {key: prediction_dict[key] for key in keys_except_last_12}

    # Extract figures for +30 mins and +60 mins
    figures_30 = [nested_dict["+30min"] for nested_dict in prediction_dict.values() if "+30min" in nested_dict]
    figures_60 = [nested_dict["+60min"] for nested_dict in prediction_dict.values() if "+60min" in nested_dict]
    start_key = keys_except_last_12[0]
    end_key = keys_except_last_12[-1]

    if not figures_30 or not figures_60:
        raise ValueError("Insufficient figures for +30 min or +60 min.")

    # Create progress bars and placeholders
    progress_bars = []
    progress_texts = []

    with st.container():
        for _ in range(2):
            progress_bars.append(st.progress(0))
            progress_texts.append(st.empty())

    # Setup multiprocessing queue and processes
    queue = Manager().Queue()
    processes = []

    for idx, (figures, gif_type) in enumerate([(figures_30, "+30 mins"), (figures_60, "+60 mins")]):
        p = Process(
            target=create_single_gif,
            args=(queue, figures, gif_type, idx, start_key, end_key, sidebar_args, fps_gif, save_on_disk),
        )
        processes.append(p)

    # Start all processes
    for p in processes:
        p.start()

    # Initialize storage for completed GIFs
    completed_gifs = [None] * len(processes)
    completed_count = 0

    # Monitor queue for updates
    while completed_count < len(processes):
        try:
            msg_type, idx, data = queue.get(timeout=1)

            if msg_type == "progress":
                progress_bars[idx].progress(data)
                total_steps = len(keys_except_last_12)
                current_step = int(data * total_steps)
                progress_bars[idx].progress(data)
                progress_texts[idx].text(f"GIF {idx + 1}: {current_step}/{total_steps} complete")

            elif msg_type == "complete":
                completed_gifs[idx] = io.BytesIO(data)
                completed_count += 1
                progress_texts[idx].text(f"GIF {idx + 1}: Complete!")

        except:
            # Check if all processes are still alive
            if not any(p.is_alive() for p in processes):
                break

    # Clean up processes
    for p in processes:
        p.join()

    # Clear progress indicators
    for bar in progress_bars:
        bar.empty()
    for text in progress_texts:
        text.empty()

    return completed_gifs
