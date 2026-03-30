"""
Data API endpoints - predictions, groundtruth, SRI files.

These endpoints replace the direct function calls scattered through
your Streamlit pages. Instead of:

    missing, existing = check_missing_predictions(model, start_dt, end_dt)
    st.write(f"Missing: {len(missing)}")

The frontend calls:

    GET /api/data/predictions/check?model=ConvLSTM&start=2025-01-01T12:00&end=2025-01-01T13:00

And gets back JSON it can display however it wants.
"""
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

from nwc_webapp.config.config import get_config
from nwc_webapp.data.checking import (
    check_missing_predictions,
    check_single_prediction_exists,
    check_target_data_for_range,
    check_target_data_exists,
)

router = APIRouter(prefix="/api/data", tags=["data"])


# ============================================================================
# SRI (radar input) endpoints
# ============================================================================

@router.get("/sri/latest")
async def get_latest_sri():
    """
    Get the latest SRI radar file.

    In Streamlit: this was done by a background thread (get_latest_file in workers.py)
    that polled the SRI folder and stored the result in st.session_state.

    Here: the frontend just asks "what's the latest file?" whenever it needs to know.
    Later (Phase 1.6), we'll also push updates via WebSocket.
    """
    config = get_config()
    sri_folder = str(config.sri_folder)

    if not os.path.exists(sri_folder):
        return {"latest_file": None, "error": f"SRI folder not found: {sri_folder}"}

    files = [f for f in os.listdir(sri_folder) if f.endswith(".hdf")]
    if not files:
        return {"latest_file": None, "file_count": 0}

    # Sort by datetime in filename (DD-MM-YYYY-HH-MM.hdf)
    files.sort(
        key=lambda x: datetime.strptime(x.split(".")[0], "%d-%m-%Y-%H-%M"),
        reverse=True,
    )

    return {
        "latest_file": files[0],
        "file_count": len(files),
    }


# ============================================================================
# Prediction endpoints
# ============================================================================

@router.get("/predictions/check")
async def check_predictions(
    model: str = Query(..., description="Model name (e.g. ConvLSTM)"),
    start: str = Query(..., description="Start datetime (YYYY-MM-DDTHH:MM)"),
    end: str = Query(..., description="End datetime (YYYY-MM-DDTHH:MM)"),
):
    """
    Check which predictions exist for a model in a date range.

    In Streamlit (nowcasting.py line ~302):
        missing_timestamps, existing_timestamps = check_missing_predictions(model_name, start_dt, end_dt)
        st.success(f"✅ All {total_count} predictions exist")

    Here: returns JSON with missing/existing counts and timestamps.

    Example:
        GET /api/data/predictions/check?model=ConvLSTM&start=2025-01-01T12:00&end=2025-01-01T13:00
    """
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {e}")

    if end_dt < start_dt:
        raise HTTPException(status_code=400, detail="End datetime must be after start datetime")

    missing, existing = check_missing_predictions(model, start_dt, end_dt)

    return {
        "model": model,
        "start": start,
        "end": end,
        "total": len(missing) + len(existing),
        "existing_count": len(existing),
        "missing_count": len(missing),
        "all_exist": len(missing) == 0,
        "existing_timestamps": [dt.isoformat() for dt in existing],
        "missing_timestamps": [dt.isoformat() for dt in missing],
    }


@router.get("/predictions/check-single")
async def check_single_prediction(
    model: str = Query(..., description="Model name"),
    timestamp: str = Query(..., description="Datetime (YYYY-MM-DDTHH:MM)"),
):
    """
    Check if a single prediction exists.

    In Streamlit (prediction_by_date.py):
        pred_exists = check_single_prediction_exists(selected_model, selected_datetime)
    """
    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {e}")

    exists = check_single_prediction_exists(model, dt)

    return {
        "model": model,
        "timestamp": timestamp,
        "exists": exists,
    }


# ============================================================================
# Target (groundtruth) data endpoints
# ============================================================================

@router.get("/target/check-range")
async def check_target_data_range(
    start: str = Query(..., description="Start datetime"),
    end: str = Query(..., description="End datetime"),
):
    """
    Check if target (groundtruth) data exists for a date range.

    In Streamlit (nowcasting.py):
        all_target_exist, missing, existing = check_target_data_for_range(start_dt, end_dt)
    """
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {e}")

    all_exist, missing, existing = check_target_data_for_range(start_dt, end_dt)

    return {
        "all_exist": all_exist,
        "missing_count": len(missing),
        "existing_count": len(existing),
        "missing_timestamps": [dt.isoformat() for dt in missing],
    }


@router.get("/target/check-single")
async def check_single_target(
    timestamp: str = Query(..., description="Datetime (YYYY-MM-DDTHH:MM)"),
):
    """
    Check if target data exists for a single timestamp.

    In Streamlit (prediction_by_date.py):
        targets_exist, found_count, total_count = check_target_data_exists(selected_datetime)
    """
    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {e}")

    exists, found_count, total_count = check_target_data_exists(dt)

    return {
        "timestamp": timestamp,
        "exists": exists,
        "found_count": found_count,
        "total_count": total_count,
    }


# ============================================================================
# Calendar availability endpoints
# ============================================================================

@router.get("/predictions/calendar")
async def predictions_calendar(
    models: str = Query(..., description="Comma-separated model names"),
    year: int = Query(..., description="Year"),
    month: int = Query(..., description="Month (1-12)"),
):
    """
    Get which dates in a given month have at least one prediction file.
    Used by the frontend to highlight available dates in the calendar.

    Scans the prediction directories for all requested models, collects
    unique dates that have .npy files matching the requested year/month.
    """
    config = get_config()
    model_list = [m.strip() for m in models.split(",") if m.strip()]

    dates_with_predictions = set()

    for model_name in model_list:
        pred_dir = config.real_time_pred / model_name
        if not pred_dir.exists():
            continue

        for f in os.listdir(str(pred_dir)):
            if not f.endswith(".npy"):
                continue
            try:
                dt = datetime.strptime(f.replace(".npy", ""), "%d-%m-%Y-%H-%M")
                if dt.year == year and dt.month == month:
                    dates_with_predictions.add(dt.strftime("%Y-%m-%d"))
            except ValueError:
                continue

    return {"dates": sorted(dates_with_predictions)}


@router.get("/predictions/day-detail")
async def predictions_day_detail(
    models: str = Query(..., description="Comma-separated model names"),
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
):
    """
    Get per-timestamp model availability for a specific date.
    Returns which models have prediction files at each 5-minute slot.

    Used by the frontend to show a time availability panel so the user
    knows exactly which timestamps have data and for which models.
    """
    config = get_config()
    model_list = [m.strip() for m in models.split(",") if m.strip()]

    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    # Collect available timestamps per model for this date
    slots = {}  # time_str -> [model_names]

    for model_name in model_list:
        pred_dir = config.real_time_pred / model_name
        if not pred_dir.exists():
            continue

        for f in os.listdir(str(pred_dir)):
            if not f.endswith(".npy"):
                continue
            try:
                dt = datetime.strptime(f.replace(".npy", ""), "%d-%m-%Y-%H-%M")
                if dt.date() == target_date:
                    time_str = dt.strftime("%H:%M")
                    if time_str not in slots:
                        slots[time_str] = []
                    slots[time_str].append(model_name)
            except ValueError:
                continue

    return {
        "date": date,
        "models": model_list,
        "slots": dict(sorted(slots.items())),
        "total_models": len(model_list),
    }


# ============================================================================
# Data Explorer endpoints
# ============================================================================

@router.get("/explorer/timestamps")
async def explorer_timestamps(
    start: str = Query(..., description="Start datetime (YYYY-MM-DDTHH:MM)"),
    end: str = Query(..., description="End datetime (YYYY-MM-DDTHH:MM)"),
    product: str = Query("SRI_adj", description="Radar product (SRI_adj, VMI, ETM, VIL)"),
):
    """
    List available HDF5 timestamps for a radar product in a date range.

    Generates expected 5-minute timestamps and checks which files exist
    in the product folder. Returns found + missing lists.

    Maximum range: controlled by explorer_max_hours in cfg.yaml.
    """
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {e}")

    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="End datetime must be after start datetime")

    config = get_config()
    max_hours = config.explorer_max_hours
    if (end_dt - start_dt).total_seconds() > max_hours * 3600:
        raise HTTPException(status_code=400, detail=f"Date range cannot exceed {max_hours} hours")

    config = get_config()
    products = config.radar_products
    if product not in products:
        raise HTTPException(status_code=400, detail=f"Unknown product: {product}. Available: {list(products.keys())}")

    product_folder = config.get_product_folder(product)
    if not product_folder:
        # No folder configured for this environment — return empty result (not an error)
        return {
            "timestamps": [],
            "missing": [],
            "total_expected": 0,
            "total_found": 0,
            "product": product,
        }

    # Determine file extension based on product format
    product_cfg = products[product]
    file_format = product_cfg.get("file_format", "hdf")
    file_ext = ".tif" if file_format == "tiff" else ".hdf"

    # Generate all expected 5-minute timestamps
    expected = []
    current = start_dt
    while current <= end_dt:
        expected.append(current)
        current += timedelta(minutes=5)

    found = []
    missing = []
    for dt in expected:
        stem = dt.strftime("%d-%m-%Y-%H-%M")
        primary_filename = stem + file_ext
        alt_filename = (stem + ".tiff") if file_format == "tiff" else None

        # Use find_product_file so both the recent flat folder and the archive
        # YYYY/MM/DD/product/ structure are checked transparently.
        resolved = config.find_product_file(product, dt, primary_filename)
        if resolved is None and alt_filename:
            resolved = config.find_product_file(product, dt, alt_filename)

        if resolved is not None:
            found.append(dt.isoformat())
        else:
            missing.append(dt.isoformat())

    return {
        "timestamps": found,
        "missing": missing,
        "total_expected": len(expected),
        "total_found": len(found),
        "product": product,
    }


# ============================================================================
# Mock data generation (local development only)
# ============================================================================

@router.post("/mock/generate-next")
async def generate_next_mock_data():
    """
    Generate the next SRI file + prediction files for all models.

    Finds the latest SRI file, creates a new one at +5 minutes,
    and generates corresponding prediction files for every model.
    Used by the real-time simulation loop in local development.
    """
    from nwc_webapp.mock.generator import (
        create_mock_hdf_file,
        generate_temporal_sequence,
    )

    import numpy as np

    config = get_config()
    sri_folder = Path(str(config.sri_folder))
    sri_folder.mkdir(parents=True, exist_ok=True)

    # Find the latest SRI file to determine the next timestamp
    hdf_files = [f for f in os.listdir(sri_folder) if f.endswith(".hdf")]

    if hdf_files:
        hdf_files.sort(
            key=lambda x: datetime.strptime(x.split(".")[0], "%d-%m-%Y-%H-%M"),
            reverse=True,
        )
        latest_name = hdf_files[0].replace(".hdf", "")
        latest_dt = datetime.strptime(latest_name, "%d-%m-%Y-%H-%M")
        next_dt = latest_dt + timedelta(minutes=5)
    else:
        # No files yet — start from now (rounded to 5 min)
        now = datetime.now()
        next_dt = now.replace(
            minute=(now.minute // 5) * 5, second=0, microsecond=0
        )

    # 1) Create the new SRI file
    sri_filename = next_dt.strftime("%d-%m-%Y-%H-%M") + ".hdf"
    create_mock_hdf_file(sri_folder / sri_filename, next_dt)
    logger.info(f"Mock SRI created: {sri_filename}")

    # 2) Create prediction files for every model
    models = config.models
    for model_name in models:
        pred_folder = config.real_time_pred / model_name
        pred_folder.mkdir(parents=True, exist_ok=True)

        pred_filename = next_dt.strftime("%d-%m-%Y-%H-%M") + ".npy"
        pred_path = pred_folder / pred_filename

        if not pred_path.exists():
            prediction = generate_temporal_sequence(
                num_timesteps=12,
                shape=(1400, 1200),
                base_seed=int(next_dt.timestamp()),
            )
            if model_name == "ED_ConvLSTM":
                prediction = np.expand_dims(prediction, axis=0)
            np.save(pred_path, prediction)
            logger.info(f"Mock prediction created: {model_name}/{pred_filename}")

    return {
        "timestamp": next_dt.isoformat(),
        "filename": sri_filename,
    }