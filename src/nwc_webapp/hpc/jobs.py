"""
Job submission utilities for date-range and real-time predictions.
Handles HPC (PBS), server (direct subprocess), and local (mock) modes.
"""

import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml

from nwc_webapp.config.environment import is_hpc, is_server
from nwc_webapp.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)


def modify_yaml_config_for_date_range(model_name: str, start_dt: datetime, end_dt: datetime) -> Path:
    """
    Modify YAML config with start/end dates for date-range predictions.

    Reads the config from config/model_configs/start_end/{model_name}.yaml,
    modifies the start_date and end_date fields in dataframe_strategy.args,
    and overwrites the file.

    Args:
        model_name: Model name (e.g., 'ConvLSTM', 'IAM4VP', 'PredFormer', 'SPROG')
        start_dt: Start datetime
        end_dt: End datetime

    Returns:
        Path to the modified YAML config file
    """
    # Source YAML path — resolved via config so v2/model_configs/ is used when running v2
    from nwc_webapp.config.config import get_config as _get_cfg
    config_path = _get_cfg().model_configs_path / "start_end" / f"{model_name}.yaml"

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Read the YAML file
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Format dates as "YYYY-MM-DD HH:MM" (matches inference script expectations)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M")
    end_str = end_dt.strftime("%Y-%m-%d %H:%M")

    # Modify start_date and end_date in dataframe_strategy.args
    if "dataframe_strategy" in config_data and "args" in config_data["dataframe_strategy"]:
        config_data["dataframe_strategy"]["args"]["start_date"] = start_str
        config_data["dataframe_strategy"]["args"]["end_date"] = end_str
        logger.info(f"Modified {model_name} config: start_date={start_str}, end_date={end_str}")
    else:
        logger.warning(f"Could not find dataframe_strategy.args in {model_name} config")

    # Overwrite the original file
    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Overwritten config at: {config_path}")
    return config_path


def submit_date_range_prediction_job(model_name: str, start_dt: datetime, end_dt: datetime) -> Optional[str]:
    """
    Submit PBS job for date-range predictions (HPC) or generate mock predictions (local).

    IMPORTANT: Most models need 1 hour of groundtruth data BEFORE the start_dt to make the
    first prediction at start_dt+5min. Therefore, we adjust the actual job start time
    to be 1 hour earlier than the requested start_dt.

    EXCEPTION: ED_ConvLSTM handles the lookback internally (goes back 12 timesteps from start_dt)
    so it does NOT need the -1 hour adjustment.

    Example: If user selects 12:00 start:
    - ConvLSTM/IAM4VP/etc: use 11:00 as actual start, first prediction at 12:05
    - ED_ConvLSTM: use 12:00 as actual start (goes back 12 timesteps internally)

    HPC mode:
    1. Modifies the YAML config with start/end dates (adjusted by -1 hour for most models)
    2. Modifies the PBS script to use absolute path for config or date parameters
    3. Submits the PBS job using the modified script
    4. Returns the job ID

    Local mode:
    1. Generates mock prediction files instantly
    2. Returns a fake job ID for UI compatibility

    Args:
        model_name: Model name (e.g., 'ConvLSTM', 'ED_ConvLSTM', 'IAM4VP', 'PredFormer', 'SPROG')
        start_dt: Start datetime (user-selected)
        end_dt: End datetime (user-selected)

    Returns:
        Job ID string if successful, None if failed
    """
    logger.info(f"User requested range: {start_dt} to {end_dt}")
    logger.info(f"First prediction will be at: {start_dt + timedelta(minutes=5)}")

    # Watch-folder mode: predictions are delivered externally (e.g. via rsync from HPC).
    # Skip submission entirely and let the frontend poll the predictions folder.
    from nwc_webapp.config.config import get_config
    if not get_config().submit_jobs:
        logger.info(f"submit_jobs=false — skipping job submission for {model_name}, watching output folder")
        return "watch_only"

    # Server mode: run inference directly as a subprocess (no job scheduler)
    if is_server():
        logger.info(f"🖥️  Running in SERVER mode - launching inference directly for {model_name}")
        try:
            config_path = modify_yaml_config_for_date_range(model_name, start_dt, end_dt)
        except Exception as e:
            logger.error(f"Failed to prepare config for {model_name}: {e}")
            return None

        server_cfg = get_config().server
        cmd = [
            "conda", "run", "-n", server_cfg.conda_env, "--no-capture-output",
            "python", server_cfg.inference_script_path,
            "--cfg_path", str(config_path),
        ]
        logger.info(f"Command: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            job_id = f"server_{proc.pid}"
            logger.info(f"✅ Inference process started! PID: {proc.pid}, job_id: {job_id}")
            return job_id
        except Exception as e:
            logger.error(f"Failed to launch inference process: {e}")
            return None

    # Local mode: generate mock predictions
    if not is_hpc():
        logger.info(f"🖥️  Running in LOCAL mode - generating mock predictions for {model_name}")

        try:
            from nwc_webapp.mock.generator import generate_mock_predictions_for_range

            created_count = generate_mock_predictions_for_range(model_name, start_dt, end_dt)

            if created_count >= 0:
                fake_job_id = f"mock_{int(datetime.now().timestamp())}"
                logger.info(f"✅ Mock predictions generated successfully! Fake job ID: {fake_job_id}")
                return fake_job_id
            else:
                logger.error("Failed to generate mock predictions")
                return None

        except Exception as e:
            logger.error(f"Error generating mock predictions: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    # HPC mode: Submit real PBS job
    logger.info(f"🖥️  Running in HPC mode - submitting PBS job for {model_name}")

    # ED_ConvLSTM uses a different interface than other models
    if model_name == "ED_ConvLSTM":
        # ED_ConvLSTM: Pass dates directly as environment variables (format: DD-MM-YYYY-HH-MM)
        start_str = start_dt.strftime("%d-%m-%Y-%H-%M")
        end_str = end_dt.strftime("%d-%m-%Y-%H-%M")

        # Step 1: Get the PBS script path
        pbs_script_path = (
            Path(__file__).parent
            / "scripts"
            / "start_end_pred_scripts"
            / f"run_{model_name}_inference_startend.sh"
        )

        if not pbs_script_path.exists():
            logger.error(f"PBS script not found: {pbs_script_path}")
            return None

        # Step 2: Modify PBS script to inject START_DATE and END_DATE
        try:
            with open(pbs_script_path, "r") as f:
                script_content = f.read()

            # Replace $START_DATE and $END_DATE with actual values
            modified_script = script_content.replace('"$START_DATE"', f'"{start_str}"')
            modified_script = modified_script.replace('"$END_DATE"', f'"{end_str}"')

            # Write modified script to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as tmp:
                tmp.write(modified_script)
                tmp_script_path = tmp.name

            logger.info(f"Created modified PBS script for ED_ConvLSTM: {tmp_script_path}")
            logger.info(f"START_DATE={start_str}, END_DATE={end_str}")

        except Exception as e:
            logger.error(f"Failed to modify PBS script: {e}")
            return None

        # Step 3: Submit the modified PBS job
        command = ["qsub", tmp_script_path]

        logger.info(f"Submitting PBS job for {model_name} (range: {start_dt} to {end_dt})")
        logger.info(f"Command: {' '.join(command)}")

    else:
        # Other models: Use YAML config approach

        # Step 1: Modify the YAML config with date range (use adjusted start time)
        try:
            config_path = modify_yaml_config_for_date_range(model_name, start_dt, end_dt)
            logger.info(f"Modified config for {model_name}: {config_path}")
            logger.info(f"Config will use adjusted range: {start_dt} to {end_dt}")
        except Exception as e:
            logger.error(f"Failed to modify config for {model_name}: {e}")
            return None

        # Step 2: Get the PBS script path
        pbs_script_path = (
            Path(__file__).parent
            / "scripts"
            / "start_end_pred_scripts"
            / f"run_{model_name}_inference_startend.sh"
        )

        if not pbs_script_path.exists():
            logger.error(f"PBS script not found: {pbs_script_path}")
            return None

        # Step 3: Modify PBS script to use absolute config path
        try:
            with open(pbs_script_path, "r") as f:
                script_content = f.read()

            # Replace $CFG_PATH with absolute path
            modified_script = script_content.replace('--cfg_path "$CFG_PATH"', f'--cfg_path "{config_path}"')

            # Write modified script to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as tmp:
                tmp.write(modified_script)
                tmp_script_path = tmp.name

            logger.info(f"Created modified PBS script: {tmp_script_path}")

        except Exception as e:
            logger.error(f"Failed to modify PBS script: {e}")
            return None

        # Step 4: Submit the modified PBS job
        command = ["qsub", tmp_script_path]

        logger.info(f"Submitting PBS job for {model_name} (range: {start_dt} to {end_dt})")
        logger.info(f"Command: {' '.join(command)}")
        logger.info(f"Config path: {config_path}")

    # Submit the job (common for both ED_ConvLSTM and other models)
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)

        # Extract job ID from output (format: "123456.davinci-mgt01")
        job_id = result.stdout.strip().split(".")[0]
        logger.info(f"✅ [{model_name}] Job submitted successfully! Job ID: {job_id}")

        # Clean up temp file
        try:
            Path(tmp_script_path).unlink()
        except:
            pass

        return job_id

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ [{model_name}] Failed to submit PBS job!")
        logger.error(f"Error: {e.stderr.strip() if e.stderr else 'Unknown error'}")
        # Clean up temp file
        try:
            Path(tmp_script_path).unlink()
        except:
            pass
        return None
    except Exception as e:
        logger.error(f"❌ [{model_name}] Unexpected error submitting job: {e}")
        # Clean up temp file
        try:
            Path(tmp_script_path).unlink()
        except:
            pass
        return None


def start_realtime_prediction_server(model: str, latest_sri: str) -> Optional[str]:
    """
    Launch a real-time inference subprocess on a GPU server (no job scheduler).

    Args:
        model: Model name (e.g. 'ConvLSTM')
        latest_sri: SRI filename (e.g. '12-04-2026-15-00.hdf')

    Returns:
        Job ID string 'server_{pid}' if successful, None if failed.
    """
    if model.upper() == "TEST":
        return None

    from nwc_webapp.config.config import get_config
    config = get_config()
    config_path = config.model_configs_path / "real_time" / f"{model}.yaml"

    if not config_path.exists():
        logger.error(f"Real-time config not found for {model}: {config_path}")
        return None

    server_cfg = config.server
    cmd = [
        "conda", "run", "-n", server_cfg.conda_env, "--no-capture-output",
        "python", server_cfg.inference_script_path,
        "--cfg_path", str(config_path),
    ]
    logger.info(f"[{model}] Launching real-time inference: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        job_id = f"server_{proc.pid}"
        logger.info(f"✅ [{model}] Inference started, PID: {proc.pid}")
        return job_id
    except Exception as e:
        logger.error(f"❌ [{model}] Failed to launch inference: {e}")
        return None
