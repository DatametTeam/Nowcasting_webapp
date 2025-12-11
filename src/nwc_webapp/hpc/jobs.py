"""
Job submission utilities for date-range predictions.
Handles both HPC (PBS) and local (mock) job submission.
"""

import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml

from nwc_webapp.config.environment import is_hpc
from nwc_webapp.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)


def modify_yaml_config_for_date_range(model_name: str, start_dt: datetime, end_dt: datetime) -> Path:
    """
    Modify YAML config with start/end dates for date-range predictions.

    Reads the config from config/model_configs/start_end/{model_name}.yaml,
    modifies the start_dt and end_dt fields in the start_end slicer, and overwrites the file.

    Args:
        model_name: Model name (e.g., 'ConvLSTM', 'IAM4VP', 'PredFormer', 'SPROG')
        start_dt: Start datetime
        end_dt: End datetime

    Returns:
        Path to the modified YAML config file
    """
    # Source YAML path
    config_path = Path(__file__).parent.parent / "config" / "model_configs" / "start_end" / f"{model_name}.yaml"

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Read the YAML file
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Format dates as "YYYY-MM-DD HH:MM:SS" (new format includes seconds)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    # Modify the start_dt and end_dt fields in the start_end slicer
    if "dataframe_strategy" in config_data and "slicers" in config_data["dataframe_strategy"]:
        # Find the start_end slicer and update its args
        start_end_slicer_found = False
        for slicer in config_data["dataframe_strategy"]["slicers"]:
            if slicer.get("name") == "start_end":
                slicer["args"]["start_dt"] = start_str
                slicer["args"]["end_dt"] = end_str
                start_end_slicer_found = True
                logger.info(f"Modified {model_name} config: start_dt={start_str}, end_dt={end_str}")
                break

        if not start_end_slicer_found:
            logger.warning(f"Could not find start_end slicer in {model_name} config")
    else:
        logger.warning(f"Could not find dataframe_strategy.slicers in {model_name} config")

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

    # Check if running locally
    if not is_hpc():
        logger.info(f"🖥️  Running in LOCAL mode - generating mock predictions for {model_name}")

        try:
            from nwc_webapp.mock.generator import generate_mock_predictions_for_range

            # Generate mock predictions with ORIGINAL start_dt (not adjusted)
            # The mock generator should handle predictions starting from start_dt
            created_count = generate_mock_predictions_for_range(model_name, start_dt, end_dt)

            if created_count >= 0:
                # Return a fake job ID for UI compatibility
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
            Path(__file__).parent.parent
            / "pbs_scripts"
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
            Path(__file__).parent.parent
            / "pbs_scripts"
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
