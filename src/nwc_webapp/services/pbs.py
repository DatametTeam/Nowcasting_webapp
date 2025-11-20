import os.path
import re
import subprocess
from pathlib import Path

# from constants import OUTPUT_DATA_DIR, TARGET_GPU
from datetime import datetime

from nwc_webapp.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

def is_pbs_available() -> bool:
    """Check if PBS is available by running qstat silently."""
    try:
        result = subprocess.run(
            ["qstat", "-u", "guidim"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2
        )
        return result.returncode == 0
    except:
        return False


def get_job_status(job_id):
    try:
        # Run the command and capture the output
        result = subprocess.run(
            ["qstat", "-f", job_id],
            check=True,
            text=True,
            capture_output=True
        )
        # Filter for the "job_state" line
        for line in result.stdout.splitlines():
            if "job_state" in line:
                # Extract the status after the "=" sign
                _, status = line.split("=", 1)
                return status.strip()  # Remove any surrounding whitespace
        return "ended"  # job_state line not found, assume the job ended
    except subprocess.CalledProcessError:
        # If qstat fails, the job likely doesn't exist
        return "ended"


def get_model_job_status(model):
    """
    Check if there's a job running/queued for the specified model.
    Uses stored job_id for fast direct lookup.

    Args:
        model: Model name (e.g., 'ED_ConvLSTM', 'pystep')

    Returns:
        str: 'Q' (queued), 'R' (running), or None (no job found)
    """
    try:
        # Try fast path: check stored job_id directly
        try:
            import streamlit as st
            job_id = st.session_state.get(f"job_id_{model}")
            if job_id:
                # Direct check for this specific job
                result = subprocess.run(
                    ["qstat", "-f", job_id],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=2
                )
                if result.returncode == 0:
                    # Job still exists - extract status from line 3 (always "job_state = X")
                    lines = result.stdout.splitlines()
                    for line in lines:
                        if "job_state" in line:
                            status = line.split("=", 1)[1].strip()
                            logger.debug(f"[QSTAT-FAST] Model {model}: Job {job_id} - Status={status}")
                            return status
                # Job doesn't exist anymore - clear stored id
                del st.session_state[f"job_id_{model}"]
                logger.debug(f"[QSTAT-FAST] Model {model}: Job {job_id} ended, cleared stored ID")
                return None
        except Exception:
            pass  # Fall back to slow method

        # Slow fallback path: search all jobs (for backward compatibility)
        job_name = f"nwc_{model}"
        result = subprocess.run(
            ["qstat", "-u", "guidim"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

        for line in result.stdout.split('\n'):
            if line.startswith("Job id") or line.startswith("---") or line.startswith("davinci") or not line.strip():
                continue

            if line and line[0].isdigit():
                job_id = line.split()[0].split('.')[0]
                result2 = subprocess.run(
                    ["qstat", "-f", f"{job_id}"],
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL
                )
                if job_name in result2.stdout:
                    for line2 in result2.stdout.splitlines():
                        if "job_state" in line2:
                            status = line2.split("=", 1)[1].strip()
                            logger.info(f"[QSTAT-SLOW] Model {model}: FOUND - Status={status}")
                            return status

        # Job not in queue - return None silently (already logged in workers.py)
        return None

    except subprocess.CalledProcessError:
        return None
    except Exception:
        return None


def get_pbs_header(job_name, q_name, pbs_log_path, target_gpu=None):
    if target_gpu is None:
        return f"""
#PBS -N {job_name}
#PBS -q {q_name}
#PBS -l select=1:ncpus=0:ngpus=0
#PBS -k oe
#PBS -j oe
#PBS -o {pbs_log_path} 
"""
    else:
        return f"""
#PBS -N {job_name}
#PBS -q {q_name}
#PBS -l host={target_gpu},walltime=12:00:00
#PBS -k oe
#PBS -j oe
#PBS -o {pbs_log_path} 
"""


# TO UPDATE!
def get_pbs_env(model):
    if model == 'ED_ConvLSTM':
        env = f"""
            module load proxy
            module load anaconda3
            source activate protezionecivile
            """
    elif model == 'pystep':

        env = f"""
            module load proxy
            module load anaconda3
            source activate nowcasting
            """
    else:
        env = f"""
            module load proxy
            module load anaconda3
            source activate nowcasting3.12
            """
    return env


def submit_inference(args) -> tuple[str, str]:
    return 0, []

    start_date, end_date, model_name, submitted = args

    # TO UPDATE!
    out_dir = OUTPUT_DATA_DIR / model_name / start_date.strftime("%Y%m%d") / end_date.strftime("%Y%m%d")

    # DEFINE THE OUTPUT DIRECTORY ! TO UPDATE!
    date_now = datetime.now().strftime("%Y%m%d%H%M%S")
    out_images_dir = out_dir / "generations" / date_now
    out_images_dir.mkdir(parents=True, exist_ok=True)

    fine_tuned_model_dir = out_dir / "finetuned_model"

    cmd_string = f"""
python3 "$WORKDIR/faradai/dreambooth_scripts/run_inference.py" \
--fine-tuned-model-dir={str(fine_tuned_model_dir)}
"""

    logger.info(f"cmd_string: \n > {cmd_string}")
    pbs_logs = out_dir / "pbs_logs"
    pbs_logs.mkdir(parents=True, exist_ok=True)

    pbs_script = "#!/bin/bash"
    pbs_script += get_pbs_header("sole24ore_demo", TARGET_GPU, str(pbs_logs / "pbs.log"))
    pbs_script += get_pbs_env()
    pbs_script += f"\n{cmd_string}"

    pbs_scripts = out_dir / "pbs_script"
    pbs_scripts.mkdir(parents=True, exist_ok=True)
    pbs_script_path = pbs_scripts / "run_inference.sh"
    with open(pbs_script_path, "w", encoding="utf-8") as f:
        f.write(pbs_script)

    # Command to execute the script with qsub
    command = ["qsub", pbs_script_path]

    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        logger.info("Inference job submitted successfully!")
        job_id = result.stdout.strip().split(".davinci-mgt01")[0]
        logger.info("Job ID:", job_id)
        return job_id, out_images_dir

    except subprocess.CalledProcessError as e:
        logger.info("Error occurred while submitting the job!")
        logger.info("Error message:", e.stderr.strip())
        return None, None


def start_prediction_job(model, latest_data):
    # Defensive check - never submit job for TEST model
    if model == "TEST" or model.upper() == "TEST":
        logger.warning(f"[{model}] TEST model detected - skipping job submission")
        return None

    latest_data = latest_data.split('.')[0]

    if model == 'ED_ConvLSTM':

        cmd_string = f"""
    python "/davinci-1/work/protezionecivile/backup_old_stuff/nowcasting_OLD_TEO_CODE/nwc_test_webapp.py" \
        start_date={str(latest_data)}
        """
    else:
        cmd_string = f"""
    python "/davinci-1/home/guidim/spatiotemporal-nowcast/spatiotemporal_forecast/scripts/webapp_predictions.py" \
        --cfg_path "/davinci-1/work/protezionecivile/nwc_webapp/configs/{model}.yaml"
        """

    logger.info(f"cmd_string: \n > {cmd_string}")
    home_dir = Path.home()
    pbs_logs = home_dir / "pbs_logs"
    pbs_logs.mkdir(parents=True, exist_ok=True)

    pbs_script = "#!/bin/bash"
    pbs_script += get_pbs_header(f"nwc_{model}", 'fast', str(pbs_logs / "pbs.log"))
    pbs_script += get_pbs_env(model)
    pbs_script += f"\n{cmd_string}"

    src_dir = Path(__file__).resolve().parent.parent
    pbs_scripts = Path(os.path.join(src_dir, "pbs_scripts"))
    pbs_scripts.mkdir(parents=True, exist_ok=True)
    pbs_script_path = pbs_scripts / f"run_{model}_inference.sh"
    with open(pbs_script_path, "w", encoding="utf-8") as f:
        f.write(pbs_script)

    # Command to execute the script with qsub
    command = ["qsub", pbs_script_path]

    try:
        logger.info("COMMAND")
        logger.info(command)
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        logger.info("Inference job submitted successfully!")
        job_id = result.stdout.strip().split(".")[0]
        logger.info(f"Job ID: {job_id}")
        return job_id

    except subprocess.CalledProcessError as e:
        logger.error("Error occurred while submitting the job!")
        logger.error(f"Error message: {e.stderr.strip()}")
        return None
