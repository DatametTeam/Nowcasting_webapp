import os.path
import subprocess
from pathlib import Path

# from constants import OUTPUT_DATA_DIR, TARGET_GPU
from datetime import datetime

from nwc_webapp.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

def is_pbs_available() -> bool:
    import subprocess
    return subprocess.call(["qstat"], shell=True) == 0


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

    Args:
        model: Model name (e.g., 'ED_ConvLSTM', 'pystep')

    Returns:
        str: 'Q' (queued), 'R' (running), or None (no job found)
    """
    try:
        job_name = f"nwc_{model}"
        # Run qstat to get all jobs
        result = subprocess.run(
            ["qstat"],
            check=True,
            text=True,
            capture_output=True
        )

        # Parse qstat output to find jobs with matching name
        for line in result.stdout.splitlines():
            if job_name in line:
                # qstat output format: job_id job_name user time status queue
                # Status is typically the 5th column (index 4)
                parts = line.split()
                if len(parts) >= 5:
                    status = parts[4]  # 'Q' for queued, 'R' for running
                    logger.debug(f"Found job for model {model} with status: {status}")
                    return status

        return None  # No job found for this model

    except subprocess.CalledProcessError:
        logger.warning("qstat command failed - PBS might not be available")
        return None
    except Exception as e:
        logger.error(f"Error checking job status for model {model}: {e}")
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
            source activate sole24_310
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
        job_id = result.stdout.strip().split(".davinci-mgt01")[0]
        logger.info("Job ID:", job_id)
        return job_id

    except subprocess.CalledProcessError as e:
        logger.error("Error occurred while submitting the job!")
        logger.error("Error message:", e.stderr.strip())
        return None
