#!/bin/bash
#PBS -N nwc_PredFormer_range
#PBS -q fast
#PBS -l select=1:ncpus=0:ngpus=0
#PBS -k oe
#PBS -j oe
#PBS -o /davinci-1/home/guidim/pbs_logs/pbs.log

module load proxy
module load anaconda3
source activate nowcasting3.12

# Construct cfg_path dynamically relative to script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CFG_PATH="${SCRIPT_DIR}/../../resources/cfg/start_end_prediction_cfg/PredFormer.yaml"

python "/davinci-1/home/guidim/spatiotemporal-nowcast/spatiotemporal_forecast/scripts/webapp_predictions.py" --cfg_path "$CFG_PATH"
