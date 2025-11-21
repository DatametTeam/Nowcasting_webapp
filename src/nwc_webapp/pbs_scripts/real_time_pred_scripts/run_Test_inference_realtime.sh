#!/bin/bash
#PBS -N nwc_Test
#PBS -q fast
#PBS -l select=1:ncpus=0:ngpus=0
#PBS -k oe
#PBS -j oe
#PBS -o /davinci-1/home/guidim/pbs_logs/pbs.log 

            module load proxy
            module load anaconda3
            source activate nowcasting3.12
            

    python "/davinci-1/home/guidim/spatiotemporal-nowcast/spatiotemporal_forecast/scripts/webapp_predictions.py"         --cfg_path "/davinci-1/work/protezionecivile/nwc_webapp/configs/Test.yaml"
        