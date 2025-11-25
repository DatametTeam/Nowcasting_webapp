#!/bin/bash
#PBS -N sole24ore_demo
#PBS -q fast
#PBS -l select=1:ncpus=0:ngpus=0
#PBS -k oe
#PBS -j oe
#PBS -o /Users/matte/pbs_logs/pbs.log

            module load proxy
            module load anaconda3
            source activate protezionecivile


    python "/davinci-1/work/protezionecivile/backup_old_stuff/nowcasting_OLD_TEO_CODE/nwc_test_webapp.py"         start_date=11-11-2025-11-49
