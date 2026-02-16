#!/bin/bash
#PBS -N nwc_ED_ConvLSTM
#PBS -q fast
#PBS -l select=1:ncpus=0:ngpus=0
#PBS -k oe
#PBS -j oe
#PBS -o /davinci-1/home/guidim/pbs_logs/pbs.log

            module load proxy
            module load anaconda3
            source activate protezionecivile
            

    python "/davinci-1/work/protezionecivile/backup_old_stuff/nowcasting_OLD_TEO_CODE/nwc_test_webapp.py"         start_date=13-02-2026-11-35
        