name: data_prep.sh
cd /home/bdube/miniconda3
conda init bash
conda activate networks
cd /mnt/c/Users/benja/sys_review_dis


python find_cites.py
python author_work.py
python text_analysis.py
