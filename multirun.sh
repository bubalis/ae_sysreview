name: data_prep.sh
cd /home/bdube/miniconda3
conda init bash
conda activate networks
cd /mnt/c/Users/benja/sys_review_dis




python citations_matrix.py
python author_network.py
python author_cite_network.py
