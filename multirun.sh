me: data_prep.sh
cd /home/bdube/miniconda3
conda init bash
conda activate networks
cd /mnt/c/Users/benja/sys_review_dis

#python -m data_prep.crop_field_setup
#python -m data_prep.Raster_processing
python find_cites.py
python author_work.py
python co_occurence_matrix.py
