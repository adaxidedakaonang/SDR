### train step 0
python -u -m torch.distributed.launch --nproc_per_node=1 run.py --batch_size 8 --logdir logs/offline/ --dataset voc --task offline --step 0 --lr 0.001 --epochs 30 --sample_num 10 --where_to_sim GPU_windows --data_root D:\\ADAXI\\Datasets\\VOC_SDR --val_interval 5 --cross_val --print_interval 50 --method FT --name FT


### train step 1
python -u -m torch.distributed.launch --nproc_per_node=1 run.py --batch_size 8 --logdir logs/19-1/ --dataset voc --task 19-1 --step 1 --lr 0.0001 --epochs 30 --sample_num 10 --where_to_sim GPU_windows --data_root D:\\ADAXI\\Datasets\\VOC_SDR --val_interval 5 --cross_val --print_interval 50 --method SDR --name SDR_more_test --step_ckpt logs/19-1//19-1-voc_Experiment//19-1-voc_Experiment_0.pth --replay_path D:\ADAXI\Datasets\increment\replay_images_and_labels\19

python -u -m torch.distributed.launch --nproc_per_node=1 run.py --batch_size 8 --logdir logs/19-1/ --dataset voc --task 19-1 --step 1 --lr 0.0001 --epochs 30 --sample_num 10 --where_to_sim GPU_windows --data_root D:\\ADAXI\\Datasets\\VOC_SDR --val_interval 9 --print_interval 50 --method FT --name F_M_R --freeze --mix --replay --step_ckpt logs/19-1/19-1-voc_FT/19-1-voc_FT_0.pth

python -u -m torch.distributed.launch --nproc_per_node=1 run.py --batch_size 8 --logdir logs/19-1/ --dataset voc --task 19-1 --step 1 --lr 0.0001 --epochs 30 --sample_num 10 --where_to_sim GPU_windows --data_root D:\\ADAXI\\Datasets\\VOC_SDR --val_interval 9 --print_interval 50 --method FT --name FT_F_R_M --freeze --mix --replay --step_ckpt logs/19-1//19-1-voc_FT//19-1-voc_FT_0.pth --replay_path D:\ADAXI\Datasets\increment\replay_images_and_labels\19

### test
python -u -m torch.distributed.launch --nproc_per_node=1 run.py --batch_size 8 --logdir logs/offline/ --dataset voc --task offline --step 0  --where_to_sim GPU_windows --data_root D:\\ADAXI\\Datasets\\VOC_SDR --test --name test --ckpt logs/offline//offline-voc_SDR//offline-voc_SDR_0.pth

===========================

python -u -m torch.distributed.launch --nproc_per_node=1 run.py --batch_size 8 --logdir logs/10-10/ --dataset voc --task 10-10 --step 1 --lr 0.0001 --epochs 30 --sample_num 10 --where_to_sim GPU_windows --data_root D:\\ADAXI\\Datasets\\VOC_SDR --val_interval 5 --print_interval 50 --method FT --name freeze_body_and_replay_and_mix_interleave_new_dataset --freeze --replay --mix --step_ckpt logs/10-10/10-10-voc_FT/10-10-voc_FT.pth
