### train step 0
python -u -m torch.distributed.launch --nproc_per_node=1 run.py --batch_size 8 --logdir logs/10-10/ --dataset voc --task 10-10 --step 0 --lr 0.001 --epochs 30 --sample_num 10 --where_to_sim GPU_windows --data_root D:\\ADAXI\\Datasets\\VOC_SDR --val_interval 5 --cross_val --print_interval 50 --method FT --name first_step --loss_de_prototypes 1


### train step 1
python -u -m torch.distributed.launch --nproc_per_node=1 run.py --batch_size 8 --logdir logs/19-1/ --dataset voc --task 19-1 --step 1 --lr 0.0001 --epochs 30 --sample_num 10 --where_to_sim GPU_windows --data_root D:\\ADAXI\\Datasets\\VOC_SDR --val_interval 5 --cross_val --print_interval 50 --method SDR --name SDR_more_test --step_ckpt logs/19-1//19-1-voc_Experiment//19-1-voc_Experiment_0.pth

python -u -m torch.distributed.launch --nproc_per_node=1 run.py --batch_size 8 --logdir logs/10-10/ --dataset voc --task 10-10 --step 1 --lr 0.0001 --epochs 30 --sample_num 10 --where_to_sim GPU_windows --data_root D:\\ADAXI\\Datasets\\VOC_SDR --val_interval 5 --cross_val --print_interval 50 --method SDR --name inc_step --step_ckpt logs/10-10/10-10-voc_first_step/10-10-voc_first_step_0.pth

### test
python -u -m torch.distributed.launch --nproc_per_node=1 run.py --batch_size 8 --logdir logs/19-1/ --dataset voc --task 19-1 --step 0  --where_to_sim GPU_windows --data_root D:\\ADAXI\\Datasets\\VOC_SDR --test --name test --step_ckpt logs/19-1//19-1-voc_Experiment//19-1-voc_Experiment_0.pth

===========================

python -u -m torch.distributed.launch 1> 'outputs/19-1b/output_19-1b_step0_FT.txt'  2>&1 --nproc_per_node=1 run.py --batch_size 8 --logdir logs/19-1b/ --dataset voc --task 19-1b --step 0 --lr 0.001 --epochs 30 --sample_num 10 --where_to_sim GPU_windows --data_root D:\\ADAXI\\Datasets\\VOC_SDR --val_interval 5 --cross_val --print_interval 50 --method FT --name FT --loss_de_prototypes 1
