# Set the path to save checkpoints

# OUTPUT_DIR='./exp/dota_k400_videomae_pretrain_small_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600'
OUTPUT_DIR='exp/dota_k400_videomae_pretrain_small_patch16_224_frame_16x4_tube_mask_ratio_0.8_e1600'

# Set the path to Kinetics train set. 
# DATA_PATH='dota_train_home_normal.csv'
DATA_PATH='dota_train_normal.csv'

# source activate
# source activate videomae

# batch_size can be adjusted according to number of GPUs
# this script is for 32 GPUs (4 nodes x 8 GPUs)
OMP_NUM_THREADS=1 /home/hyde.hu/.conda/envs/videomae/bin/python -m torch.distributed.launch --nproc_per_node=8 \
--master_port 12320 --nnodes=1 --node_rank=0 run_mae_pretraining.py \
--data_path ${DATA_PATH} \
--mask_type tube \
--mask_ratio 0.8 \
--model pretrain_videomae_small_patch16_224 \
--model_path pretrained/vit-s-k400-ckpt.pth \
--decoder_depth 4 \
--batch_size 8 \
--num_frames 16 \
--sampling_rate 2 \
--opt adamw \
--input_size 560 320 \
--opt_betas 0.9 0.95 \
--warmup_epochs 40 \
--save_ckpt_freq 100 \
--epochs 1601 \
--log_dir ${OUTPUT_DIR} \
--output_dir ${OUTPUT_DIR}

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 \
#         --master_port 12320 --nnodes=1 --node_rank=0 \
#         run_mae_pretraining.py \
#         --data_path ${DATA_PATH} \
#         --mask_type tube \
#         --mask_ratio 0.9 \
#         --model pretrain_videomae_small_patch16_224 \
#         --decoder_depth 4 \
#         --batch_size 8 \
#         --num_frames 16 \
#         --sampling_rate 4 \
#         --opt adamw \
#         --opt_betas 0.9 0.95 \
#         --warmup_epochs 40 \
#         --save_ckpt_freq 20 \
#         --epochs 1601 \
#         --log_dir ${OUTPUT_DIR} \
#         --output_dir ${OUTPUT_DIR}
