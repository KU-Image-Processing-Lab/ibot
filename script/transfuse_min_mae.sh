# 環境変数
export PYTHONPATH=/home/student/Git/medical-image-processing:$PYTHONPATH

# --------------------------------------------------------------

SCRIPT_DIR='mae'
TRAIN_SCRIPT_NAME='main_pretrain_deit.py'

sudo mkdir -p ./logs

# --------------------------------------------------------------

for i in {0..4}; do
    TRAIN_SAVE="transfuse_min_mae_cv${i}"
    python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
        --batch_size 64 \
        --epochs 400\
        --accum_iter 1\
        --model mae_vit_large_patch16\
        --input_size 352\
        --mask_ratio 0.75\
        --norm_pix_loss \
        --weight_decay  0.05\
        --lr 0.001\
        --blr 1e-3\
        --min_lr 0.0\
        --warmup_epochs 40\
        --data_path ./dataset/CrossValidation/mae/0/\
        --output_dir ./snapshots/${TRAIN_SAVE}/cv${i}\
        --log_dir ./logs/${TRAIN_SAVE}/cv${i}\
        --device cuda\
        --seed 0\
        --resume ""\
        --start_epoch 0\
        --num_workers 10\
        --pin_mem \
        --world_size 1\
        --local_rank 1\
        --dist_on_itp \
        --dist_url "env://"
done