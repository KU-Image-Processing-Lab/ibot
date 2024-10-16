# 環境変数
export PYTHONPATH=/home/student/git/medical-image-processing:$PYTHONPATH

# --------------------------------------------------------------
LF='
'
SCRIPT_DIR='transcaranet_ResNet10-mtl'
TRAIN_SCRIPT_NAME='train_trans_caranet_nash.py'
TEST_SCRIPT_NAME='test_trans_caranet_nash.py'
SCRIPT_DIR_2='transfuse-mtl'
TRAIN_SCRIPT_NAME_2='train_trans_nash.py'
TEST_SCRIPT_NAME_2='test_trans_nash.py'

mkdir -p ./logs

# --------------------------------------------------------------
TRAIN_SAVE="transcara_resnet10_cv_keisan"
TRAIN_SAVE_2="transfuse_res2net_cv_keisan"

python3 ./src/${SCRIPT_DIR_2}/${TRAIN_SCRIPT_NAME_2} \
    --epoch 2 \
    --batchsize 4 \
    --trainsize 352 \
    --discriminator res2net \
    --log_iter 200 \
    --save_epoch 100 \
    --tuning \
    --train_path ./dataset/CrossValidation/cv0_TrainDataset \
    --val_path ./dataset/CrossValidation/cv0_ValDataset \
    --train_save ${TRAIN_SAVE} \
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME_2}_stdout.log >&1) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME_2}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
    --epoch 2 \
    --batchsize 4 \
    --trainsize 352 \
    --discriminator resnet10 \
    --log_iter 200 \
    --save_epoch 100 \
    --tuning \
    --train_path ./dataset/CrossValidation/cv0_TrainDataset \
    --val_path ./dataset/CrossValidation/cv0_ValDataset \
    --train_save ${TRAIN_SAVE} \
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)


python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
    --epoch 6 \
    --batchsize 4 \
    --trainsize 352 \
    --discriminator resnet10 \
    --log_iter 200 \
    --save_epoch 100 \
    --tuning \
    --train_path ./dataset/CrossValidation/cv0_TrainDataset \
    --val_path ./dataset/CrossValidation/cv0_ValDataset \
    --train_save ${TRAIN_SAVE} \
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
    --epoch 11 \
    --batchsize 4 \
    --trainsize 352 \
    --discriminator resnet10 \
    --log_iter 200 \
    --save_epoch 100 \
    --tuning \
    --train_path ./dataset/CrossValidation/cv0_TrainDataset \
    --val_path ./dataset/CrossValidation/cv0_ValDataset \
    --train_save ${TRAIN_SAVE} \
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR_2}/${TRAIN_SCRIPT_NAME_2} \
    --epoch 6 \
    --batchsize 4 \
    --trainsize 352 \
    --discriminator res2net \
    --log_iter 200 \
    --save_epoch 100 \
    --tuning \
    --train_path ./dataset/CrossValidation/cv0_TrainDataset \
    --val_path ./dataset/CrossValidation/cv0_ValDataset \
    --train_save ${TRAIN_SAVE} \
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME_2}_stdout.log >&1) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME_2}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR_2}/${TRAIN_SCRIPT_NAME_2} \
    --epoch 11 \
    --batchsize 4 \
    --trainsize 352 \
    --discriminator res2net \
    --log_iter 200 \
    --save_epoch 100 \
    --tuning \
    --train_path ./dataset/CrossValidation/cv0_TrainDataset \
    --val_path ./dataset/CrossValidation/cv0_ValDataset \
    --train_save ${TRAIN_SAVE} \
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME_2}_stdout.log >&1) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME_2}_stderr.log >&2)