# 環境変数
export PYTHONPATH=/home/student/git/medical-image-processing:$PYTHONPATH

# --------------------------------------------------------------

SCRIPT_DIR='transfuse-mtl'
TRAIN_SCRIPT_NAME='train_trans_nash.py'
TEST_SCRIPT_NAME='test_trans_nash.py'
TRAIN_SAVE="transfuse_resnet10"

mkdir -p ./logs

# --------------------------------------------------------------

python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
    --epoch 200 \
    --batchsize 4 \
    --trainsize 352 \
    --discriminator resnet10 \
    --log_epoch 50 \
    --save_epoch 50 \
    --train_save ${TRAIN_SAVE} \
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/Transfuse-best.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-best.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)
