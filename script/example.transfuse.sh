# 環境変数
export PYTHONPATH=/home/student/git/medical-image-processing:$PYTHONPATH

# --------------------------------------------------------------

SCRIPT_DIR='transfuse'
TRAIN_SCRIPT_NAME='train_trans.py'
TEST_SCRIPT_NAME='test_trans.py'
TRAIN_SAVE="transfuse"

mkdir -p ./logs

# --------------------------------------------------------------

python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
    --epoch 100 \
    --batchsize 4 \
    --trainsize 352 \
    --train_save ${TRAIN_SAVE} \
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --pth_path ./snapshots/${TRAIN_SAVE}/Transfuse-best.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)
