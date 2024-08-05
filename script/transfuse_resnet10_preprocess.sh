# 環境変数
export PYTHONPATH=/home/student/git/medical-image-processing:$PYTHONPATH

# --------------------------------------------------------------
LF='
'
SCRIPT_DIR='transfuse-mtl'
TRAIN_SCRIPT_NAME='train_trans_nash.py'
TEST_SCRIPT_NAME='test_trans_nash.py'
TRAIN_SAVE="transfuse_resnet10_preprocess"

mkdir -p ./logs

# --------------------------------------------------------------
python3 ./utils/slack_bot.py --text "SCRIPT_DIR=${SCRIPT_DIR}${LF}TRAIN_SCRIPT_NAME=${TRAIN_SCRIPT_NAME}${LF}TEST_SCRIPT_NAME=${TEST_SCRIPT_NAME}${LF}TRAIN_SAVE=${TRAIN_SAVE}"

python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
    --epoch 200 \
    --batchsize 4 \
    --trainsize 352 \
    --discriminator resnet10 \
    --log_iter 20 \
    --save_epoch 50 \
    --train_path ./dataset/original_dataset_only_left/TrainDataset \
    --val_path ./dataset/original_dataset_only_left/ValDataset \
    --train_save ${TRAIN_SAVE} \
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)
python3 ./utils/slack_bot.py --text "`cat ./logs/${TRAIN_SCRIPT_NAME}_stderr.log`"

WEIGHT='best'
python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path0 ./snapshots/${TRAIN_SAVE}/Preprocess-${WEIGHT}.pth \
    --pth_path ./snapshots/${TRAIN_SAVE}/Transfuse-${WEIGHT}.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-${WEIGHT}.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout-${WEIGHT}.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr-${WEIGHT}.log >&2)
python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stdout-${WEIGHT}.log`"
python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stderr-${WEIGHT}.log`"

WEIGHT='best2'
python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path0 ./snapshots/${TRAIN_SAVE}/Preprocess-${WEIGHT}.pth \
    --pth_path ./snapshots/${TRAIN_SAVE}/Transfuse-${WEIGHT}.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-${WEIGHT}.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout-${WEIGHT}.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr-${WEIGHT}.log >&2)
python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stdout-${WEIGHT}.log`"
python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stderr-${WEIGHT}.log`"

WEIGHT='99'
python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path0 ./snapshots/${TRAIN_SAVE}/Preprocess-${WEIGHT}.pth \
    --pth_path ./snapshots/${TRAIN_SAVE}/Transfuse-${WEIGHT}.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-${WEIGHT}.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout-${WEIGHT}.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr-${WEIGHT}.log >&2)
python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stdout-${WEIGHT}.log`"
python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stderr-${WEIGHT}.log`"

WEIGHT='199'
python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path0 ./snapshots/${TRAIN_SAVE}/Preprocess-${WEIGHT}.pth \
    --pth_path ./snapshots/${TRAIN_SAVE}/Transfuse-${WEIGHT}.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-${WEIGHT}.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout-${WEIGHT}.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr-${WEIGHT}.log >&2)
python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stdout-${WEIGHT}.log`"
python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stderr-${WEIGHT}.log`"
