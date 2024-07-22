# 環境変数
export PYTHONPATH=/home/student/git/medical-image-processing:$PYTHONPATH

# --------------------------------------------------------------
LF='
'
SCRIPT_DIR='transfuse-mtl-preprocessing'
TRAIN_SCRIPT_NAME='train_trans_nash.py'
TEST_SCRIPT_NAME='test_trans_nash.py'
TRAIN_SAVE="transfuse_resnet10_preprocess"

mkdir -p ./logs

# --------------------------------------------------------------
python3 ./utils/slack_bot.py --text "SCRIPT_DIR=${SCRIPT_DIR}${LF}TRAIN_SCRIPT_NAME=${TRAIN_SCRIPT_NAME}${LF}TEST_SCRIPT_NAME=${TEST_SCRIPT_NAME}${LF}TRAIN_SAVE=${TRAIN_SAVE}"

# python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
#     --epoch 200 \
#     --batchsize 4 \
#     --trainsize 352 \
#     --discriminator resnet10 \
#     --log_epoch 20 \
#     --save_epoch 50 \
#     --train_save ${TRAIN_SAVE} \
#     1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1 ) \
#     2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)
# python3 ./utils/slack_bot.py --text `cat ./logs/${TRAIN_SCRIPT_NAME}_stderr.log`

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path0 ./snapshots/${TRAIN_SAVE}/Preprocess-best.pth \
    --pth_path ./snapshots/${TRAIN_SAVE}/Transfuse-best.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-best.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)
# python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stdout.log`"
# python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stderr.log`"

# python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
#     --testsize 352 \
#     --discriminator resnet10 \
#     --pth_path0 ./snapshots/${TRAIN_SAVE}/Preprocess-99.pth \
#     --pth_path ./snapshots/${TRAIN_SAVE}/Transfuse-99.pth \
#     --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-99.pth \
#     1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
#     2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)
# # python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stdout.log`"
# # python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stderr.log`"

# python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
#     --testsize 352 \
#     --discriminator resnet10 \
#     --pth_path0 ./snapshots/${TRAIN_SAVE}/Preprocess-199.pth \
#     --pth_path ./snapshots/${TRAIN_SAVE}/Transfuse-199.pth \
#     --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-199.pth \
#     1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
#     2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)
# # python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stdout.log`"
# # python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stderr.log`"
