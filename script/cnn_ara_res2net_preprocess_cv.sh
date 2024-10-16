# 環境変数
export PYTHONPATH=/home/student/git/medical-image-processing:$PYTHONPATH

# --------------------------------------------------------------
LF='
'
SCRIPT_DIR='transcaranet_CNN_ARA-mtl'
TRAIN_SCRIPT_NAME='train_trans_caranet_nash.py'
TEST_SCRIPT_NAME='test_trans_caranet_nash.py'

mkdir -p ./logs

# --------------------------------------------------------------
# python3 ./utils/slack_bot.py --text "SCRIPT_DIR=${SCRIPT_DIR}${LF}TRAIN_SCRIPT_NAME=${TRAIN_SCRIPT_NAME}${LF}TEST_SCRIPT_NAME=${TEST_SCRIPT_NAME}"

weightArray=("best" "best2" "99" "199")

for i in {0..4}; do

    TRAIN_SAVE="cnn_ara_res2net_preprocess_cv${i}"

    # python3 ./utils/slack_bot.py --text "Cross Validation: ${i}"
    python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
        --epoch 200 \
        --batchsize 4 \
        --trainsize 352 \
        --discriminator res2net \
        --log_iter 50 \
        --save_epoch 100 \
        --train_path ./dataset/CrossValidation/cv${i}_TrainDataset \
        --val_path ./dataset/CrossValidation/cv${i}_ValDataset \
        --train_save ${TRAIN_SAVE} \
        1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1) \
        2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)
    # python3 ./utils/slack_bot.py --text "`cat ./logs/${TRAIN_SCRIPT_NAME}_stderr.log`"

    for weight in ${weightArray[@]}; do

        python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
            --testsize 352 \
            --discriminator res2net \
            --pth_path0 ./snapshots/${TRAIN_SAVE}/Preprocess-${weight}.pth \
            --pth_path ./snapshots/${TRAIN_SAVE}/Transfuse-${weight}.pth \
            --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-${weight}.pth \
            --test_data_path1 ./dataset/CrossValidation_original/cv${i}_TestDataset \
            --test_data_path2 ./dataset/CrossValidation_original/cv${i}_sekkai_TestDataset \
            --save_path ./results/${TRAIN_SAVE}/cv${i} \
            1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1) \
            2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)
        # python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stdout.log`"
        # python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stderr.log`"

    done

done
