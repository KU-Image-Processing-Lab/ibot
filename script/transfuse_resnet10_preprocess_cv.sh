# 環境変数
export PYTHONPATH=/home/student/git/medical-image-processing:$PYTHONPATH

# --------------------------------------------------------------
LF='
'
SCRIPT_DIR='transfuse-mtl-preprocessing'
TRAIN_SCRIPT_NAME='train_trans_nash.py'
TEST_SCRIPT_NAME='test_trans_nash.py'
DATE=`date '+%Y-%m-%d'`

echo "Date:                 ${DATE}"
echo "Script directory:     ${SCRIPT_DIR}"
echo "Training script path: ${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME}"
echo "Testing script path:  ${SCRIPT_DIR}/${TEST_SCRIPT_NAME}"

mkdir -p ./logs

# --------------------------------------------------------------
python3 ./utils/slack_bot.py --text "SCRIPT_DIR=${SCRIPT_DIR}${LF}TRAIN_SCRIPT_NAME=${TRAIN_SCRIPT_NAME}${LF}TEST_SCRIPT_NAME=${TEST_SCRIPT_NAME}"

weightArray=("best" "best2" "99")
echo "weightArray:          ${weightArray[@]}"

D_INIT=0.625
MEAN=0
STD=0.001
INIT_TYPE="random"

for i in {0..0}; do

    TRAIN_SAVE="transfuse_res2net_preprocess_cv${i}_${DATE}"
    echo "TRAIN_SAVE:           ${TRAIN_SAVE}"

    python3 ./utils/slack_bot.py --text "Cross Validation: ${i}${LF}TRAIN_SAVE=${TRAIN_SAVE}"
    python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
        --epoch 100 \
        --batchsize 4 \
        --trainsize 352 \
        --discriminator res2net \
        --preprocess resnet10 \
        --init_type ${INIT_TYPE} \
        --log_iter 20 \
        --save_epoch 50 \
        --tuning \
        --lambda_d 0.001 \
        --d_init ${D_INIT} \
        --mean ${MEAN} \
        --std ${STD} \
        --loss_dir ./results/${TRAIN_SAVE} \
        --seg_loss_path ./results/${TRAIN_SAVE}/seg_loss.png \
        --dis_loss_path ./results/${TRAIN_SAVE}/dis_loss.png \
        --pre_loss_path ./results/${TRAIN_SAVE}/pre_loss.png \
        --train_path ./dataset/CrossValidation_original/cv${i}_TrainDataset \
        --val_path ./dataset/CrossValidation_original/cv${i}_ValDataset \
        --train_save ${TRAIN_SAVE} \
        1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout_${DATE}.log >&1) \
        2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr_${DATE}.log >&2)
    python3 ./utils/slack_bot.py --text "`cat ./logs/${TRAIN_SCRIPT_NAME}_stderr_${DATE}.log`"

    for weight in ${weightArray[@]}; do

        python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
            --testsize 352 \
            --discriminator res2net \
            --preprocess resnet10 \
            --init_type ${INIT_TYPE} \
            --d_init ${D_INIT} \
            --mean ${MEAN} \
            --std ${STD} \
            --pth_path0 ./snapshots/${TRAIN_SAVE}/Preprocess-${weight}.pth \
            --pth_path ./snapshots/${TRAIN_SAVE}/TransFuse-${weight}.pth \
            --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-${weight}.pth \
            --test_data_path1 ./dataset/CrossValidation_original/cv${i}_TestDataset \
            --test_data_path2 ./dataset/CrossValidation_original/cv${i}_sekkai_TestDataset \
            --test_save_path ./results/${TRAIN_SAVE}/cv${i} \
            1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout_${DATE}.log >&1) \
            2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr_${DATE}.log >&2)
        python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stdout_${DATE}.log`"
        python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stderr_${DATE}.log`"

    done

done

D_INIT=0.625
MEAN=0
STD=0.001
INIT_TYPE="default"

for i in {0..0}; do

    TRAIN_SAVE="transfuse_resnet10_preprocess_cv${i}_${DATE}"
    echo "TRAIN_SAVE:           ${TRAIN_SAVE}"

    python3 ./utils/slack_bot.py --text "Cross Validation: ${i}${LF}TRAIN_SAVE=${TRAIN_SAVE}"
    python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
        --epoch 100 \
        --batchsize 4 \
        --trainsize 352 \
        --discriminator resnet10 \
        --preprocess resnet10 \
        --init_type ${INIT_TYPE} \
        --log_iter 20 \
        --save_epoch 50 \
        --tuning \
        --lambda_d 0.001 \
        --d_init ${D_INIT} \
        --mean ${MEAN} \
        --std ${STD} \
        --loss_dir ./results/${TRAIN_SAVE} \
        --seg_loss_path ./results/${TRAIN_SAVE}/seg_loss.png \
        --dis_loss_path ./results/${TRAIN_SAVE}/dis_loss.png \
        --pre_loss_path ./results/${TRAIN_SAVE}/pre_loss.png \
        --train_path ./dataset/CrossValidation_original/cv${i}_TrainDataset \
        --val_path ./dataset/CrossValidation_original/cv${i}_ValDataset \
        --train_save ${TRAIN_SAVE} \
        1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout_${DATE}.log >&1) \
        2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr_${DATE}.log >&2)
    python3 ./utils/slack_bot.py --text "`cat ./logs/${TRAIN_SCRIPT_NAME}_stderr_${DATE}.log`"

    for weight in ${weightArray[@]}; do

        python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
            --testsize 352 \
            --discriminator resnet10 \
            --preprocess resnet10 \
            --init_type ${INIT_TYPE} \
            --d_init ${D_INIT} \
            --mean ${MEAN} \
            --std ${STD} \
            --pth_path0 ./snapshots/${TRAIN_SAVE}/Preprocess-${weight}.pth \
            --pth_path ./snapshots/${TRAIN_SAVE}/TransFuse-${weight}.pth \
            --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-${weight}.pth \
            --test_data_path1 ./dataset/CrossValidation_original/cv${i}_TestDataset \
            --test_data_path2 ./dataset/CrossValidation_original/cv${i}_sekkai_TestDataset \
            --test_save_path ./results/${TRAIN_SAVE}/cv${i} \
            1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout_${DATE}.log >&1) \
            2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr_${DATE}.log >&2)
        python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stdout_${DATE}.log`"
        python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stderr_${DATE}.log`"

    done

done
