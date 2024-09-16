# 環境変数
export PYTHONPATH=/home/student/git/medical-image-processing:$PYTHONPATH

# --------------------------------------------------------------
LF='
'
SCRIPT_DIR='transfuse-mtl'
TRAIN_SCRIPT_NAME='train_trans_nash.py'
TEST_SCRIPT_NAME='test_trans_nash.py'

mkdir -p ./logs

# --------------------------------------------------------------
python3 ./utils/slack_bot.py --text "SCRIPT_DIR=${SCRIPT_DIR}${LF}TRAIN_SCRIPT_NAME=${TRAIN_SCRIPT_NAME}${LF}TEST_SCRIPT_NAME=${TEST_SCRIPT_NAME}"

# weightArray=("best" "best2" "99" "199")
weightArray=("best" "best2" "99")
# paramD=(0.625 0.615 0.62 0.63 0.635 0.55 0.575 0.6 0.65 0.675 0.7)
paramD=(0.6 0.615 0.62 0.625 0.63 0.635 0.65 0.675)

for d in ${paramD[@]}; do
    for i in {1..4}; do

        TRAIN_SAVE="transfuse_resnet10_const_param_d_${d}_cv${i}"

        python3 ./utils/slack_bot.py --text "Param D: ${d}"
        python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
            --epoch 100 \
            --batchsize 4 \
            --trainsize 352 \
            --discriminator resnet10 \
            --seg_loss_path ./fig/seg_loss_img_${d}.png \
            --dis_loss_path ./fig/dis_loss_img_${d}.png \
            --log_iter 50 \
            --save_epoch 50 \
            --tuning \
            --train_path ./dataset/CrossValidation_original_crop_${d}/cv${i}_TrainDataset \
            --val_path ./dataset/CrossValidation_original_crop_${d}/cv${i}_ValDataset \
            --train_save ${TRAIN_SAVE} \
            1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1) \
            2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)
        python3 ./utils/slack_bot.py --text "`cat ./logs/${TRAIN_SCRIPT_NAME}_stderr.log`"

        for weight in ${weightArray[@]}; do

            python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
                --testsize 352 \
                --discriminator resnet10 \
                --pth_path ./snapshots/${TRAIN_SAVE}/Transfuse-${weight}.pth \
                --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-${weight}.pth \
                --test_data_path1 ./dataset/CrossValidation_original_crop_${d}/cv${i}_TestDataset \
                --test_data_path2 ./dataset/CrossValidation_original_crop_${d}/cv${i}_sekkai_TestDataset \
                --test_save_path ./results/${TRAIN_SAVE}/cv${i}/ \
                1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1) \
                2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)
            python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stdout.log`"
            # python3 ./utils/slack_bot.py --text "`cat ./logs/${TEST_SCRIPT_NAME}_stderr.log`"

        done
    done
done
