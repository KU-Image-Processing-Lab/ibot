# 環境変数
export PYTHONPATH=/home/student/Git/medical-image-processing:$PYTHONPATH

# --------------------------------------------------------------

SCRIPT_DIR='improvement-discriminator'
TRAIN_SCRIPT_NAME='train_trans_nash_pool.py'
TEST_SCRIPT_NAME='test_trans_nash.py'

mkdir -p ./logs

# --------------------------------------------------------------
weightArray=("best" "best2" "99")
poolmode=("max" "avg")
kernel_sizes=(3 2 4)
strides=(2 2 4)

for i in {0..4}; do
    for pool in "${poolmode[@]}"; do
        for ((idx=0; idx<${#kernel_sizes[@]}; idx++)); do
            k=${kernel_sizes[$idx]}
            s=${strides[$idx]}
            TRAIN_SAVE="transfuse_resnet10_${pool}_cv${i}"
            python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
                --epoch 100 \
                --batchsize 4 \
                --trainsize 352 \
                --discriminator resnet10 \
                --seg_loss_path ./fig/seg_loss_img_bicu_${i}_${pool}_${k}${s}.png \
                --dis_loss_path ./fig/dis_loss_img_bicu_${i}_${pool}_${k}${s}.png \
                --log_iter 300 \
                --save_epoch 100 \
                --train_path ./dataset/CrossValidation/cv${i}_TrainDataset \
                --val_path ./dataset/CrossValidation/cv${i}_ValDataset \
                --train_save ${TRAIN_SAVE}/${pool}_${k}_${s} \
                --mae_weight ./weights/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth \
                --kernel_size ${k} \
                --stride ${s} \
                --pool ${pool}
            for weight in "${weightArray[@]}"; do
                ROC_SAVE="transfuse_resnet10_${pool}_cv${i}_${weight}_${k}_${s}"
                python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
                    --testsize 352 \
                    --discriminator resnet10 \
                    --pth_path ./snapshots/${TRAIN_SAVE}/${pool}_${k}_${s}/Transfuse-${weight}.pth \
                    --pth_path2 ./snapshots/${TRAIN_SAVE}/${pool}_${k}_${s}/Discriminator-${weight}.pth \
                    --test_data_path1 ./dataset/CrossValidation/cv${i}_TestDataset \
                    --test_data_path2 ./dataset/CrossValidation/cv${i}_sekkai_TestDataset \
                    --test_save_path ./results/${TRAIN_SAVE}/${pool}_${k}_${s}/cv${i}/ \
                    --test_roc_save ./fig/${ROC_SAVE}/
            done
        done
    done
done





# for pool in "${poolmode[@]}"; do
#     for ks_pair in "${kernel_stride_pairs[@]}"; do
#         set -- $ks_pair
#         k=$1
#         s=$2
#         for weight in "${weightArray[@]}"; do
#             TRAIN_SAVE="transfuse_resnet10_${pool}_cv${i}_${k}_${s}"
#             for i in {0..4}; do
#                 ROC_SAVE="transfuse_resnet10_${pool}_cv${i}_${weight}_${k}_${s}"
#                 python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
#                     --testsize 352 \
#                     --discriminator resnet10 \
#                     --pth_path ./snapshots/${TRAIN_SAVE}/Transfuse-${weight}.pth \
#                     --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-${weight}.pth \
#                     --test_data_path1 ./dataset/CrossValidation/cv${i}_TestDataset \
#                     --test_data_path2 ./dataset/CrossValidation/cv${i}_sekkai_TestDataset \
#                     --test_save_path ./results/${TRAIN_SAVE}/cv${i}/ \
#                     --test_roc_save ./fig/${ROC_SAVE}/ \
#                     --pool_mode ${pool}
#             done
#         done
#     done
# done