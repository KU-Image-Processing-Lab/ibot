# 環境変数
export PYTHONPATH=/home/student/Git/medical-image-processing:$PYTHONPATH

# --------------------------------------------------------------

SCRIPT_DIR='improvement-discriminator'
TRAIN_SCRIPT_NAME='train_trans_nash_bi.py'
TEST_SCRIPT_NAME='test_trans_nash.py'

sudo mkdir -p ./logs

# --------------------------------------------------------------
weightArray=("best" "best2" "99")

TRAIN_SCRIPT_NAME='train_trans_nash_bi.py'
TRAIN_SAVE="transfuse_resnet10_improve_bi_test"

for i in {0..4}; do
    TRAIN_SAVE="transfuse_resnet10_bi__cv${i}"
    python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
        --epoch 100 \
        --batchsize 4 \
        --trainsize 352 \
        --discriminator resnet10 \
        --seg_loss_path ./fig/seg_loss_img_bi_${d}.png \
        --dis_loss_path ./fig/dis_loss_img_bi_${d}.png \
        --log_iter 200 \
        --save_epoch 50 \
        --tuning \
        --train_path ./dataset/CrossValidation/cv${i}_TrainDataset \
        --val_path ./dataset/CrossValidation/cv${i}_ValDataset \
        --train_save ${TRAIN_SAVE}/cv${i} \
        --mae_weight ./weights/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth
    for weight in ${weightArray[@]}; do
        python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
            --testsize 352 \
            --discriminator resnet10 \
            --pth_path ./snapshots/${TRAIN_SAVE}/cv${i}/Transfuse-${weight}.pth \
            --pth_path2 ./snapshots/${TRAIN_SAVE}/cv${i}/Discriminator-${weight}.pth \
            --test_data_path1 ./dataset/CrossValidation/cv${i}_TestDataset \
            --test_data_path2 ./dataset/CrossValidation/cv${i}_sekkai_TestDataset \
            --test_save_path ./results/${TRAIN_SAVE}/cv${i}/ 
    done
done
