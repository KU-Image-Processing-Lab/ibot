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
weightArray=("best" "best2" "99" "199")

for i in {0..4}; do

    TRAIN_SAVE="transfuse_res2net_cv${i}"

    # python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
    #     --epoch 200 \
    #     --batchsize 4 \
    #     --trainsize 352 \
    #     --discriminator res2net \
    #     --log_iter 200 \
    #     --save_epoch 100 \
    #     --mae_weight ./weights/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth\
    #     --tuning \
    #     --train_path ./dataset/CrossValidation/cv${i}_TrainDataset \
    #     --val_path ./dataset/CrossValidation/cv${i}_ValDataset \
    #     --train_save ${TRAIN_SAVE} \
    #     1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1) \
    #     2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)

    for weight in ${weightArray[@]}; do

        python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
            --testsize 352 \
            --discriminator res2net \
            --pth_path ./snapshots/${TRAIN_SAVE}/Transfuse-${weight}.pth \
            --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-${weight}.pth \
            --test_data_path1 ./dataset/CrossValidation/cv${i}_TestDataset \
            --test_data_path2 ./dataset/CrossValidation/cv${i}_sekkai_TestDataset \
            --test_save_path ./results/test/${TRAIN_SAVE}/cv${i} \
            1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1) \
            2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)
        
    done

done
