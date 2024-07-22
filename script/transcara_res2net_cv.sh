# 環境変数
export PYTHONPATH=/home/student/Git/medical-image-processing:$PYTHONPATH

# --------------------------------------------------------------

SCRIPT_DIR='transcaranet-mtl'
TRAIN_SCRIPT_NAME='train_trans_caranet_nash.py'
TEST_SCRIPT_NAME='test_trans_caranet_nash.py'
TRAIN_SAVE="transcara_res2net"

sudo mkdir -p ./logs

# --------------------------------------------------------------

python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
    --epoch 200 \
    --batchsize 4 \
    --trainsize 352 \
    --train_path ./dataset/CrossValidation/cv0_TrainDataset\
    --val_path ./dataset/CrossValidation/cv0_ValDataset\
    --loss_dir ./fig/cara/0/\
    --seg_loss_path ./fig/cara/0/seg_loss_img.png\
    --dis_loss_path ./fig/cara/0/dis_loss_img.png\
    --discriminator res2net \
    --log_iter 500 \
    --save_epoch 100 \
    --train_save ${TRAIN_SAVE}/cara/0 \
    --mae_weight ./weights/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth\
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
    --epoch 200 \
    --batchsize 4 \
    --trainsize 352 \
    --train_path ./dataset/CrossValidation/cv1_TrainDataset\
    --val_path ./dataset/CrossValidation/cv1_ValDataset\
    --seg_loss_path ./fig/cara/1/seg_loss_img.png\
    --dis_loss_path ./fig/cara/1/dis_loss_img.png\
    --tuning False\
    --discriminator res2net \
    --log_epoch 50 \
    --save_epoch 50 \
    --train_save ${TRAIN_SAVE}/cara/1 \
    --mae_weight ./weights/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth\
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
    --epoch 200 \
    --batchsize 4 \
    --trainsize 352 \
    --train_path ./dataset/CrossValidation/cv2_TrainDataset\
    --val_path ./dataset/CrossValidation/cv2_ValDataset\
    --seg_loss_path ./fig/cara/2/seg_loss_img.png\
    --dis_loss_path ./fig/cara/2/dis_loss_img.png\
    --tuning False\
    --discriminator res2net \
    --log_epoch 50 \
    --save_epoch 50 \
    --train_save ${TRAIN_SAVE}/cara/2 \
    --mae_weight ./weights/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth\
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
    --epoch 200 \
    --batchsize 4 \
    --trainsize 352 \
    --train_path ./dataset/CrossValidation/cv3_TrainDataset\
    --val_path ./dataset/CrossValidation/cv3_ValDataset\
    --seg_loss_path ./fig/cara/3/seg_loss_img.png\
    --dis_loss_path ./fig/cara/3/dis_loss_img.png\
    --tuning False\
    --discriminator res2net \
    --log_epoch 50 \
    --save_epoch 50 \
    --train_save ${TRAIN_SAVE}/cara/3 \
    --mae_weight ./weights/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth\
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
    --epoch 200 \
    --batchsize 4 \
    --trainsize 352 \
    --train_path ./dataset/CrossValidation/cv4_TrainDataset\
    --val_path ./dataset/CrossValidation/cv4_ValDataset\
    --seg_loss_path ./fig/cara/4/seg_loss_img.png\
    --dis_loss_path ./fig/cara/4/dis_loss_img.png\
    --tuning False\
    --discriminator res2net \
    --log_epoch 50 \
    --save_epoch 50 \
    --train_save ${TRAIN_SAVE}/cara/4 \
    --mae_weight ./weights/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth\
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/cara/0/Transfuse-best.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/0/Discriminator-best.pth \
    --test_data_path1 ./dataset/CrossValidation/cv0_TestDataset\
    --test_data_path2 ./dataset/CrossValidation/cv0_sekkai_TestDataset\
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

# python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
#     --testsize 352 \
#     --discriminator res2net \
#     --pth_path ./snapshots/${TRAIN_SAVE}/cara/1/Transfuse-best.pth \
#     --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/1/Discriminator-best.pth \
#     --test_data_path1 ./dataset/CrossValidation/cv1_TestDataset\
#     --test_data_path2 ./dataset/CrossValidation/cv1_sekkai_TestDataset\
#     1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
#     2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

# python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
#     --testsize 352 \
#     --discriminator res2net \
#     --pth_path ./snapshots/${TRAIN_SAVE}/cara/2/Transfuse-best.pth \
#     --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/2/Discriminator-best.pth \
#     --test_data_path1 ./dataset/CrossValidation/cv2_TestDataset\
#     --test_data_path2 ./dataset/CrossValidation/cv2_sekkai_TestDataset\
#     1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
#     2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

# python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
#     --testsize 352 \
#     --discriminator res2net \
#     --pth_path ./snapshots/${TRAIN_SAVE}/cara/3/Transfuse-best.pth \
#     --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/3/Discriminator-best.pth \
#     --test_data_path1 ./dataset/CrossValidation/cv3_TestDataset\
#     --test_data_path2 ./dataset/CrossValidation/cv3_sekkai_TestDataset\
#     1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
#     2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

# python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
#     --testsize 352 \
#     --discriminator res2net \
#     --pth_path ./snapshots/${TRAIN_SAVE}/cara/4/Transfuse-best.pth \
#     --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/4/Discriminator-best.pth \
#     --test_data_path1 ./dataset/CrossValidation/cv4_TestDataset\
#     --test_data_path2 ./dataset/CrossValidation/cv4_sekkai_TestDataset\
#     1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
#     2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/cara/0/Transfuse-best2.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/0/Discriminator-best2.pth \
    --test_data_path1 ./dataset/CrossValidation/cv0_TestDataset\
    --test_data_path2 ./dataset/CrossValidation/cv0_sekkai_TestDataset\
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/cara/1/Transfuse-best2.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/1/Discriminator-best2.pth \
    --test_data_path1 ./dataset/CrossValidation/cv1_TestDataset\
    --test_data_path2 ./dataset/CrossValidation/cv1_sekkai_TestDataset\
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/cara/2/Transfuse-best2.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/2/Discriminator-best2.pth \
    --test_data_path1 ./dataset/CrossValidation/cv2_TestDataset\
    --test_data_path2 ./dataset/CrossValidation/cv2_sekkai_TestDataset\
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/cara/3/Transfuse-best2.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/3/Discriminator-best2.pth \
    --test_data_path1 ./dataset/CrossValidation/cv3_TestDataset\
    --test_data_path2 ./dataset/CrossValidation/cv3_sekkai_TestDataset\
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/cara/4/Transfuse-best2.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/4/Discriminator-best2.pth \
    --test_data_path1 ./dataset/CrossValidation/cv4_TestDataset\
    --test_data_path2 ./dataset/CrossValidation/cv4_sekkai_TestDataset\
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/cara/0/Transfuse-99.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/0/Discriminator-99.pth \
    --test_data_path1 ./dataset/CrossValidation/cv0_TestDataset\
    --test_data_path2 ./dataset/CrossValidation/cv0_sekkai_TestDataset\
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/cara/1/Transfuse-99.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/1/Discriminator-99.pth \
    --test_data_path1 ./dataset/CrossValidation/cv1_TestDataset\
    --test_data_path2 ./dataset/CrossValidation/cv1_sekkai_TestDataset\
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/cara/2/Transfuse-99.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/2/Discriminator-99.pth \
    --test_data_path1 ./dataset/CrossValidation/cv2_TestDataset\
    --test_data_path2 ./dataset/CrossValidation/cv2_sekkai_TestDataset\
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/cara/3/Transfuse-99.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/3/Discriminator-99.pth \
    --test_data_path1 ./dataset/CrossValidation/cv3_TestDataset\
    --test_data_path2 ./dataset/CrossValidation/cv3_sekkai_TestDataset\
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/cara/4/Transfuse-99.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/4/Discriminator-99.pth \
    --test_data_path1 ./dataset/CrossValidation/cv4_TestDataset\
    --test_data_path2 ./dataset/CrossValidation/cv4_sekkai_TestDataset\
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/cara/0/Transfuse-199.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/0/Discriminator-199.pth \
    --test_data_path1 ./dataset/CrossValidation/cv0_TestDataset\
    --test_data_path2 ./dataset/CrossValidation/cv0_sekkai_TestDataset\
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/cara/1/Transfuse-199.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/1/Discriminator-199.pth \
    --test_data_path1 ./dataset/CrossValidation/cv1_TestDataset\
    --test_data_path2 ./dataset/CrossValidation/cv1_sekkai_TestDataset\
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/cara/2/Transfuse-199.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/2/Discriminator-199.pth \
    --test_data_path1 ./dataset/CrossValidation/cv2_TestDataset\
    --test_data_path2 ./dataset/CrossValidation/cv2_sekkai_TestDataset\
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/cara/3/Transfuse-199.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/3/Discriminator-199.pth \
    --test_data_path1 ./dataset/CrossValidation/cv3_TestDataset\
    --test_data_path2 ./dataset/CrossValidation/cv3_sekkai_TestDataset\
    # --test_save_path ./results/Transfuse_10/199/3\
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/cara/4/Transfuse-199.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/cara/4/Discriminator-199.pth \
    --test_data_path1 ./dataset/CrossValidation/cv4_TestDataset\
    --test_data_path2 ./dataset/CrossValidation/cv4_sekkai_TestDataset\
    # --test_save_path ./results/Transfuse_10/199/4\
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)
