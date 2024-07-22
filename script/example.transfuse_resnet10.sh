# 環境変数
export PYTHONPATH=/home/student/Git/medical-image-processing:$PYTHONPATH

# --------------------------------------------------------------

SCRIPT_DIR='transfuse-mtl'
TRAIN_SCRIPT_NAME='train_trans_nash.py'
TEST_SCRIPT_NAME='test_trans_nash.py'
TRAIN_SAVE="transfuse_resnet10"

sudo mkdir -p ./logs

# --------------------------------------------------------------

python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
    --epoch 200 \
    --batchsize 4 \
    --trainsize 352 \
    --train_path ./dataset/CrossValidation/cv0_TrainDataset\
    --val_path ./dataset/CrossValidation/cv0_ValDataset\
    --seg_loss_path ./fig/0/seg_loss_img.png\
    --dis_loss_path ./fig/0/dis_loss_img.png\
    --tuning False\
    --discriminator resnet10 \
    --log_epoch 50 \
    --save_epoch 50 \
    --train_save ${TRAIN_SAVE}/2/0 \
    --mae_weight ./weights/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth\
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
    --epoch 200 \
    --batchsize 4 \
    --trainsize 352 \
    --train_path ./dataset/CrossValidation/cv1_TrainDataset\
    --val_path ./dataset/CrossValidation/cv1_ValDataset\
    --seg_loss_path ./fig/1/seg_loss_img.png\
    --dis_loss_path ./fig/1/dis_loss_img.png\
    --tuning False\
    --discriminator resnet10 \
    --log_epoch 50 \
    --save_epoch 50 \
    --train_save ${TRAIN_SAVE}/2/1 \
    --mae_weight ./weights/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth\
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
    --epoch 200 \
    --batchsize 4 \
    --trainsize 352 \
    --train_path ./dataset/CrossValidation/cv2_TrainDataset\
    --val_path ./dataset/CrossValidation/cv2_ValDataset\
    --seg_loss_path ./fig/2/seg_loss_img.png\
    --dis_loss_path ./fig/2/dis_loss_img.png\
    --tuning False\
    --discriminator resnet10 \
    --log_epoch 50 \
    --save_epoch 50 \
    --train_save ${TRAIN_SAVE}/2/2 \
    --mae_weight ./weights/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth\
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
    --epoch 200 \
    --batchsize 4 \
    --trainsize 352 \
    --train_path ./dataset/CrossValidation/cv3_TrainDataset\
    --val_path ./dataset/CrossValidation/cv3_ValDataset\
    --seg_loss_path ./fig/3/seg_loss_img.png\
    --dis_loss_path ./fig/3/dis_loss_img.png\
    --tuning False\
    --discriminator resnet10 \
    --log_epoch 50 \
    --save_epoch 50 \
    --train_save ${TRAIN_SAVE}/2/3 \
    --mae_weight ./weights/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth\
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME} \
    --epoch 200 \
    --batchsize 4 \
    --trainsize 352 \
    --train_path ./dataset/CrossValidation/cv4_TrainDataset\
    --val_path ./dataset/CrossValidation/cv4_ValDataset\
    --seg_loss_path ./fig/4/seg_loss_img.png\
    --dis_loss_path ./fig/4/dis_loss_img.png\
    --tuning False\
    --discriminator resnet10 \
    --log_epoch 50 \
    --save_epoch 50 \
    --train_save ${TRAIN_SAVE}/2/4 \
    --mae_weight ./weights/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth\
    1> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TRAIN_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/0/Transfuse-best.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/0/Discriminator-best.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/1/Transfuse-best.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/1/Discriminator-best.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/2/Transfuse-best.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/2/Discriminator-best.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/3/Transfuse-best.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/3/Discriminator-best.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/4/Transfuse-best.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/4/Discriminator-best.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/0/Transfuse-best2.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/0/Discriminator-best2.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/1/Transfuse-best2.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/1/Discriminator-best2.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/2/Transfuse-best2.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/2/Discriminator-best2.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/3/Transfuse-best2.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/3/Discriminator-best2.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/4/Transfuse-best2.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/4/Discriminator-best2.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/0/Transfuse-99.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/0/Discriminator-99.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/1/Transfuse-99.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/1/Discriminator-99.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/2/Transfuse-99.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/2/Discriminator-99.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/3/Transfuse-99.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/3/Discriminator-99.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/4/Transfuse-99.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/4/Discriminator-99.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/0/Transfuse-199.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/0/Discriminator-199.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/1/Transfuse-199.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/1/Discriminator-199.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/2/Transfuse-199.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/2/Discriminator-199.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/3/Transfuse-199.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/3/Discriminator-199.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)

python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/2/4/Transfuse-199.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/2/4/Discriminator-199.pth \
    1> >(tee ./logs/${TEST_SCRIPT_NAME}_stdout.log >&1 ) \
    2> >(tee ./logs/${TEST_SCRIPT_NAME}_stderr.log >&2)
