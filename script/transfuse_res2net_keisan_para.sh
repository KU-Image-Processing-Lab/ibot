# 環境変数
export PYTHONPATH=/home/student/git/medical-image-processing:$PYTHONPATH
# --------------------------------------------------------------
LF='
'
SCRIPT_DIR='trans-mtl-para'
TEST_SCRIPT_NAME='test_trans_caranet_nash.py'
TEST_SCRIPT_NAME2='test_trans_nash.py'

mkdir -p ./logs
# --------------------------------------------------------------
TRAIN_SAVE="transfuse_res2net_cv0"
python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/Transfuse-best.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-best.pth \
    --test_data_path1 ./dataset/CrossValidation/cv0_TestDataset \
    --test_data_path2 ./dataset/CrossValidation/cv0_sekkai_TestDataset \
    --test_save_path ./results/test/${TRAIN_SAVE}/cv0 \

# --------------------------------------------------------------
TRAIN_SAVE="transfuse_resnet10/1st/0"
python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/Transfuse-best.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-best.pth \
    --test_data_path1 ./dataset/CrossValidation/cv0_TestDataset \
    --test_data_path2 ./dataset/CrossValidation/cv0_sekkai_TestDataset \
    --test_save_path ./results/test/${TRAIN_SAVE}/cv0 \

# --------------------------------------------------------------
TRAIN_SAVE="transfuse_res2net_cv0"
python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME2} \
    --testsize 352 \
    --discriminator res2net \
    --pth_path ./snapshots/${TRAIN_SAVE}/Transfuse-best.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-best.pth \
    --test_data_path1 ./dataset/CrossValidation/cv0_TestDataset \
    --test_data_path2 ./dataset/CrossValidation/cv0_sekkai_TestDataset \
    --test_save_path ./results/test/${TRAIN_SAVE}/cv0 \

# --------------------------------------------------------------
TRAIN_SAVE="transfuse_resnet10/1st/0"
python3 ./src/${SCRIPT_DIR}/${TEST_SCRIPT_NAME2} \
    --testsize 352 \
    --discriminator resnet10 \
    --pth_path ./snapshots/${TRAIN_SAVE}/Transfuse-best.pth \
    --pth_path2 ./snapshots/${TRAIN_SAVE}/Discriminator-best.pth \
    --test_data_path1 ./dataset/CrossValidation/cv0_TestDataset \
    --test_data_path2 ./dataset/CrossValidation/cv0_sekkai_TestDataset \
    --test_save_path ./results/test/${TRAIN_SAVE}/cv0 \