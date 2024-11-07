
# 環境変数
export PYTHONPATH=/home/student/git/medical-image-processing:$PYTHONPATH

# --------------------------------------------------------------
LF='
'
SCRIPT_DIR='transfuse-mtl-preprocessing'
TRAIN_SCRIPT_NAME='train_trans_nash.py'
TEST_SCRIPT_NAME='test_trans_nash.py'

mkdir -p ./logs

# --------------------------------------------------------------

python3 ./utils/create_dataset.py --data_type mae