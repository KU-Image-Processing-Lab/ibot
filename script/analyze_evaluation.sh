
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

python3 ./utils/analyze_prediction.py \
  --filepath results/transfuse_resnet10_preprocess_cv0_2024-10-28/2024-10-29_02-49-15_evaluation.csv \
  --save_root_path ./results/transfuse_resnet10_preprocess_cv0_2024-10-28

python3 ./utils/analyze_prediction.py \
  --filepath results/Transfuse_S/2024-10-14_23-31-35_evaluation.csv \
  --save_root_path ./results/Transfuse_S
