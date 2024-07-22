#!/bin/bash

ARRAY=(1 2 3 4 5)
output_name="test224"
method_name="lvat"
N="3000"
filename="${output_name}_${method_name}_ssl${N}.txt"
echo $filename
echo "----------start----------" > $filename

for num in ${ARRAY[@]}; do
  echo $num
  echo "" >> $filename
  echo "-----------${num}-----------" >> $filename
  python train_trans_lvat_ssl.py --epoch 60 --batchsize 16 --trainsize 224 --train_path './dataset/sekkai_TrainDataset528_unity' --nonlabel_path './DISCRI/dataset_ssl/sd3000_unity'
  python -u test_transT.py --testsize 224 --pth_path './snapshots/TransFuse_S/TransFuse-59.pth' --data_path './dataset/sekkai_TestDataset_unity' --dice 1 >> $filename
  python -u test_transT.py --testsize 224 --pth_path './snapshots/TransFuse_S/TransFuse-best.pth' --data_path './dataset/sekkai_TestDataset_unity' --dice 1 >> $filename
  python save_weight_file.py --file_name "transfuse_${method_name}_224_ssl${N}" --attempt $num --train_path './dataset/sekkai_TrainDataset528_unity'
done

python make_result.py --file_name "${filename}" --read_line 5
