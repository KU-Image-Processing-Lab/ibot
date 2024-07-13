# 環境変数
PYTHONPATH=path_to_project_root_directory:$PYTHONPATH

# --------------------------------------------------------------

python3 path_to_python_script 1> >(tee ./logs/train_trans_caranet_nash_stdout.log >&1 ) 2> >(tee ./logs/train_trans_caranet_nash_stderr.log >&2)
