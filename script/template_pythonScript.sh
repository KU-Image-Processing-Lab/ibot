# 環境変数
export PYTHONPATH=path_to_project_root_directory:$PYTHONPATH

# --------------------------------------------------------------

SCRIPT_DIR='path_to_script_directory'
SCRIPT_NAME='path_to_save_directory'

python3 ./src/${SCRIPT_DIR}/${SCRIPT_NAME} 1> >(tee ./logs/${SCRIPT_NAME}_stdout.log >&1 ) 2> >(tee ./logs/${SCRIPT_NAME}_stderr.log >&2)
