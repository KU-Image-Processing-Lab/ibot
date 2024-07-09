"""
Our experimental codes are based on 
https://github.com/McGregorWwww/UCTransNet
We thankfully acknowledge the contributions of the authors
"""

import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # change this as needed

use_cuda = torch.cuda.is_available()
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)

n_filts = 32            # change this to train larger ACC-UNet model
cosineLR = True         # whether use cosineLR or not
n_channels = 1
n_labels = 1
epochs = 500
img_size = 352
print_frequency = 1
save_frequency = 100
vis_frequency = 10
early_stopping_patience = 250

pretrain = False


# task_name = 'GlaS_exp1'
#task_name = 'ISIC18_exp1'
#task_name = 'Clinic_exp1'
#task_name = 'Covid_exp1'
#task_name = 'BUSI_exp1'


learning_rate = 1e-4
batch_size = 2

model_name = 'ACC_UNet'
#model_name = 'SwinUnet'
#model_name = 'SMESwinUnet'
#model_name = 'UCTransNet'
#model_name = 'UNet_base'
#model_name = 'MultiResUnet1_32_1.67'



test_session = "session"         #


train_dataset = './dataset/sekkai_TrainDataset'
val_dataset = './dataset/sekkai_ValDataset'
test_dataset = './dataset/sekkai_TestDataset'
session_name       = 'session'  #time.strftime('%m.%d_%Hh%M')
save_path          = './snapshots/ACCUnet/e-4'
model_path         = save_path + '/models/'
tensorboard_folder = save_path + '/tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + '/visualize_val/'




##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config




# used in testing phase, copy the session name in training phase
