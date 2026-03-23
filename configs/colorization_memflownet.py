# configs/colorization_memflownet.py

from yacs.config import CfgNode as CN

_CN = CN()

_CN.name = ''
_CN.suffix = 'colorization_memflownet'
_CN.eval_only = False
_CN.log_dir = './logs' 
_CN.gamma = 0.8

_CN.batch_size = 8
_CN.sum_freq = 100
_CN.val_freq = 5000
_CN.image_size = [224, 224]
_CN.data_path = '/media/SSD2/m11302113/data-vc-2023-maybex2-final/images/images,/media/SSD2/m11302113/ntire-dataset/ntire_videos,/media/SSD2/m11302113/davis-dataset-vc/davis_videos,/media/SSD2/m11302113/reds-optflow/reds'
_CN.critical_params = []

_CN.network = 'MemFlowNet_skflow'

_CN.restore_ckpt = None
_CN.restore_steps = 0
_CN.mixed_precision = True

###############################################
# Memory參數
_CN.input_frames = 3
_CN.num_ref_frames = 2
_CN.train_avg_length = (224 * 224 // 64) * 3 / 2

################################################
# Network
_CN.MemFlowNet_skflow = CN()
_CN.MemFlowNet_skflow.pretrain = True #False
_CN.MemFlowNet_skflow.freeze_encoder = True  # freeze encoder weights (only train projection heads)
_CN.MemFlowNet_skflow.cnet = 'swinv2'  # 'twins' | 'swinv2' | 'basicencoder'
_CN.MemFlowNet_skflow.fnet = 'swinv2'  # 'twins' | 'swinv2' | 'basicencoder'
_CN.MemFlowNet_skflow.gma = 'GMA-SK2'
_CN.MemFlowNet_skflow.down_ratio = 8
_CN.MemFlowNet_skflow.feat_dim = 256
_CN.MemFlowNet_skflow.corr_fn = 'default'
_CN.MemFlowNet_skflow.corr_levels = 4
_CN.MemFlowNet_skflow.decoder_depth = 9
_CN.MemFlowNet_skflow.critical_params = ["cnet", "fnet", "pretrain", "corr_levels", "decoder_depth", "train_avg_length"]
_CN.MemFlowNet_skflow.train_avg_length = _CN.train_avg_length

### Trainer
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'
_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 1.75e-4 #1.75e-4
_CN.trainer.adamw_decay = 1e-4
_CN.trainer.clip = 10.0 #1.0
_CN.trainer.num_steps = 80000 #50000
_CN.trainer.epsilon = 1e-7 #1e-8
_CN.trainer.anneal_strategy = 'linear'

def get_cfg():
    return _CN.clone()


