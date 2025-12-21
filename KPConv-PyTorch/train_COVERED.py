#
#   KPConv fine-tuning script for COVERED dataset
#
#   Author: Vinit Pimpale
#   Date: October 2025
#

import os
import time
import torch
from torch.utils.data import DataLoader

from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPFCNN
from datasets.COVERED import COVEREDDataset, COVEREDSampler, COVEREDCollate


# ==========================================================================================================
# Configuration
# ==========================================================================================================
class COVEREDConfig(Config):
    """Configuration for fine-tuning KPConv on COVERED dataset."""

    def __init__(self):
        super().__init__()

        # --------------------
        # Dataset parameters
        # --------------------
        self.dataset = 'COVERED'
        self.dataset_task = 'cloud_segmentation'
        self.data_path = 'data' 
        self.input_threads = 8

        # --------------------
        # Architecture
        # --------------------
        self.architecture = [
            'simple',
            'resnetb',
            'resnetb_strided',
            'resnetb',
            'resnetb',
            'resnetb_strided',
            'resnetb',
            'resnetb',
            'resnetb_strided',
            'resnetb_deformable',
            'resnetb_deformable',
            'resnetb_deformable_strided',
            'resnetb_deformable',
            'resnetb_deformable',
            'nearest_upsample',
            'unary',
            'nearest_upsample',
            'unary',
            'nearest_upsample',
            'unary',
            'nearest_upsample',
            'unary'
        ]

        # --------------------
        # KPConv parameters
        # --------------------
        self.num_kernel_points = 15
        self.in_radius = 0.6               
        self.first_subsampling_dl = 0.07    
        self.conv_radius = 2.5
        self.deform_radius = 5.0
        self.KP_extent = 1.2
        self.KP_influence = 'linear'
        self.aggregation_mode = 'sum'
        self.fixed_kernel_points = 'center'
        self.modulated = False


        # --------------------
        # Feature dimensions
        # --------------------
        self.first_features_dim = 128
        self.in_features_dim = 5            # we use RGB only from COVERED .ply
        self.use_batch_norm = True
        self.batch_norm_momentum = 0.02

        # --------------------
        # Deformation fitting
        # --------------------
        self.deform_fitting_mode = 'point2point'
        self.deform_fitting_power = 1.0
        self.deform_lr_factor = 0.1
        self.repulse_extent = 1.2

        # --------------------
        # Training parameters
        # --------------------
        self.learning_rate = 1e-4
        self.momentum = 0.98
        self.weight_decay = 1e-4
        self.grad_clip_norm = 10.0
        self.max_epoch = 200
        self.batch_num = 1
        self.checkpoint_gap = 10

        # --------------------
        # Augmentations
        # --------------------
        self.augment_scale_anisotropic = True
        self.augment_rotation = 'vertical'
        self.augment_scale_min = 0.9
        self.augment_scale_max = 1.1
        self.augment_noise = 0.001
        self.augment_color = 0.8

        # --------------------
        # Saving
        # --------------------
        self.saving = True
        self.saving_path = time.strftime(
            'training_logs_COVERED_KPFCNN/Log_%Y-%m-%d_%H-%M-%S', time.gmtime()
        )

        # --------------------
        # Pretrained checkpoint path
        # --------------------
        self.pretrained_path = "training_logs_S3DIS_KPFCNN/checkpoints/chkp_0200.tar"


# ==========================================================================================================
# Main entry
# ==========================================================================================================
if __name__ == '__main__':
    # Select GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    previous_training_path = 'training_logs_COVERED_KPFCNN/Log_2025-10-27_22-28-06'

    print("\nData Preparation")
    print("****************")

    # Global config (used by COVEREDCollate)
    global config
    config = COVEREDConfig()
    config.num_classes = 6

    # Datasets
    training_dataset = COVEREDDataset(config, set='training', use_potentials=True)
    test_dataset = COVEREDDataset(config, set='validation', use_potentials=True)

    # Samplers
    training_sampler = COVEREDSampler(training_dataset)
    test_sampler = COVEREDSampler(test_dataset)

    # Dataloaders (Jetson-safe)
    training_loader = DataLoader(
        training_dataset,
        batch_size=1,
        sampler=training_sampler,
        collate_fn=COVEREDCollate,
        num_workers=0,          
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,
        collate_fn=COVEREDCollate,
        num_workers=0,
        pin_memory=False,
    )

    print("\nModel Preparation")
    print("*****************")

    # Initialize KPConv model
    net = KPFCNN(config, training_dataset.label_values, training_dataset.ignored_labels)

    chosen_chkp = None
    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join(previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f.startswith('chkp')]

        # Find which snapshot to restore
        if 'current_chkp.tar' in chkps:
            chosen_chkp = os.path.join(chkp_path, 'current_chkp.tar')
        elif chkps:
            # Find latest chkp if current_chkp.tar is missing
            chkps.sort()
            chosen_chkp = os.path.join(chkp_path, chkps[-1])
        
        if chosen_chkp:
            print(f"Found checkpoint to resume from: {chosen_chkp}")
            # Load the config from the previous training
            config.load(previous_training_path)
            config.saving_path = previous_training_path # Ensure it saves to the same folder
            #train_COVERED.config = config # Re-patch the global config
        else:
            print(f"Could not find checkpoint in {previous_training_path}. Starting new training.")
            previous_training_path = '' # Reset to treat as new training



    # Load pretrained S3DIS weights (excluding head)
    if not chosen_chkp and os.path.exists(config.pretrained_path):
        checkpoint = torch.load(config.pretrained_path, map_location='cpu', weights_only=True)
        pretrained_dict = (
            checkpoint["model_state_dict"]
            if "model_state_dict" in checkpoint
            else checkpoint
        )
        model_dict = net.state_dict()
        filtered = {k: v for k, v in pretrained_dict.items() if not k.startswith("head_softmax")}
        model_dict.update(filtered)
        net.load_state_dict(model_dict, strict=False)
        print(f" Loaded pretrained weights (excluding head) from: {config.pretrained_path}")
    elif not chosen_chkp:
        print(" Pretrained checkpoint not found. Training from scratch.")
    else:
        print(" Resuming training. S3DIS weights will be loaded from checkpoint.")
    

    # Freeze encoder layers for fine-tuning
    for name, param in net.named_parameters():
        if "decoder" not in name and "head" not in name:
            param.requires_grad = False
    print("Frozen encoder layers → fine-tuning decoder and head only.")

    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    print("\nStart Training")
    print("**************")

    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    trainer.train(net, training_loader, test_loader, config)
