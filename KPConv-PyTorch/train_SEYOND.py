import os
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.trainer import ModelTrainer
from models.architectures import KPFCNN
from datasets.Seyond_LiDAR import SeyondLiDARDataset
from datasets.COVERED import COVEREDSampler, COVEREDCollate
from train_COVERED import COVEREDConfig


# ======================================================================
#   Configuration for Seyond LiDAR fine-tuning (5 classes)
# ======================================================================
class SeyondConfig(COVEREDConfig):
    def __init__(self):
        super().__init__()

        # Dataset and output
        self.dataset = 'Seyond_LiDAR'
        self.num_classes = 5  # Floor, Wall, Column, Robo Dog, Screen+Stand
        self.data_path = 'data'
        self.saving_path = time.strftime(
            'training_logs_SEYOND_KPFCNN/Log_%Y-%m-%d_%H-%M-%S',
            time.gmtime()
        )

        # Fine-tuning parameters
        self.learning_rate = 1e-4
        self.batch_num = 1
        self.max_epoch = 50             # 30–50 epochs sufficient
        self.weight_decay = 1e-4
        self.grad_clip_norm = 10.0

        # Pretrained checkpoint (from COVERED)
        self.pretrained_path = (
            "training_logs_SEYOND_KPFCNN/Log_2025-11-04_02-22-11/checkpoints/best_chkp_adaptive.tar"
        )

        # Automatically set stable steps per epoch
        dataset_path = os.path.join(self.data_path, 'Seyond_LiDAR')
        dataset_size = len([f for f in os.listdir(dataset_path) if f.endswith('.ply')])
        self.epoch_steps = min(dataset_size, 1000)
        print(f"Epoch steps automatically set to {self.epoch_steps} (based on {dataset_size} frames)")

        # Validation and worker settings
        self.validation_size = 100
        self.input_threads = 4


# ======================================================================
#   Custom Trainer with Adaptive Learning Rate and Logging
# ======================================================================
class AdaptiveTrainer(ModelTrainer):
    def train(self, net, training_loader, val_loader, config):
        """Train with adaptive LR scheduler and logging."""
        print("\nUsing adaptive learning rate (ReduceLROnPlateau)")
        scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5,
            patience=5, verbose=True, min_lr=1e-7
        )

        # Prepare log files
        if config.saving:
            os.makedirs(config.saving_path, exist_ok=True)
            train_log = os.path.join(config.saving_path, 'training.txt')
            val_log = os.path.join(config.saving_path, 'val_IoUs.txt')
            with open(train_log, "w") as f:
                f.write("epoch step loss mIoU lr\n")
            with open(val_log, "w") as f:
                f.write("epoch mIoU\n")

        best_miou = 0.0
        for epoch in range(config.max_epoch):
            net.train()
            total_loss = 0.0

            # Training Loop
            for step, batch in enumerate(training_loader):
                if step >= config.epoch_steps:
                    break

                batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = net(batch, config)
                loss = net.loss(outputs, batch.labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if (step + 1) % 100 == 0:
                    print(f"Epoch {epoch+1} [{step+1}/{config.epoch_steps}] | Loss: {loss.item():.4f}")

            # Validation
            net.eval()
            with torch.no_grad():
                mIoU = self.validation(net, val_loader, config)

            scheduler.step(mIoU)
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{config.max_epoch} — Mean IoU: {mIoU:.2f}% | LR: {lr:.2e}")

            # Logging
            if config.saving:
                with open(train_log, "a") as f:
                    f.write(f"{epoch+1} {config.epoch_steps} {total_loss/config.epoch_steps:.4f} {mIoU:.3f} {lr:.6f}\n")
                with open(val_log, "a") as f:
                    f.write(f"{epoch+1} {mIoU:.3f}\n")

            # Save best checkpoint
            if mIoU > best_miou:
                best_miou = mIoU
                checkpoint_dir = os.path.join(config.saving_path, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(checkpoint_dir, "best_chkp_adaptive.tar"))
                print(f"New best mIoU: {best_miou:.2f}% — model saved.")

        print("\nTraining finished.")
        print(f"Best mIoU achieved: {best_miou:.2f}%")


# ======================================================================
#   Main Fine-Tuning Procedure
# ======================================================================
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print("\nData Preparation")
    print("****************")

    config = SeyondConfig()

    # Initialize datasets
    train_ds = SeyondLiDARDataset(config, set='training', use_potentials=True)
    val_ds = SeyondLiDARDataset(config, set='validation', use_potentials=True)

    # Automatically adjust validation size
    config.validation_size = len(val_ds.files)
    print(f"Validation size automatically set to {config.validation_size} frames")

    # Initialize loaders
    train_loader = DataLoader(
        train_ds, batch_size=1, collate_fn=COVEREDCollate,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, collate_fn=COVEREDCollate,
        num_workers=0, pin_memory=False
    )

    print("\nModel Preparation")
    print("*****************")

    # Load model and COVERED checkpoint (excluding old head)
    net = KPFCNN(config, train_ds.label_values, train_ds.ignored_labels)
    checkpoint = torch.load(config.pretrained_path, map_location='cpu')
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = net.state_dict()

    # Exclude old classification head weights
    filtered = {k: v for k, v in pretrained_dict.items() if not k.startswith("head")}
    model_dict.update(filtered)
    net.load_state_dict(model_dict, strict=False)
    print(f"Loaded COVERED encoder/decoder weights from: {config.pretrained_path}")

    # Freeze encoder, fine-tune decoder + head
    for name, param in net.named_parameters():
        if "decoder" not in name and "head" not in name:
            param.requires_grad = False
    print("Frozen encoder layers → fine-tuning decoder and head only.")

    # Move to GPU
    net.cuda()

    print("\nStart Fine-Tuning on Seyond LiDAR with Adaptive LR")
    print("****************************************************")

    trainer = AdaptiveTrainer(net, config)
    trainer.train(net, train_loader, val_loader, config)
