import torch
from utils.config import Config
from models.architectures import KPFCNN

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():

    # -------------------------------------------------------------
    # Path to the SEYOND training log directory (NOT parameters.txt)
    # -------------------------------------------------------------
    log_dir = "training_logs_SEYOND_KPFCNN/Log_2025-11-04_02-22-11"

    print("\nLoading configuration from:", log_dir)

    # Load config
    config = Config()
    config.load(log_dir)

    # Number of inputs (SEYOND uses RGB + 2 dummy → 5 channels)
    num_inputs = getattr(config, "num_inputs", 5)

    # Class names used during training
    label_values = ['Floor', 'Wall', 'Column', 'Robo Dog', 'Screen+Stand']
    ignored_labels = []

    # Build the KPConv model
    print("\nBuilding KPConv model...")
    net = KPFCNN(config, label_values, ignored_labels)

    # Count parameters
    total_params, trainable_params = count_parameters(net)

    print("\n======= KPConv Parameter Count =======")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("======================================\n")


if __name__ == "__main__":
    main()
