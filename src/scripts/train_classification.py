from src.models.efficientnet import EfficientNet
from src.models.baseline_cnn import Baseline
from src.training import training
from src.utils import torch_utils, transform_utils
import matplotlib.pyplot as plt
import torchvision
import torch
import json
import argparse
nn = torch.nn
"""
Script to train an implemented model.
Arguments are:
    - epochs;
    - batch_size;
    - n_workers;
    - reshape_size;
    - model_type;
    - model_name;
    - lr;
    - model_args.
"""


# Models available
CLASS_MODELS = {'efficientnet': EfficientNet, 'baseline_cnn': Baseline}

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--reshape_size", type=tuple, default=(129, 129))
    parser.add_argument("--model_type", type=str, default="efficientnet")
    parser.add_argument("--model_name", type=str, default="efficientnet-b0")
    parser.add_argument("--lr", type=float, default=1e-5)
    # In windows, add: --model_args '{"key_1": "value_1", "key_2": "value_2"}'
    parser.add_argument("--model_args", required=False, type=json.loads,
                        help="Model kwargs", default='{}')

    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    n_workers = args.n_workers
    reshape_size = args.reshape_size
    model_type = args.model_type
    model_name = args.model_name
    model_args = args.model_args
    lr = args.lr

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model
    model = CLASS_MODELS[model_type](**model_args).to(device)

    # Optimizer
    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training parameters
    audio_transform = transform_utils.baseline_transform

    image_transform_no_standardization = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Resize(reshape_size)])
    mean, std = torch_utils.compute_mean_std_classif(
        audio_transform=audio_transform, image_transform=image_transform_no_standardization, batch_size=batch_size, num_workers=n_workers)
    print(f"Mean dataset: {mean}, std dataset: {std}")
    image_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(
    ), torchvision.transforms.Resize(reshape_size), torchvision.transforms.Normalize(mean=mean, std=std)])

    # Datasets
    train_dataset = torch_utils.create_classif_dataset(
        audio_transform, image_transform, train_test=True)
    test_dataset = torch_utils.create_classif_dataset(
        audio_transform, image_transform, train_test=False)

    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   drop_last=True, num_workers=n_workers, pin_memory=True if torch.cuda.is_available() else False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                  drop_last=False, num_workers=n_workers, pin_memory=True if torch.cuda.is_available() else False)

    print(
        f"Training on {len(train_dataset)} samples, testing on {len(test_dataset)} samples")
    print(
        f"Training on {len(train_dataloader)} batches, testing on {len(test_dataloader)} batches")

    # Losses
    train_losses = list()
    test_losses = list()

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        print("-"*20)

        train_loss = training.train_loop(
            model, train_dataloader, optimizer, loss_fn, device)
        test_loss = training.test_loop(test_dataloader, model, loss_fn, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        torch.save(model.state_dict(),
                   f"trained_models/{model_name}_{epoch}.pt")

    # Plotting
    plt.figure()
    plt.plot(train_losses, label="Train loss")
    plt.plot(test_losses, label="Test loss")
    plt.legend()
    plt.savefig(f"trained_models/{model_name}_loss.png")
    plt.show()
