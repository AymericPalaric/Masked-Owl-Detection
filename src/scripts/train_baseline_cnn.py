from ..models.efficientnet import EfficientNet
from ..models.baseline_cnn.model import Baseline
from ..models.baseline_cnn import training
from ..utils import torch_utils, transform_utils
import matplotlib.pyplot as plt
import torchvision
import torch
nn = torch.nn


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model
    model = EfficientNet().to(device)

    # Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-6)

    # Training parameters
    epochs = 10
    batch_size = 64

    # Loader parameters
    n_workers = 4

    # Transforms
    reshape_size = (129, 129)
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
                   f"trained_models/efficientnet_model_{epoch}.pt")

    # Plotting
    plt.figure()
    plt.plot(train_losses, label="Train loss")
    plt.plot(test_losses, label="Test loss")
    plt.legend()
    plt.savefig("trained_models/efficientnet_loss.png")
    plt.show()
