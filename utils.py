import torch
import torchvision
import random
import torch
import torch.nn as nn
import numpy as np
from couette_solver import my3DCouetteSolver
from dataset import MyFlowDataset
from torch.utils.data import DataLoader


class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(batch_size, num_workers, pin_memory, couette_dim):
    # Consider that the couette solver now requires a desired_timesteps
    # parameter for improved reusabilty
    sigma = 0.3
    my_couette_data = my3DCouetteSolver(
        desired_timesteps=1000, vertical_resolution=couette_dim, sigma=sigma)
    print(f'Noise level: {sigma}.')
    # print(f'Shape of initial 3DCouetteData: {my_couette_data.shape}')
    my_images = my_couette_data[:-1]
    my_masks = my_couette_data[1:]
    my_zip = list(zip(my_images, my_masks))
    random.shuffle(my_zip)
    my_shuffled_images, my_shuffled_masks = zip(*my_zip)
    total_images = len(my_shuffled_images)
    # Implement a 90/10/10:train/dev/test split:
    # Consider that the couette solver yields 1100 - 1 timesteps
    number_train = int(0.8*total_images)
    number_val = int(0.9*total_images)
    my_train_images = my_shuffled_images[:number_train]
    # print(type(my_train_images))
    # print(my_train_images[0].shape)
    my_train_masks = my_shuffled_masks[:number_train]
    my_val_images = my_shuffled_images[number_train:number_val]
    my_val_masks = my_shuffled_masks[number_train:number_val]
    my_test_images = my_shuffled_images[number_val:]
    my_test_masks = my_shuffled_masks[number_val:]

    train_ds = MyFlowDataset(
        my_train_images,
        my_train_masks,
    )
    print(type(train_ds[7]))
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=pin_memory,
    )

    val_ds = MyFlowDataset(
            my_val_images,
            my_val_masks,
        )

    val_loader = DataLoader(
            dataset=val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            # pin_memory=pin_memory,
        )

    test_ds = MyFlowDataset(
        my_test_images,
        my_test_masks,
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader

# TODO: implement SSIM as a metric for comparing similarity between two images
# as proposed in the paper "Image Quality Assessment: From Error Visibility to
# Structural Similarity" by Zhou et al. (2004)


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    # TODO: Remove dice score, as it is only applicable to classification tasks
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()


train_loader, val_loader, test_loader = get_loaders(
    32, 1, True, 31)
