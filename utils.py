import torch
import torchvision
import random
import torch
import torch.nn as nn
import numpy as np
from couette_solver import my3DCouetteSolver
from dataset import MyFlowDataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(batch_size, num_workers, pin_memory, timesteps, couette_dim, sigma=0):
    # Consider that the couette solver now requires a desired_timesteps
    # parameter for improved reusabilty

    my_couette_data = my3DCouetteSolver(
        desired_timesteps=timesteps, vertical_resolution=couette_dim, sigma=sigma, my_seed=1)
    my_couette_data_valid = my3DCouetteSolver(
        desired_timesteps=timesteps, vertical_resolution=couette_dim, sigma=sigma, my_seed=2)

    my_images = my_couette_data[:-1]
    my_masks = my_couette_data[1:]

    my_images_valid = my_couette_data_valid[:-1]
    my_masks_valid = my_couette_data_valid[1:]

    my_zip = list(zip(my_images, my_masks))
    random.shuffle(my_zip)
    my_shuffled_images, my_shuffled_masks = zip(*my_zip)

    my_train_images = my_shuffled_images
    my_train_masks = my_shuffled_masks

    my_val_images = my_images_valid
    my_val_masks = my_masks_valid

    train_ds = MyFlowDataset(
        my_train_images,
        my_train_masks,
    )
    # print(type(train_ds[7]))
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

    return train_loader, val_loader


def get_5_loaders(batch_size, num_workers, pin_memory, timesteps, couette_dim, sigma=0):
    my_couette_data_1 = my3DCouetteSolver(
        desired_timesteps=timesteps, vertical_resolution=couette_dim, sigma=sigma, my_seed=1)
    my_couette_data_2 = my3DCouetteSolver(
        desired_timesteps=timesteps, vertical_resolution=couette_dim, sigma=sigma, my_seed=3)
    my_couette_data_3 = my3DCouetteSolver(
        desired_timesteps=timesteps, vertical_resolution=couette_dim, sigma=sigma, my_seed=4)
    my_couette_data_4 = my3DCouetteSolver(
        desired_timesteps=timesteps, vertical_resolution=couette_dim, sigma=sigma, my_seed=5)
    my_couette_data_5 = my3DCouetteSolver(
        desired_timesteps=timesteps, vertical_resolution=couette_dim, sigma=sigma, my_seed=6)
    my_couette_data_valid = my3DCouetteSolver(
        desired_timesteps=timesteps, vertical_resolution=couette_dim, sigma=sigma, my_seed=2)

    my_images = np.concatenate((my_couette_data_1[:-1], my_couette_data_2[:-1],
                               my_couette_data_3[:-1], my_couette_data_4[:-1], my_couette_data_5[:-1]), axis=0)
    my_masks = np.concatenate((my_couette_data_1[1:], my_couette_data_2[1:],
                              my_couette_data_3[1:], my_couette_data_4[1:], my_couette_data_5[1:]), axis=0)

    my_images_valid = my_couette_data_valid[:-1]
    my_masks_valid = my_couette_data_valid[1:]

    my_zip = list(zip(my_images, my_masks))
    random.shuffle(my_zip)
    my_shuffled_images, my_shuffled_masks = zip(*my_zip)

    my_train_images = my_shuffled_images
    my_train_masks = my_shuffled_masks

    my_val_images = my_images_valid[: 101]
    my_val_masks = my_masks_valid[: 101]

    train_ds = MyFlowDataset(
        my_train_images,
        my_train_masks,
    )
    # print(type(train_ds[7]))
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

    return train_loader, val_loader


def get_loaders_test(batch_size, num_workers, pin_memory, timesteps=1000, couette_dim=31, sigma=0.3):

    # 01 test: Different seed=2
    my_couette_data_test_1 = my3DCouetteSolver(
        desired_timesteps=timesteps, vertical_resolution=couette_dim, sigma=sigma, my_seed=3)
    # 02 test: Increased noise sigma=0.5
    my_couette_data_test_2 = my3DCouetteSolver(
        desired_timesteps=timesteps, u_wall=5, vertical_resolution=couette_dim, sigma=sigma, my_seed=3)
    # 03 test: Lower wall speed
    my_couette_data_test_3 = my3DCouetteSolver(
        desired_timesteps=timesteps, vertical_resolution=couette_dim, sigma=0.5, my_seed=3)
    # 04 test: Increased wall height
    my_couette_data_test_4 = my3DCouetteSolver(
        desired_timesteps=timesteps, vertical_resolution=63, sigma=0.5, my_seed=3)

    my_images_1 = my_couette_data_test_1[:101]
    my_masks_1 = my_couette_data_test_1[1:102]

    my_images_2 = my_couette_data_test_2[:101]
    my_masks_2 = my_couette_data_test_2[1:102]

    my_images_3 = my_couette_data_test_3[:101]
    my_masks_3 = my_couette_data_test_3[1:102]

    my_images_4 = my_couette_data_test_4[:101]
    my_masks_4 = my_couette_data_test_4[1:102]

    # ############################################################
    test_ds_1 = MyFlowDataset(
        my_images_1,
        my_masks_1)

    test_loader_1 = DataLoader(
        dataset=test_ds_1,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)
    # ############################################################
    test_ds_2 = MyFlowDataset(
        my_images_2,
        my_masks_2)

    test_loader_2 = DataLoader(
        dataset=test_ds_2,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)
    # ############################################################
    test_ds_3 = MyFlowDataset(
        my_images_3,
        my_masks_3)

    test_loader_3 = DataLoader(
        dataset=test_ds_3,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)
    # ############################################################
    test_ds_4 = MyFlowDataset(
        my_images_4,
        my_masks_4)

    test_loader_4 = DataLoader(
        dataset=test_ds_4,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)
    # ############################################################

    return test_loader_1, test_loader_2, test_loader_3, test_loader_4


def load3D_RGBArrayFromFile(input_file, output_shape):
    # 3) load 2D array from file
    loaded_array = np.loadtxt(
        f'/Users/sebastianlerdo/github/MD_U-Net/{input_file}')
    #f'/Users/sebastianlerdo/github/MD_U-Net/{input_file}')
    t, c, d, h, w = output_shape

    # 4) Revert 2D array to 3D array
    original_array = loaded_array.reshape(t, c, d, h, w)
    return original_array


def get_loaders_from_file(batch_size, num_workers, pin_memory):
    file_name = "latent_space_test_MSE_999_64_2_2_2.csv"
    data_shape = (999, 64, 2, 2, 2)
    my_loaded_data_1 = load3D_RGBArrayFromFile(file_name, data_shape)
    print("Checking dimension of loaded data: ", my_loaded_data_1.shape)
    my_loaded_data_1 = np.reshape(my_loaded_data_1, (999, 512))
    print("Checking dimension of reshaped data: ", my_loaded_data_1.shape)
    lstm_data = np.zeros((994, 5, 512))
    for i in range(my_loaded_data_1.shape[0]-5):
        lstm_data[i] = my_loaded_data_1[i:i+5, :]
    print("Checking dimension of lstm data: ", lstm_data.shape)
    my_images_1 = lstm_data
    print("Checking dimension of input data: ", my_images_1.shape)
    my_masks_1 = my_loaded_data_1[5:, :]
    print("Checking dimension of target data: ", my_masks_1.shape)

    my_dataset = MyFlowDataset(my_images_1, my_masks_1)
    my_loader = DataLoader(
        dataset=my_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
        )

    return my_loader


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


def losses2file(losses, filename):
    np.savetxt(f"Losses_{filename}.csv", losses, delimiter=", ", fmt='% s')


if __name__ == "__main__":
    a = get_loaders_from_file(batch_size=32, num_workers=2, pin_memory=True)
    '''
    random_data = np.random.rand(999, 64, 2, 2, 2)
    print(random_data.shape)
    random_structured_data = np.reshape(random_data, (999, 512))
    print(random_structured_data.shape)
    lstm_data = np.zeros((994, 5, 512))

    for i in range(random_structured_data.shape[0]-5):
        lstm_data[i] = random_structured_data[i:i+5, :]

    print(lstm_data.shape)
    '''
    pass
