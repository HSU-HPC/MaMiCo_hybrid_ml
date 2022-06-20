import numpy as np
import csv
import torch
from plotting import colorMap, showSample, plotLoss, plotLoss34


def randomRGBArray(t=0, ch=0, d=0, h=0, w=1):
    return np.random.rand(t, ch, d, h, w)


def save3D_RGBArray2File(input_array, string_file_name):
    # 1) Convert 3D array to 2D array
    input_reshaped = input_array.reshape(input_array.shape[0], -1)

    # 2) Save 2D array to file
    t, c, d, h, w = input_array.shape
    if t != 32:
        name = f'{string_file_name}_{t}_{c}_{d}_{h}_{w}'
        np.savetxt(f'{name}.csv', input_reshaped)


def load3D_RGBArrayFromFile(input_file, output_shape):
    # 3) load 2D array from file
    loaded_array = np.loadtxt(f'Results/{input_file}')
    #f'/Users/sebastianlerdo/Desktop/Bundeswehr/Uni_Bw/HPC Laboratory_Project/Python Scripts/Predictions and Targets/Test/{input_file}')
    t, c, d, h, w = output_shape

    # 4) Revert 2D array to 3D array
    original_array = loaded_array.reshape(t, c, d, h, w)
    return original_array


def loadLoss2List(input_file):
    return np.loadtxt(f'Results/{input_file}.csv')


def checkSaveLoad(input_array, loaded_array):
    print("shape of input array: ", input_array.shape)
    print("shape of loaded array: ", loaded_array.shape)

    if (input_array == loaded_array).all():
        print("Yes, both the arrays are same")
    else:
        print("No, both the arrays are not same")


def calculateFLOPS(c, d, h, w, features):
    '''
    This model assumes:
        input_dim:    [d, h, w] are perfectly divisible by 2, i.e powers of two
        convolutions: kernel_size = 3, stride = 1
        batchnorm:    Parameters folded into convolution when deployed in final state
        activations:  ReLu
        bottleneck:   doubles the channels
    '''

    print(f'This model takes an input size of: {c} x {d} x {h} x {w}')
    print(f'It applies a U-Net with the following feature list: {features}')
    flops = 0
    maccs = 0
    data_dim = [c, d, h, w]
    k = 3
    # ##########
    # Flops during contracting path
    for c_out_d in features:
        c_in_d = data_dim[0]
        voxels_plus_1_d = (data_dim[1]+1) * (data_dim[2]+1) * (data_dim[3] + 1)
        voxels_d = (data_dim[1]) * (data_dim[2]) * (data_dim[3])
        # First convolution flops:
        # 2 * k * k * k * c_in * (d+1) * (h+1) * (w+1) * c_out
        maccs += k * k * k * c_in_d * voxels_plus_1_d * c_out_d
        flops += 2 * k * k * k * c_in_d * voxels_plus_1_d * c_out_d
        # BatchNorm3d Negligible due to parameters folded into convolution
        flops += 0
        # ReLU flops
        flops += voxels_d * c_out_d
        # tanh flops
        # flops += voxels_d * c_out_d * 8
        # Second convolution flops:
        maccs += k * k * k * c_out_d * voxels_plus_1_d * c_out_d
        flops += 2 * k * k * k * c_out_d * voxels_plus_1_d * c_out_d
        # BatchNorm3d Negligible due to parameters folded into convolution
        flops += 0
        # ReLU flops
        flops += voxels_d * c_out_d
        # tanh flops
        # flops += voxels_d * c_out_d * 8
        # Pooling flops
        flops += voxels_d * c_out_d
        data_dim = [c_out_d, int(data_dim[1]*0.5),
                    int(data_dim[2]*0.5), int(data_dim[3]*0.5)]

    # ##########
    # flops during bottleneck
    c_in_b, d_b, h_b, w_b = data_dim
    c_out_b = 2 * c_in_b
    voxels_plus_1_b = (d_b + 1) * (h_b + 1) * (w_b + 1)
    voxels_b = d_b * h_b * w_b
    # First convolution flops:
    # 2 * k * k * k * c_in * (d+1) * (h+1) * (w+1) * c_out
    maccs += k * k * k * c_in_b * voxels_plus_1_b * c_out_b
    flops += 2 * k * k * k * c_in_b * voxels_plus_1_b * c_out_b
    # BatchNorm3d negligible due to parameters folded into convolution
    flops += 0
    # ReLU flops
    flops += voxels_b * c_out_b
    # tanh flops
    # flops += voxels_b * c_out_b * 8
    # Second convolution flops:
    maccs += k * k * k * c_out_b * voxels_plus_1_b * c_out_b
    flops += 2 * k * k * k * c_out_b * voxels_plus_1_b * c_out_b
    # BatchNorm3d negligible due to parameters folded into convolution
    flops += 0
    # ReLU flops
    flops += voxels_b * c_out_b
    # tanh flops
    # flops += voxels_b * c_out_b * 8
    data_dim[0] = c_out_b

    # ##########
    # Flops during expanding path
    for c_out_u in features:
        voxels_u = (data_dim[1]) * (data_dim[2]) * (data_dim[3])
        c_in_u = data_dim[0]
        # ConvTranspose3d flops
        maccs += c_in_u * voxels_u * k * k * k * c_out_u
        flops += 2 * c_in_u * voxels_u * k * k * k * c_out_u
        data_dim = [c_out_u, data_dim[1]*2, data_dim[2]*2, data_dim[3]*2]
        # New dimensions
        voxels_plus_1_u = (data_dim[1]+1) * (data_dim[2]+1) * (data_dim[3] + 1)
        voxels_u = (data_dim[1]) * (data_dim[2]) * (data_dim[3])

        # First convolution (consider concatenation, i.e. *2)
        # 2 * k * k * k * c_in * (d+1) * (h+1) * (w+1) * c_out
        maccs += 2 * k * k * k * c_out_u * voxels_plus_1_u * c_out_u
        flops += 2 * 2 * k * k * k * c_out_u * voxels_plus_1_u * c_out_u
        # BatchNorm3d Negligible due to parameters folded into convolution
        flops += 0
        # ReLU flops
        flops += voxels_u * c_out_u
        # tanh flops
        # flops += voxels_u * c_out_u * 8
        # Second convolution flops:
        maccs += k * k * k * c_out_u * voxels_plus_1_u * c_out_u
        flops += 2 * k * k * k * c_out_u * voxels_plus_1_u * c_out_u
        # BatchNorm3d Negligible due to parameters folded into convolution
        flops += 0
        # ReLU flops
        flops += voxels_u * c_out_u
        # tanh flops
        # flops += voxels_u * c_out_u * 8

    # ##########
    # Flops for final convolution
    voxels_f = (data_dim[1]) * (data_dim[2]) * (data_dim[3])
    maccs += 1 * 1 * 1 * c_out_u * voxels_f * 3
    flops += 2 * 1 * 1 * 1 * c_out_u * voxels_f * 3

    print(f'The total number of FLOPS is: {flops}')
    print(f'The total number of GFLOPS is: {flops/(10**9)}')
    print(f'The therein included number of MACCs is: {maccs} ')
    print(f'The therein included number of GMACCs is: {maccs/(10**9)} ')


def testColorMap():
    '''
    p_MAE = load3D_RGBArrayFromFile(
        'predictions_1_MAE_5_3_32_32_32.csv', (5, 3, 32, 32, 32))
    p_MSE = load3D_RGBArrayFromFile(
        'predictions_1_MSE_5_3_32_32_32.csv', (5, 3, 32, 32, 32))
    target = load3D_RGBArrayFromFile(
        'targets_1_MAE_5_3_32_32_32.csv', (5, 3, 32, 32, 32))
    analytical = load3D_RGBArrayFromFile(
        'analytical_10_99_1_3_32_32_32.csv', (1, 3, 32, 32, 32))
    u_1 = target[2, 0, :, :, :]
    u_2 = analytical[0, 0, :, :, :]
    u_3 = p_MAE[2, 0, :, :, :]
    '''
    t = 1000
    u = 10
    w = 20
    n = 2
    v = 31
    sigma = 0
    seed = 1
    analytical = my3DCouetteSolver(desired_timesteps=t, u_wall=u, wall_height=w,
                                   nu=n, vertical_resolution=v, sigma=sigma, my_seed=seed)
    u_1 = analytical[1, 0, :, :, :]
    '''
    u_2 = analytical[15, 0, :, :, :]
    u_3 = analytical[30, 0, :, :, :]
    u_4 = analytical[62, 0, :, :, :]
    u_5 = analytical[125, 0, :, :, :]
    u_6 = analytical[250, 0, :, :, :]
    u_7 = analytical[500, 0, :, :, :]
    '''
    u_8 = analytical[1000-1, 0, :, :, :]
    u = [u_1, u_8]
    colorMap(u)


def trial1():
    p_MAE = load3D_RGBArrayFromFile(
        'predictions_1_MAE_5_3_32_32_32.csv', (5, 3, 32, 32, 32))
    p_MSE = load3D_RGBArrayFromFile(
        'predictions_1_MSE_5_3_32_32_32.csv', (5, 3, 32, 32, 32))
    target = load3D_RGBArrayFromFile(
        'targets_1_MAE_5_3_32_32_32.csv', (5, 3, 32, 32, 32))
    analytical = load3D_RGBArrayFromFile(
        'analytical_10_99_1_3_32_32_32.csv', (1, 3, 32, 32, 32))
    analytical = analytical[0]

    title = 'Trial 1'
    save_as = 'Trial_01'
    compareFlowProfile(
        title, save_as, p_MAE[2], p_MSE[2], target[2], analytical)


def losses():

    for i in [1, 2, 5, 6]:
        losses_MAE = loadLoss2List(f'Losses_trial_{i}_MAE')
        losses_MSE = loadLoss2List(f'Losses_trial_{i}_MSE')
        plotLoss(losses_MAE, losses_MSE, f'{i}')

    labels = [['MAE', '1e-3', 'MAE', '2e-3', 'MSE', '1e-3', 'MSE', '2e-3'],
              ['MAE', '20', 'MAE', '40', 'MSE', '20', 'MSE', '40']]

    for i in [3, 4]:
        losses_MAE_1 = loadLoss2List(
            f'Losses_trial_{i}_{labels[(i-3)][0]}_{labels[(i-3)][1]}')
        losses_MAE_2 = loadLoss2List(
            f'Losses_trial_{i}_{labels[(i-3)][2]}_{labels[(i-3)][3]}')
        losses_MSE_1 = loadLoss2List(
            f'Losses_trial_{i}_{labels[(i-3)][4]}_{labels[(i-3)][5]}')
        losses_MSE_2 = loadLoss2List(
            f'Losses_trial_{i}_{labels[(i-3)][6]}_{labels[(i-3)][7]}')
        plotLoss34(losses_MAE_1, losses_MAE_2, losses_MSE_1,
                   losses_MSE_2, labels[i-3], f'{i}')


def test1():
    p_MAE = load3D_RGBArrayFromFile(
        'T_1_pred_MAE_5_3_32_32_32.csv', (5, 3, 32, 32, 32))
    p_MSE = load3D_RGBArrayFromFile(
        'T_1_pred_MSE_5_3_32_32_32.csv', (5, 3, 32, 32, 32))
    target = load3D_RGBArrayFromFile(
        'T_1_target_MAE_5_3_32_32_32.csv', (5, 3, 32, 32, 32))
    analytical = load3D_RGBArrayFromFile(
        'analytical_10_99_1_3_32_32_32.csv', (1, 3, 32, 32, 32))
    analytical = analytical[0]

    title = 'Test 1: Different Random Seed'
    save_as = 'Test_01'
    # compareFlowProfile(
    #    title, save_as, p_MAE[2], p_MSE[2], target[2], analytical)
    compareUxFlowProfile(
        title, save_as, p_MAE[2], p_MSE[2], target[2], analytical)


def test2():
    p_MAE = load3D_RGBArrayFromFile(
        'T_2_pred_MAE_5_3_32_32_32.csv', (5, 3, 32, 32, 32))
    p_MSE = load3D_RGBArrayFromFile(
        'T_2_pred_MSE_5_3_32_32_32.csv', (5, 3, 32, 32, 32))
    target = load3D_RGBArrayFromFile(
        'T_2_target_MAE_5_3_32_32_32.csv', (5, 3, 32, 32, 32))
    analytical = load3D_RGBArrayFromFile(
        'analytical_5_99_1_3_32_32_32.csv', (1, 3, 32, 32, 32))
    analytical = analytical[0]

    title = 'Test 2: Wall Speed $U = 5$'
    save_as = 'Test_02'
    # compareFlowProfile(
    #     title, save_as, p_MAE[2], p_MSE[2], target[2], analytical)
    compareUxFlowProfile(
        title, save_as, p_MAE[2], p_MSE[2], target[2], analytical)


def test3():
    p_MAE = load3D_RGBArrayFromFile(
        'T_3_pred_MAE_5_3_32_32_32.csv', (5, 3, 32, 32, 32))
    p_MSE = load3D_RGBArrayFromFile(
        'T_3_pred_MSE_5_3_32_32_32.csv', (5, 3, 32, 32, 32))
    target = load3D_RGBArrayFromFile(
        'T_3_target_MAE_5_3_32_32_32.csv', (5, 3, 32, 32, 32))
    analytical = load3D_RGBArrayFromFile(
        'analytical_10_99_1_3_32_32_32.csv', (1, 3, 32, 32, 32))
    analytical = analytical[0]

    title = 'Test 3: Increased Noise $ \sigma = 0.5U$'
    save_as = 'Test_03'
    # compareFlowProfile(
    #     title, save_as, p_MAE[2], p_MSE[2], target[2], analytical)
    compareUxFlowProfile(
        title, save_as, p_MAE[2], p_MSE[2], target[2], analytical)


def test4():
    p_MAE = load3D_RGBArrayFromFile(
        'T_4_pred_MAE_5_3_64_64_64.csv', (5, 3, 64, 64, 64))
    p_MSE = load3D_RGBArrayFromFile(
        'T_4_pred_MSE_5_3_64_64_64.csv', (5, 3, 64, 64, 64))
    target = load3D_RGBArrayFromFile(
        'T_4_target_MAE_5_3_64_64_64.csv', (5, 3, 64, 64, 64))
    analytical = load3D_RGBArrayFromFile(
        'analytical_10_99_1_3_64_64_64.csv', (1, 3, 64, 64, 64))
    analytical = analytical[0]

    title = 'Test 4: Increased Spatial Resolution 64 x 64 x 64'
    save_as = 'Test_04'
    compareFlowProfile(
        title, save_as, p_MAE[2], p_MSE[2], target[2], analytical)
    titles = ['MAE Prediction', 'MSE Prediction',
              'Target', 'Analytical']
    colorMap([p_MAE[2, 0, :, :, :], p_MSE[2, 0, :, :, :],
              target[2, 0, :, :, :], analytical[0, :, :, :]], titles)
    # compareUxFlowProfile(
    #     title, save_as, p_MAE[2], p_MSE[2], target[2], analytical)


def analytical_1_3():
    t = 1000
    u = 10
    w = 20
    n = 2
    v = 31
    sigma = 0
    seed = 1
    analytical = my3DCouetteSolver(desired_timesteps=t, u_wall=u, wall_height=w,
                                   nu=n, vertical_resolution=v, sigma=sigma, my_seed=seed)
    print(analytical.shape)
    # Extract t = 99
    analytical = analytical[99].reshape(1, 3, 32, 32, 32)
    save3D_RGBArray2File(analytical, 'analytical_10_99')


def analytical_2():
    t = 1000
    u = 5
    w = 20
    n = 2
    v = 31
    sigma = 0
    seed = 1
    analytical = my3DCouetteSolver(desired_timesteps=t, u_wall=u, wall_height=w,
                                   nu=n, vertical_resolution=v, sigma=sigma, my_seed=seed)
    print(analytical.shape)
    # Extract t = 99
    analytical = analytical[99].reshape(1, 3, 32, 32, 32)
    save3D_RGBArray2File(analytical, 'analytical_5_99')

    pass


def analytical_4():
    t = 1000
    u = 10
    w = 20
    n = 2
    v = 63
    sigma = 0
    seed = 1
    analytical = my3DCouetteSolver(desired_timesteps=t, u_wall=u, wall_height=w,
                                   nu=n, vertical_resolution=v, sigma=sigma, my_seed=seed)
    print(analytical.shape)
    # Extract t = 99
    analytical = analytical[99].reshape(1, 3, 64, 64, 64)
    save3D_RGBArray2File(analytical, 'analytical_10_99')


def variance(train, valid):
    return abs(train-valid)/train


def main():
    x = torch.rand(3, 32, 32, 32)
    print(x.shape)
    print(torch.unsqueeze(x, 0).shape)

    input_dim = (3, 32, 32, 32)
    h_dims = []
    out_dim = (3, 32, 32, 32)
    layer_dims = [input_dim] + h_dims + [out_dim]
    print(layer_dims)


if __name__ == "__main__":
    main()
