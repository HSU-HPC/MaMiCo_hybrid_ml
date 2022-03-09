import numpy as np
from couette_solver import my3DCouetteSolver
from plotting import compareFlowProfile


def randomRGBArray(t=0, ch=0, d=0, h=0, w=1):
    return np.random.rand(t, ch, d, h, w)


def save3DArray2File(input_array, prediction):
    # 1) Convert 3D array to 2D array
    input_reshaped = input_array.reshape(input_array.shape[0], -1)

    # 2) Save 2D array to file
    t, c, x, y = input_array.shape
    name = f'{prediction}_{t}_{x}_{y}'
    np.savetxt(f'{name}.csv', input_reshaped)


def save3D_RGBArray2File(input_array, string_file_name):
    # 1) Convert 3D array to 2D array
    input_reshaped = input_array.reshape(input_array.shape[0], -1)

    # 2) Save 2D array to file
    t, c, d, h, w = input_array.shape
    if t != 32:
        name = f'{string_file_name}_{t}_{c}_{d}_{h}_{w}'
        np.savetxt(f'{name}.csv', input_reshaped)


def load3DArrayFromFile(input_file, input_shape):
    # 3) load 2D array from file
    loaded_array = np.loadtxt(f'{input_file}')

    # 4) Revert 2D array to 3D array
    original_array = loaded_array.reshape(
        loaded_array.shape[0], loaded_array.shape[1] // input_shape[2], input_shape[2])
    return original_array


def load3D_RGBArrayFromFile(input_file, output_shape):
    # 3) load 2D array from file
    loaded_array = np.loadtxt(f'{input_file}')
    t, c, d, h, w = output_shape

    # 4) Revert 2D array to 3D array
    original_array = loaded_array.reshape(t, c, d, h, w)
    return original_array


def checkSaveLoad(input_array, loaded_array):
    print("shape of input array: ", input_array.shape)
    print("shape of loaded array: ", loaded_array.shape)

    if (input_array == loaded_array).all():
        print("Yes, both the arrays are same")
    else:
        print("No, both the arrays are not same")


def main():
    '''
    my_RGB_couette_noisy = my3DCouetteSolver(20, sigma=0.3)
    my_RGB_couette_clean = my3DCouetteSolver(20, sigma=0.0)
    print(my_RGB_couette_clean.shape)
    print(my_RGB_couette_clean[0].shape)
    my_RGB_couette_shape = my_RGB_couette_noisy.shape
    save3D_RGBArray2File(my_RGB_couette_noisy, 'prediction')
    '''

    a = np.random.rand(10, 3, 32, 32, 32)
    b = np.random.rand(10, 3, 32, 32, 32)

    c = np.concatenate((a, b), axis=0)

    print(c.shape)


if __name__ == "__main__":
    main()
