"""data_preprocessing

This script allows to perform the required data preprocessing including data
cleaning and data visualization. Data cleaning is necessary since the initial
MaMiCo output files use faulty delimiters. Visualization of the raw data aims
to validate correctness/plausibility so as to reduce the likelihood of faulty
simulation data.

"""
import torch
import csv
import glob
import numpy as np
import torch.multiprocessing as mp
from plotting import plot_flow_profile


def clean_mamico_data(directory, file_name):
    """The clean_mamico_data cleans the raw mamico csv data. In particular,
    it replaces faulty ',' delimiters with proper ';' delimiters.

    Args:
        directory:
          Object of string type containing the path working directory (pwd) of
          the dataset to be cleaned. This is also the pwd of the cleaned file.
          Default: "/beegfs/project/MaMiCo/mamico-ml/dataset"
        file_name:
          Object of string type containing the name of the csv file to be
          cleaned.

    Returns:
        NONE:
          This function does not have a return value. Instead it saves the
          cleaned mamico data to file.
    """

    print(f'Cleaning MaMiCo Dataset: {file_name}.')
    text = open(f"{directory}/01_raw/{file_name}", "r")
    text = ''.join([i for i in text]) \
        .replace(",", ";")
    x = open(f"{directory}/02_clean/clean_{file_name}", "w")
    x.writelines(text)
    x.close()
    return


def clean_mamico_data_mp():
    """The clean_mamico_data_mp function is used to call clean_mamico_data
    function in a multiprocessing manner.
    """
    _directory = "/beegfs/project/MaMiCo/mamico-ml/dataset"
    _raw_files = glob.glob(
        f"{_directory}/01_raw/kvs_combined_domain_init*.csv")
    _files = []

    for _file in _raw_files:
        _file = _file.replace(_directory+'/01_raw/', '')
        _files.append(_file)

    processes = []

    for i in range(len(_raw_files)):
        p = mp.Process(
            target=clean_mamico_data,
            args=(_directory, _files[i],)
        )
        p.start()
        processes.append(p)
        print(f'Creating Process Number: {i+1}')

    for process in processes:
        process.join()
        print('Joining Process')


def clean2dataset(file_name):
    """The clean2dataset function reads from (cleaned) mamico
    generated csv files and returns the dataset in the form of a
    numpy array of shape (1000 x 3 x 26 x 26 x 26).

    Args:
        file_name:
          Object of string type containing the name of the csv file to be
          loaded as a dataset.
    Returns:
        dataset:
          A numpy array of shape (d_0 x d_1 x d_2 x d_3 x d_4) containing the
          MD dataset. Here, the first dimension, d_0, refers to the amount of
          coupling cycles. The second dimension, d_1, refers to the
          individual velocity components(=3=[u_x, U_y, u_z]). Finally, the
          remaining dimensions, d_2 = d_3 = d_4, refer to the spatial co-
          ordinates and reference the MD cells. The dataset is hardcoded for
          d_0 = 1000, d_1 = 3, d_2 = d_3 = d_4 = 26.
    """
    _directory = '/beegfs/project/MaMiCo/mamico-ml/dataset/02_clean/'
    print('Loading MaMiCo dataset from csv: ',
          file_name.replace(_directory, ''))
    _pwd = _directory + file_name
    print(_pwd)
    dataset = np.zeros((1000, 3, 26, 26, 26))
    counter = 0

    with open(_pwd) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')

        for row in csv_reader:
            a = row
            if(len(a) > 6):
                dataset[int(a[0])-1, 0, int(a[1])-1, int(a[2])
                        - 1, int(a[3])-1] = float(a[4])
                dataset[int(a[0])-1, 1, int(a[1])-1, int(a[2])
                        - 1, int(a[3])-1] = float(a[5])
                dataset[int(a[0])-1, 2, int(a[1])-1, int(a[2])
                        - 1, int(a[3])-1] = float(a[6])
                counter += 1
    counter = int(counter/17576)
    dataset = dataset[:counter+1]
    print(dataset.shape)
    return dataset


def clean2mlready(file_name):
    """The clean2mlready function is required to further clean the MaMiCo
    datasets such that they can be loaded more quickly. It loads the dataset
    via the clean2dataset function. It takes the cleaned datasets and removes
    zero-padding. Removing the padding is time consuming. This function therfor
    creates .csv files without the zero-padding making it much quicker and
    thus mlready.

    Args:
        file_name:
          Object of string type containing the name of the csv file (without pwd)
          to be loaded as a dataset.

    Returns:
        NONE:
          This function does not have a return value. Instead it saves the
          mlready dataset to file.
    """
    _directory = "beegfs/project/MaMiCo/mamico-ml/dataset"
    _dataset = clean2dataset(file_name)

    print(f'Saving dataset to csv: {file_name}')
    # 1) Convert 3D array to 2D array
    _dataset_reshaped = _dataset.reshape(_dataset.shape[0], -1)

    # 2) Save 2D array to file
    np.savetxt(f'dataset_mlready/{file_name}', _dataset_reshaped)


def clean2mlready_mp():
    """The clean2mlready_mp function is used to call the clean2mlready
    function in a multiprocessing manner.
    """
    _directory = "/beegfs/project/MaMiCo/mamico-ml/dataset"
    _raw_files = glob.glob(
        f"{_directory}/02_clean/*couette_combined*.csv")
    _files = []

    for _file in _raw_files:
        print(_file)
        _file = _file.replace(_directory+'/02_clean/', '')
        print(_file)
        _files.append(_file)

    processes = []

    for i in range(len(_raw_files)):
        p = mp.Process(
            target=clean2mlready,
            args=(_files[i],)
        )
        p.start()
        processes.append(p)
        print(f'Creating Process Number: {i+1}')

    for process in processes:
        process.join()
        print('Joining Process')


def mlready2dataset(file_name):
    """The mlready2dataset function retrieves a numpy array from a csv file.

    Args:
        file_name:
          Object of string type containing the name of the csv file to be
          loaded as a dataset.

    Returns:
        dataset:
          Object of numpy array type containing the dataset read from file.
    """
    print(f'Loading Dataset from csv: {file_name}')
    dataset = np.loadtxt(f'dataset_mlready/{file_name}')

    if dataset.size != (1000 * 3 * 26 * 26 * 26):
        print("Incorrect dimensions:", dataset.size, file_name)
        return

    t, c, d, h, w = (1000, 3, 26, 26, 26)

    original_dataset = dataset.reshape(t, c, d, h, w)
    return original_dataset


def visualize_clean_mamico_data_mp():
    """The visualize_clean_mamico_data visualizes the cleaned datasets so as to
    validate proper simulation, i.e. validate that the data makes sense. This
    is done by generating meaningful plots to recognize characteristic flow
    behavior (couette, couette_oscillating, KVS)
    Args:

    Returns:
        NONE:
          This function does not have a return value. Instead it generates the
          aforementioned meaningful plots.
    """
    print('Performing: visualize_clean_mamico_data_mp()')
    _directory = "/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/dataset_mlready"
    _raw_files = glob.glob(
        f"{_directory}/*kvs_combined_domain_init*.csv")
    _file_names = []
    _datasets = []

    for file in _raw_files:
        print(file)
        file_name = file.replace(_directory+'/', '')
        print(file_name)
        _file_names.append(file_name)
        dataset = mlready2dataset(file_name)
        _datasets.append(dataset)

    processes = []

    for i in range(len(_raw_files)):
        p = mp.Process(
            target=plot_flow_profile,
            args=(_datasets[i], _file_names[i],)
        )
        p.start()
        processes.append(p)
        print(f'Creating Process Number: {i+1}')

    for process in processes:
        process.join()
        print('Joining Process')
    pass


def visualize_mlready_dataset_mp():
    """The visualize_mlready_dataset_mp visualizes the mlready datasets so as to
    validate proper simulation, i.e. validate that the data makes sense. This
    is done by generating meaningful plots to recognize characteristic flow
    behavior (couette, couette_oscillating, KVS)
    Args:
        NONE

    Returns:
        NONE:
          This function does not have a return value. Instead it generates the
          aforementioned meaningful plots.
    """
    print('Performing: visualize_mlready_dataset_mp()')
    _directory = "/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/dataset_mlready"
    _raw_files = glob.glob(
        f"{_directory}/**/*NW_1.csv", recursive=True)
    _file_names = []
    _datasets = []

    for file in _raw_files:
        print(f'Raw file: {file}')
        _file_name = file.replace(_directory+'/', '')
        print(f'New file: {_file_name}')
        _file_names.append(_file_name)
        _dataset = mlready2dataset(_file_name)
        _datasets.append(_dataset)

    processes = []

    for i in range(len(_raw_files)):
        p = mp.Process(
            target=plot_flow_profile,
            args=(_datasets[i], _file_names[i],)
        )
        p.start()
        processes.append(p)
        print(f'Creating Process Number: {i+1}')

    for process in processes:
        process.join()
        print('Joining Process')


def mlready2augmented(file_name):
    """The mlready2augmented function retrieves a numpy array from a csv file
    and augments it such that the channels are swapped and then saved as new
    datasets. Here, the augmented datasets are fitted with suffixes inorder to
    understand the augmentation.
    _1: The original channels are shifted by 1 [0]->[1], [1]->[2], [2]->[0]
    _2: The original channels are shifted by 1 [0]->[2], [1]->[0], [2]->[1]

    Args:
        file_name:
          Object of string type containing the name of the csv file as the
          basis of augmentation.

    Returns:
        NONE:
          This function saves the augmented datasets to file.
    """
    print(f'Loading Dataset from csv: {file_name} [.csv]')
    _directory = "/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/"
    _dummy = file_name.replace(_directory, '')
    print(_dummy)
    _dummy += '.csv'
    print(_dummy)
    dataset = np.loadtxt(f'{_dummy}')

    if dataset.size != (1000 * 3 * 26 * 26 * 26):
        print("Incorrect dimensions:", dataset.size, file_name)
        return

    t, c, d, h, w = (1000, 3, 26, 26, 26)

    original_dataset = dataset.reshape(t, c, d, h, w)
    augmented_1 = np.concatenate(
        (original_dataset[:, 1, :, :, :].reshape((t, 1, d, h, w)),
         original_dataset[:, 2, :, :, :].reshape((t, 1, d, h, w)),
         original_dataset[:, 0, :, :, :].reshape((t, 1, d, h, w))), 1)
    augmented_2 = np.concatenate(
        (original_dataset[:, 2, :, :, :].reshape((t, 1, d, h, w)),
         original_dataset[:, 0, :, :, :].reshape((t, 1, d, h, w)),
         original_dataset[:, 1, :, :, :].reshape((t, 1, d, h, w))), 1)

    augmented_1_reshaped = augmented_1.reshape(augmented_1.shape[0], -1)
    augmented_2_reshaped = augmented_2.reshape(augmented_2.shape[0], -1)
    np.savetxt(f'{file_name}_1.csv', augmented_1_reshaped)
    np.savetxt(f'{file_name}_2.csv', augmented_2_reshaped)


def mlready2augmented_mp():
    """The mlready2augmented_mp function is used to call the mlready2augmented
    function in a multiprocessing manner.
    """
    _directory = "/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/dataset_mlready"
    _raw_files = glob.glob(
        f"{_directory}/**/*.csv", recursive=True)
    _files = []

    for file in _raw_files:
        print(file)
        _file = file.replace('.csv', '')
        print(_file)
        _files.append(_file)

    processes = []

    for idx, file in enumerate(_files):
        p = mp.Process(
            target=mlready2augmented,
            args=(file,)
        )
        p.start()
        processes.append(p)
        print(f'Creating Process Number: {idx+1}')

    for process in processes:
        process.join()
        print('Joining Process')


if __name__ == "__main__":
    print('Starting Data Preprocessing: Visualization of Augmented Datasets')
    visualize_mlready_dataset_mp()
    # clean2mlready_mp()
    # mlready2augmented_mp()

    pass
