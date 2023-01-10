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

print('testing')


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
    dataset = np.zeros((1000, 3, 26, 26, 26))
    counter = 0

    with open(file_name) as csvfile:
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
    _directory = "/beegfs/project/MaMiCo/mamico-ml/dataset"
    _dataset = clean2dataset(file_name)

    print(f'Saving dataset to csv: {file_name}')
    # 1) Convert 3D array to 2D array
    _dataset_reshaped = _dataset.reshape(_dataset.shape[0], -1)

    # 2) Save 2D array to file
    np.savetxt(f'{_directory}/03_mlready/{file_name}.csv', _dataset_reshaped)


def clean2mlready_mp():
    """The clean2mlready_mp function is used to call the clean2mlready
    function in a multiprocessing manner.
    """
    _directory = "/beegfs/project/MaMiCo/mamico-ml/dataset"
    _raw_files = glob.glob(
        f"{_directory}/02_clean/*kvs_combined_domain_init*.csv")
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
    pass


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
    _directory = "/beegfs/project/MaMiCo/mamico-ml/dataset"
    _raw_files = glob.glob(
        f"{_directory}/02_clean/*kvs_combined_domain_init*.csv")
    _file_names = []
    _datasets = []

    for file in _raw_files:
        print(file)
        _datasets.append(clean2dataset(file))
        file_name = file.replace(_directory+'/', '')
        print(file_name)
        _file_names.append(file_name)

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


if __name__ == "__main__":
    # clean_mamico_data_mp()
    # visualize_clean_mamico_data_mp()
    clean2mlready_mp()
    pass
