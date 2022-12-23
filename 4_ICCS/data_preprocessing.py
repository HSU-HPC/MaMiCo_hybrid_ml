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
        file_name:
          Object of string type containing the name of the csv file to be
          cleaned.

    Returns:
        NONE:
          This function does not have a return value. Instead it saves the
          cleaned mamico data to file.
    """

    print(f'Cleaning MaMiCo Dataset: {file_name}.')
    text = open(f"{directory}/{file_name}", "r")
    text = ''.join([i for i in text]) \
        .replace(",", ";")
    x = open(f"{directory}/clean_{file_name}", "w")
    x.writelines(text)
    x.close()
    return


def clean_mamico_data_mp():
    """The clean_mamico_data_mp function is used to call clean_mamico_data
    function in a multiprocessing manner.
    """
    _directory = "/beegfs/project/MaMiCo/mamico-ml/dataset"
    _raw_files = glob.glob(f"{_directory}/*.csv")
    _files = []

    for _file in _raw_files:
        _file = _file.replace(_directory+'/', '')
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


def mamico_csv2dataset(file_name):
    """The mamico_csv2dataset function reads from (cleaned) mamico
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
    _directory = '/beegfs/project/MaMiCo/mamico-ml/dataset/'
    print('Loading MaMiCo dataset from csv: ',
          file_name.replace(_directory, ''))
    dataset = np.zeros((1000, 3, 26, 26, 26))
    counter = 0

    with open(_directory+file_name) as csvfile:
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
    _directory = "/beegfs/project/MaMiCo/mamico-ml/dataset"
    _raw_files = glob.glob(f"{_directory}/*.csv")
    _files = []

    for _file in _raw_files:
        _file = _file.replace(_directory+'/', '')
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
    pass


if __name__ == "__main__":
    mamico_csv2dataset('clean_couette_test_combined_domain_1_0.csv')
