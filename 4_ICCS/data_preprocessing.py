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
import torch.multiprocessing as mp


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


if __name__ == "__main__":
    clean_mamico_data_mp()
