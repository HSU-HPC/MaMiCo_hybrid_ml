import torch
import numpy as np
import csv
import glob
import torch.multiprocessing as mp
import concurrent.futures
from dataset import MyMamicoDataset_UNET_AE, MyMamicoDataset_RNN, MyMamicoDataset_RNN_analysis, MyMamicoDataset_Hybrid
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata'
    print('Loading MaMiCo dataset from csv: ',
          file_name.replace(_directory, ''))
    dataset = np.zeros((1000, 3, 26, 26, 26))

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

    return dataset


def mamico_csv2dataset_mp(file_names):
    """The mamico_csv2dataset_mp function is used to call the mamico_csv2dataset
    function in a multiprocessing manner. It takes a list of file_names and
    returns the corresponding datasets as a list.

    Args:
        file_names:
          Object of list type containing objects of string type containing the
          name of the csv files to be loaded as datasets.
    Returns:
        results:
          A list containing the corresponding datasets of type numpy array.
    """

    print('Loading MaMiCo Datasets from csv via Multiprocessing.')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(mamico_csv2dataset, file_names)
    return results


def dataset2csv(dataset, dataset_name,  model_identifier=''):
    """The dataset2csv function saves a np.array to a csv file.

    Args:
        dataset:
          Object of numpy array type containing the dataset of interest.
        dataset_name:
          Object of string type containing the name of the dataset to be saved
          to file.
        model_identifier:
          A unique string to identify the model that generated the dataset.

    Returns:
        NONE:
          This function does not have a return value. Instead it saves the
          dataset of interest to file.
    """
    print(f'Saving dataset to csv: {dataset_name}')
    # 1) Convert 3D array to 2D array
    dataset_reshaped = dataset.reshape(dataset.shape[0], -1)
    # 2) Save 2D array to file
    name = dataset_name
    if model_identifier != '':
        name = f'{dataset_name}_model_{model_identifier}'
    np.savetxt(f'{name}.csv', dataset_reshaped)


def csv2dataset(file_name, output_shape=0):
    """The csv2dataset function retrieves a numpy array from a csv file.

    Args:
        file_name:
          Object of string type containing the name of the csv file to be
          loaded as a dataset.
        output_shape:
          Object of tuple type containing the shape of the desired numpy
          array.

    Returns:
        dataset:
          Object of numpy array type containing the dataset read from file.
    """
    print(f'Loading Dataset from csv: {file_name}')
    dataset = np.loadtxt(f'{file_name}')

    if output_shape == 0:
        return dataset

    t, c, d, h, w = output_shape

    original_dataset = dataset.reshape(t, c, d, h, w)
    return original_dataset


def csv2dataset_mp(filenames, output_shape=0):
    """The csv2dataset_mp function is used to call the csv2dataset function in
    a multiprocessing manner. It takes a list of file_names and returns the
    corresponding datasets as a list.

    Args:
        file_names:
          Object of list type containing objects of string type containing the
          name of the csv files to be loaded as datasets.
    Returns:
        results:
          A list containing the corresponding datasets of type numpy array.
    """
    print('Loading Datasets from csv via Multiprocessing.')

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(csv2dataset, filenames)

    return results


def get_UNET_AE_loaders(data_distribution, batch_size=32, shuffle=True, num_workers=1):
    """The get_UNET_AE_loaders retrieves the loaders of PyTorch-type DataLoader to
    automatically feed datasets to the UNET_AE model.

    Args:
        data_distribution:
          Object of string type to differentiate between loading couette, kvs,
          both or random valued datasets:
          ['get_couette', 'get_KVS', 'get_both', 'get_random']
        num_workers:
          Object of integer type that will turn on multi-process data loading
          with the specified number of loader worker processes.
        batch_size:
          Object of integer type that specifies the batch size.
        shuffle:
          Object of boolean type used to turn data shuffling on.

    Returns:
        _dataloaders_train:
          Object of PyTorch-type DataLoader to automatically feed training datasets.
        _dataloaders_valid:
          Object of PyTorch-type DataLoader to automatically feed validation datasets.
    """
    _batch_size = batch_size
    _shuffle = shuffle
    _num_workers = num_workers
    _data_tag = ''

    if _shuffle is True:
        switch = 'on'
    elif _shuffle is False:
        switch = 'off'
        _batch_size = 1

    if data_distribution == "get_couette":
        _data_tag = 'Couette'
    elif data_distribution == "get_KVS":
        _data_tag = 'KVS'
    elif data_distribution == "get_both":
        _data_tag = 'Couette and KVS'
    elif data_distribution == "get_random":
        _data_tag = 'random'

    print('------------------------------------------------------------')
    print('                      Loader Summary                        ')
    print(f'Data Dist. \t= {_data_tag}')
    print(f'Batch size\t= {_batch_size}')
    print(f'Num worker\t= {_num_workers}')
    print(f'Shuffle\t\t= {switch}')

    _data_train = []
    _data_valid = []
    _dataloaders_train = []
    _dataloaders_valid = []
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata/'

    if _data_tag == 'Couette':
        _train_files = glob.glob(f"{_directory}CleanCouette/Training/*.csv")
        _valid_files = glob.glob(f"{_directory}CleanCouette/Validation/*.csv")
    elif _data_tag == 'KVS':
        _train_files = glob.glob(f"{_directory}CleanKVS/Training/*.csv")
        _valid_files = glob.glob(f"{_directory}CleanKVS/Validation/*.csv")
    elif _data_tag == 'Couette and KVS':
        _train_files = glob.glob(f"{_directory}CleanCouette/Training/*.csv") + \
                                 glob.glob(
                                     f"{_directory}CleanKVS/Training/*.csv")
        _valid_files = glob.glob(f"{_directory}CleanCouette/Validation/*.csv") + \
            glob.glob(f"{_directory}CleanKVS/Validation/*.csv")
    elif _data_tag == 'random':
        print('Loading ---> RANDOM <--- training datasets as loader.')
        for i in range(3):
            _data = np.random.rand(1000, 3, 26, 26, 26)
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            _data_train.append(_data)
        print('Completed loading ---> RANDOM <--- training datasets.')

        print('Loading ---> RANDOM <--- validation datasets as loader.')
        for i in range(1):
            _data = np.random.rand(1000, 3, 26, 26, 26)
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            _data_valid.append(_data)
        print('Completed loading ---> RANDOM <--- validation datasets.')

        for _data in _data_train:
            _dataset = MyMamicoDataset_UNET_AE(_data)
            _dataloader = DataLoader(
                dataset=_dataset,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_train.append(_dataloader)

        for _data in _data_valid:
            _dataset = MyMamicoDataset_UNET_AE(_data)
            _dataloader = DataLoader(
                dataset=_dataset,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_valid.append(_dataloader)

        print(f'Num Train Loaders = {len(_dataloaders_train)}')
        print(f'Num Valid Loaders = {len(_dataloaders_valid)}')
        return _dataloaders_train, _dataloaders_valid
    else:
        print('Invalid value for function parameter: data_distribution.')
        return

    _data_train = mamico_csv2dataset_mp(_train_files)
    _data_valid = mamico_csv2dataset_mp(_valid_files)

    if _shuffle is True:
        _data_train_stack = np.vstack(_data_train)
        _data_valid_stack = np.vstack(_data_valid)

        _dataset_train = MyMamicoDataset_UNET_AE(_data_train_stack)
        _dataloader_train = DataLoader(
            dataset=_dataset_train,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )

        _dataset_valid = MyMamicoDataset_UNET_AE(_data_valid_stack)
        _dataloader_valid = DataLoader(
            dataset=_dataset_valid,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )

        print(f'Num Train Loaders = {len([_dataloader_train])}')
        print(f'Num Valid Loaders = {len([_dataloader_valid])}')
        return [_dataloader_train], [_dataloader_valid]

    for _data in _data_train:
        _dataset = MyMamicoDataset_UNET_AE(_data)
        _dataloader = DataLoader(
            dataset=_dataset,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )
        _dataloaders_train.append(_dataloader)

    for _data in _data_valid:
        _dataset = MyMamicoDataset_UNET_AE(_data)
        _dataloader = DataLoader(
            dataset=_dataset,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )
        _dataloaders_valid.append(_dataloader)

    print(f'Num Train Loaders = {len(_dataloaders_train)}')
    print(f'Num Valid Loaders = {len(_dataloaders_valid)}')
    return _dataloaders_train, _dataloaders_valid


def get_RNN_loaders(data_distribution, batch_size=32, seq_length=15, shuffle=False):
    """The get_RNN_loaders retrieves the loaders of PyTorch-type DataLoader to
    automatically feed datasets to the RNN models.

    Args:
        data_distribution:
          Object of string type to differentiate between loading couette, kvs,
          both or random valued datasets:
          ['get_couette', 'get_KVS', 'get_both', 'get_random']
          with the specified number of loader worker processes.
        batch_size:
          Object of integer type that specifies the batch size.
        sequence_length:
          Object of integer type specifying the number of elements to include
          in the RNN sequence.

    Returns:
        _dataloaders_train:
          Object of PyTorch-type DataLoader to automatically feed training datasets.
        _dataloaders_valid:
          Object of PyTorch-type DataLoader to automatically feed validation datasets.
    """
    _batch_size = batch_size
    _shuffle = shuffle
    _num_workers = 1
    _data_tag = ''

    if _shuffle is True:
        switch = 'on'
    elif _shuffle is False:
        switch = 'off'
        _batch_size = 1

    if data_distribution == "get_couette":
        _data_tag = 'Couette'
    elif data_distribution == "get_KVS":
        _data_tag = 'KVS'
    elif data_distribution == "get_both":
        _data_tag = 'Couette and KVS'
    elif data_distribution == "get_AE_KVS":
        _data_tag = 'non UNET KVS'
    elif data_distribution == "get_random":
        _data_tag = 'random'

    print('------------------------------------------------------------')
    print('                      Loader Summary                        ')
    print(f'Data Dist. \t= {_data_tag}')
    print(f'Batch size\t= {_batch_size}')
    print(f'Num worker\t= {_num_workers}')
    print(f'Shuffle\t\t= {switch}')

    _data_train = []
    _data_valid = []
    _train_files = []
    _valid_files = []
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata/'

    if _data_tag == 'Couette':
        _train_files = glob.glob(f"{_directory}CleanCouetteLS/Training/*.csv")
        _valid_files = glob.glob(
            f"{_directory}CleanCouetteLS/Validation/*.csv")
    elif _data_tag == 'KVS':
        _train_files = glob.glob(f"{_directory}CleanKVSLS/Training/*.csv")
        _valid_files = glob.glob(f"{_directory}CleanKVSLS/Validation/*.csv")
    elif _data_tag == 'Couette and KVS':
        _train_files = glob.glob(f"{_directory}CleanBothLS/Training/*.csv")
        _valid_files = glob.glob(f"{_directory}CleanBothLS/Validation/*.csv")
    elif _data_tag == 'non UNET KVS':
        _train_files = glob.glob(f"{_directory}CleanKVS_AE_LS/Training/*.csv")
        _valid_files = glob.glob(
            f"{_directory}CleanKVS_AE_LS/Validation/*.csv")
    elif _data_tag == 'random':
        print('Loading ---> RANDOM <--- training datasets as loader.')
        for i in range(3):
            data = np.random.rand(1000, 256)
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            _data_train.append(data)
        print('Completed loading ---> RANDOM <--- training datasets.')

        print('Loading ---> RANDOM <--- validation datasets as loader.')
        for i in range(1):
            data = np.random.rand(1000, 256)
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            _data_valid.append(data)
        print('Completed loading ---> RANDOM <--- validation datasets.')

        _dataloaders_train = []
        _dataloaders_valid = []

        for _data in _data_train:
            _dataset = MyMamicoDataset_RNN(_data, seq_length)
            _dataloader = DataLoader(
                dataset=_dataset,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_train.append(_dataloader)

        for _data in _data_valid:
            _dataset = MyMamicoDataset_RNN(_data, seq_length)
            _dataloader = DataLoader(
                dataset=_dataset,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_valid.append(_dataloader)

        print(f'Num Train Loaders = {len(_dataloaders_train)}')
        print(f'Num Valid Loaders = {len(_dataloaders_valid)}')
        return _dataloaders_train, _dataloaders_valid
    else:
        print('Invalid value for function parameter: data_distribution.')
        return

    _data_train = csv2dataset_mp(_train_files)
    _data_valid = csv2dataset_mp(_valid_files)

    if _shuffle is True:
        _data_train_stack = np.vstack(_data_train)
        _data_valid_stack = np.vstack(_data_valid)

        _dataset_train = MyMamicoDataset_RNN(_data_train_stack, seq_length)
        _dataloader_train = DataLoader(
            dataset=_dataset_train,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )

        _dataset_valid = MyMamicoDataset_RNN(_data_train_stack, seq_length)
        _dataloader_valid = DataLoader(
            dataset=_dataset_valid,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )

        print(f'Num Train Loaders = {len([_dataloader_train])}')
        print(f'Num Valid Loaders = {len([_dataloader_valid])}')
        return [_dataloader_train], [_dataloader_valid]

    _dataloaders_train = []
    _dataloaders_valid = []

    for _data in _data_train:
        _dataset = MyMamicoDataset_RNN(_data, seq_length)
        _dataloader = DataLoader(
            dataset=_dataset,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
        )
        _dataloaders_train.append(_dataloader)

    for _data in _data_valid:
        _dataset = MyMamicoDataset_RNN(_data, seq_length)
        _dataloader = DataLoader(
            dataset=_dataset,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
        )
        _dataloaders_valid.append(_dataloader)

    print(f'Num Train Loaders = {len(_dataloaders_train)}')
    print(f'Num Valid Loaders = {len(_dataloaders_valid)}')
    return _dataloaders_train, _dataloaders_valid


def get_RNN_loaders_analysis_2(data_distribution, batch_size=32, seq_length=15, shuffle=False):
    """The get_RNN_loaders_analysis_2 retrieves the loaders of PyTorch-type DataLoader to
    automatically feed datasets to the RNN models.

    Args:
        data_distribution:
          Object of string type to differentiate between loading couette, kvs,
          both or random valued datasets:
          ['get_couette', 'get_KVS', 'get_both', 'get_random']
          with the specified number of loader worker processes.
        batch_size:
          Object of integer type that specifies the batch size.
        sequence_length:
          Object of integer type specifying the number of elements to include
          in the RNN sequence.

    Returns:
        _dataloaders_train:
          Object of PyTorch-type DataLoader to automatically feed training datasets.
        _dataloaders_valid:
          Object of PyTorch-type DataLoader to automatically feed validation datasets.
    """
    _batch_size = batch_size
    _shuffle = shuffle
    _num_workers = 1
    _data_tag = ''

    if _shuffle is True:
        switch = 'on'
    elif _shuffle is False:
        switch = 'off'
        _batch_size = 1

    if data_distribution == "get_couette":
        _data_tag = 'Couette'
    elif data_distribution == "get_KVS":
        _data_tag = 'KVS'
    elif data_distribution == "get_both":
        _data_tag = 'Couette and KVS'
    elif data_distribution == "get_AE_KVS":
        _data_tag = 'non UNET KVS'
    elif data_distribution == "get_random":
        _data_tag = 'random'

    print('------------------------------------------------------------')
    print('                      Loader Summary                        ')
    print(f'Data Dist. \t= {_data_tag}')
    print(f'Batch size\t= {_batch_size}')
    print(f'Num worker\t= {_num_workers}')
    print(f'Shuffle\t\t= {switch}')

    _data_train = []
    _data_valid = []
    _train_files = []
    _valid_files = []
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata/'

    if _data_tag == 'Couette':
        _train_files = glob.glob(f"{_directory}CleanCouetteLS/Training/*.csv")
        _valid_files = glob.glob(
            f"{_directory}CleanCouetteLS/Validation/*.csv")
    elif _data_tag == 'KVS':
        _train_files = glob.glob(f"{_directory}CleanKVSLS/Training/*.csv")
        _valid_files = glob.glob(f"{_directory}CleanKVSLS/Validation/*.csv")
    elif _data_tag == 'Couette and KVS':
        _train_files = glob.glob(f"{_directory}CleanBothLS/Training/*.csv")
        _valid_files = glob.glob(f"{_directory}CleanBothLS/Validation/*.csv")
    elif _data_tag == 'non UNET KVS':
        _train_files = glob.glob(f"{_directory}CleanKVS_AE_LS/Training/*.csv")
        _valid_files = glob.glob(
            f"{_directory}CleanKVS_AE_LS/Validation/*.csv")
    elif _data_tag == 'random':
        print('Loading ---> RANDOM <--- training datasets as loader.')
        for i in range(3):
            data = np.random.rand(1000, 256)
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            _data_train.append(data)
        print('Completed loading ---> RANDOM <--- training datasets.')

        print('Loading ---> RANDOM <--- validation datasets as loader.')
        for i in range(1):
            data = np.random.rand(1000, 256)
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            _data_valid.append(data)
        print('Completed loading ---> RANDOM <--- validation datasets.')

        _dataloaders_train = []
        _dataloaders_valid = []

        for _data in _data_train:
            _dataset = MyMamicoDataset_RNN_analysis(_data, seq_length)
            _dataloader = DataLoader(
                dataset=_dataset,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_train.append(_dataloader)

        for _data in _data_valid:
            _dataset = MyMamicoDataset_RNN_analysis(_data, seq_length)
            _dataloader = DataLoader(
                dataset=_dataset,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_valid.append(_dataloader)

        print(f'Num Train Loaders = {len(_dataloaders_train)}')
        print(f'Num Valid Loaders = {len(_dataloaders_valid)}')
        return _dataloaders_train, _dataloaders_valid
    else:
        print('Invalid value for function parameter: data_distribution.')
        return

    _data_train = csv2dataset_mp(_train_files)
    _data_valid = csv2dataset_mp(_valid_files)

    if _shuffle is True:
        _data_train_stack = np.vstack(_data_train)
        _data_valid_stack = np.vstack(_data_valid)

        _dataset_train = MyMamicoDataset_RNN_analysis(
            _data_train_stack, seq_length)
        _dataloader_train = DataLoader(
            dataset=_dataset_train,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )

        _dataset_valid = MyMamicoDataset_RNN_analysis(
            _data_train_stack, seq_length)
        _dataloader_valid = DataLoader(
            dataset=_dataset_valid,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )

        print(f'Num Train Loaders = {len([_dataloader_train])}')
        print(f'Num Valid Loaders = {len([_dataloader_valid])}')
        return [_dataloader_train], [_dataloader_valid]

    _dataloaders_train = []
    _dataloaders_valid = []

    for _data in _data_train:
        _dataset = MyMamicoDataset_RNN_analysis(_data, seq_length)
        _dataloader = DataLoader(
            dataset=_dataset,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
        )
        _dataloaders_train.append(_dataloader)

    for _data in _data_valid:
        _dataset = MyMamicoDataset_RNN_analysis(_data, seq_length)
        _dataloader = DataLoader(
            dataset=_dataset,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
        )
        _dataloaders_valid.append(_dataloader)

    print(f'Num Train Loaders = {len(_dataloaders_train)}')
    print(f'Num Valid Loaders = {len(_dataloaders_valid)}')
    return _dataloaders_train, _dataloaders_valid


def get_Hybrid_loaders(data_distribution, batch_size=1, shuffle=False):
    """The get_Hybrid_loaders retrieves the loaders of PyTorch-type DataLoader to
    automatically feed datasets to the Hybrid_MD_RNN_UNET model. As such image
    and target are consecutive timesteps as opposed to identical timesteps as
    was the case for the autoencoder.

    Args:
        data_distribution:
          Object of string type to differentiate between loading couette, kvs,
          both or random valued datasets:
          ['get_couette', 'get_KVS', 'get_both', 'get_random']
        num_workers:
          Object of integer type that will turn on multi-process data loading
          with the specified number of loader worker processes.
        batch_size:
          Object of integer type that specifies the batch size.
        shuffle:
          Object of boolean type used to turn data shuffling on.

    Returns:
        _dataloaders_train:
          Object of PyTorch-type DataLoader to automatically feed training datasets.
        _dataloaders_valid:
          Object of PyTorch-type DataLoader to automatically feed validation datasets.
    """
    _batch_size = batch_size
    _shuffle = shuffle
    _num_workers = 1
    _data_tag = ''

    if _shuffle is True:
        switch = 'on'
    elif _shuffle is False:
        switch = 'off'
        _batch_size = 1

    if data_distribution == "get_couette":
        _data_tag = 'Couette'
    elif data_distribution == "get_KVS":
        _data_tag = 'KVS'
    elif data_distribution == "get_both":
        _data_tag = 'Couette and KVS'
    elif data_distribution == "get_random":
        _data_tag = 'random'

    print('------------------------------------------------------------')
    print('                      Loader Summary                        ')
    print(f'Data Dist. \t= {_data_tag}')
    print(f'Batch size\t= {_batch_size}')
    print(f'Num worker\t= {_num_workers}')
    print(f'Shuffle\t\t= {switch}')

    _data_train = []
    _data_valid = []
    _dataloaders_train = []
    _dataloaders_valid = []
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata/'

    if _shuffle is True:
        print('Shuffle is currently set to True and as such is invalid '
              'for a hybrid model.')
        return
    if _batch_size != 1:
        print('Invalid batch_size. Hybrid models can currently only deal '
              'with batch_size = 1')
        return

    if _data_tag == 'Couette':
        _train_files = glob.glob(f"{_directory}CleanCouette/Training/*.csv")
        _valid_files = glob.glob(f"{_directory}CleanCouette/Validation/*.csv")
    elif _data_tag == 'KVS':
        _train_files = glob.glob(f"{_directory}CleanKVS/Training/*.csv")
        _valid_files = glob.glob(f"{_directory}CleanKVS/Validation/*.csv")
    elif _data_tag == 'Couette and KVS':
        _train_files = glob.glob(f"{_directory}CleanCouette/Training/*.csv") + \
                                 glob.glob(
                                     f"{_directory}CleanKVS/Training/*.csv")
        _valid_files = glob.glob(f"{_directory}CleanCouette/Validation/*.csv") + \
            glob.glob(f"{_directory}CleanKVS/Validation/*.csv")
    elif _data_tag == 'random':
        print('Loading ---> RANDOM <--- training datasets as loader.')
        for i in range(3):
            _data = np.random.rand(1000, 3, 26, 26, 26)
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            _data_train.append(_data)
        print('Completed loading ---> RANDOM <--- training datasets.')

        print('Loading ---> RANDOM <--- validation datasets as loader.')
        for i in range(1):
            _data = np.random.rand(1000, 3, 26, 26, 26)
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            _data_valid.append(_data)
        print('Completed loading ---> RANDOM <--- validation datasets.')

        for _data in _data_train:
            _dataset = MyMamicoDataset_UNET_AE(_data)
            _dataloader = DataLoader(
                dataset=_dataset,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_train.append(_dataloader)

        for _data in _data_valid:
            _dataset = MyMamicoDataset_UNET_AE(_data)
            _dataloader = DataLoader(
                dataset=_dataset,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_valid.append(_dataloader)

        print(f'Num Train Loaders = {len(_dataloaders_train)}')
        print(f'Num Valid Loaders = {len(_dataloaders_valid)}')
        return _dataloaders_train, _dataloaders_valid
    else:
        print('Invalid value for function parameter: data_distribution.')
        return

    _data_train = mamico_csv2dataset_mp(_train_files)
    _data_valid = mamico_csv2dataset_mp(_valid_files)

    for _data in _data_train:
        _dataset = MyMamicoDataset_Hybrid(_data)
        _dataloader = DataLoader(
            dataset=_dataset,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )
        _dataloaders_train.append(_dataloader)

    for _data in _data_valid:
        _dataset = MyMamicoDataset_Hybrid(_data)
        _dataloader = DataLoader(
            dataset=_dataset,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )
        _dataloaders_valid.append(_dataloader)

    print(f'Num Train Loaders = {len(_dataloaders_train)}')
    print(f'Num Valid Loaders = {len(_dataloaders_valid)}')
    return _dataloaders_train, _dataloaders_valid


def get_testing_loaders(data_distribution, batch_size=1, shuffle=False, num_workers=1):
    """The get_testing_loaders retrieves the loaders of PyTorch-type DataLoader to
    automatically feed testing datasets to the UNET_AE model.

    Args:
        data_distribution:
          Object of string type to differentiate between loading couette or kvs
          ['get_couette', 'get_KVS']
        num_workers:
          Object of integer type that will turn on multi-process data loading
          with the specified number of loader worker processes.
        batch_size:
          Object of integer type that specifies the batch size.
        shuffle:
          Object of boolean type used to turn data shuffling on.

    Returns:
        _dataloaders_test:
          Object of PyTorch-type DataLoader to automatically feed training datasets.
    """
    _batch_size = batch_size
    _shuffle = shuffle
    _num_workers = num_workers
    _data_tag = ''

    if _shuffle is True:
        switch = 'on'
    elif _shuffle is False:
        switch = 'off'
        _batch_size = 1

    if data_distribution == "get_couette":
        _data_tag = 'Couette'
    elif data_distribution == "get_KVS":
        _data_tag = 'KVS'

    print('------------------------------------------------------------')
    print('                      Loader Summary                        ')
    print(f'Data Dist. \t= {_data_tag}')
    print(f'Batch size\t= {_batch_size}')
    print(f'Num worker\t= {_num_workers}')
    print(f'Shuffle\t\t= {switch}')

    _dataloaders_test = []
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata/'

    if _data_tag == 'Couette':
        _test_files = glob.glob(f"{_directory}CleanCouette/Testing/*.csv")
    elif _data_tag == 'KVS':
        _test_files = glob.glob(f"{_directory}CleanKVS/Testing/*.csv")

    else:
        print('Invalid value for function parameter: data_distribution.')
        return

    _data_test = mamico_csv2dataset_mp(_test_files)

    for _data in _data_test:
        _dataset = MyMamicoDataset_Hybrid(_data)
        _dataloader = DataLoader(
            dataset=_dataset,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )
        _dataloaders_test.append(_dataloader)

    print(f'Num Test Loaders = {len(_dataloaders_test)}')
    return _dataloaders_test


def losses2file(losses, file_name):
    """The losses2file function saves a list to a csv file.

    Args:
        losses:
          Object of type list containing the losses of interest.
        file_name:
          Object of string type containing the name of the csv file where the
          losses will be saved to.

    Returns:
        NONE:
          This function does not have a return value. Instead it saves the
          losses of interest to file.
    """

    print(f'Saving losses to file: {file_name}')
    np.savetxt(f"{file_name}.csv", losses, delimiter=", ", fmt='% s')


def check_save_load():
    """The check_save_load function is used to validate functionality of the
    dataset2csv and csv2dataset functions.

    Args:
        None:

    Returns:
        NONE:
    """
    input_array = np.random.rand(8, 3, 18, 18, 18)

    dataset2csv(
        dataset=input_array,
        model_descriptor='test_descriptor',
        dataset_name='preds',
        counter=1
    )

    loaded_array = csv2dataset(
        filename='preds_1_model_test_descriptor.csv',
        output_shape=(8, 3, 18, 18, 18)
    )

    print("shape of input array: ", input_array.shape)
    print("shape of loaded array: ", loaded_array.shape)

    if (input_array == loaded_array).all():
        print("Yes, both the arrays are same")
    else:
        print("No, both the arrays are not same")


def check_RNN_dataset_approach():
    """The check_RNN_dataset_approach function is used to validate functionality
    of batching the RNN sequences.

    Args:
        None:

    Returns:
        NONE:
    """
    dataset = torch.ones(1000, 256)
    print(dataset.shape)

    for i in range(1000):
        dataset[i] = dataset[i] * i

    print(dataset[0, 0])
    print(dataset[100, 0])
    print(dataset[-1, 0])

    sequence_length = 25

    my_masks = dataset[sequence_length:]
    print(my_masks.shape)

    rnn_images = torch.zeros(
        len(dataset)-sequence_length-1, sequence_length, 256)
    print(rnn_images.shape)
    for i in range(len(rnn_images)):
        rnn_images[i] = dataset[i:sequence_length+i]

    if torch.equal(rnn_images[5], dataset[5:sequence_length+5]):
        print('Inputs are equal.')
        print('Dataset:', dataset[5:sequence_length+5, 0])
        print('Images:', rnn_images[5, :, 0])
        print('Masks:', my_masks[5, 0])


if __name__ == "__main__":
    _directory = "/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata"
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
