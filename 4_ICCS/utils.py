import torch
import numpy as np
import csv
import glob
import torch.multiprocessing as mp
import concurrent.futures
from dataset import MyMamicoDataset_AE, MyMamicoDataset_RNN, MyMamicoDataset_RNN_verification, MyMamicoDataset_Hybrid
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    print('currently: mlready2dataset')
    print(f'Loading Dataset from csv: {file_name}')
    dataset = np.loadtxt(f'{file_name}')

    if dataset.size != (1000 * 3 * 26 * 26 * 26):
        print("Incorrect dimensions:", dataset.size, file_name)
        return

    t, c, d, h, w = (1000, 3, 26, 26, 26)

    original_dataset = dataset.reshape(t, c, d, h, w)
    original_dataset = original_dataset[100:]
    print('Test - removed 100 timesteps? - ', original_dataset.shape)
    return original_dataset


def mlready2dataset_mp(file_names):
    """The mlready2dataset_mp function is used to call the mlready2dataset
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
    print('currently: mlready2dataset_mp')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        print('in concurrent')
        results = executor.map(mlready2dataset, file_names)
    return results


def mlready2latentspace(file_name):
    """The mlready2dataset function retrieves a numpy array from a csv file.

    Args:
        file_name:
          Object of string type containing the name of the csv file to be
          loaded as a dataset.

    Returns:
        dataset:
          Object of numpy array type containing the dataset read from file.
    """
    print('currently: mlready2dataset')
    print(f'Loading Dataset from csv: {file_name}')
    dataset = np.loadtxt(f'{file_name}')
    print(dataset.shape)

    if dataset.size != (900 * 32 * 2 * 2 * 2):
        print("Incorrect dimensions:", dataset.size, file_name)
        return

    t, c, d, h, w = (900, 32, 2, 2, 2)

    latentspace = dataset.reshape(t, c, d, h, w)
    print(latentspace.shape)
    return latentspace


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


def get_AE_loaders(data_distribution, batch_size=32, shuffle=True, num_workers=1):
    """The get_AE_loaders retrieves the loaders of PyTorch-type DataLoader
    to automatically feed datasets to the AE model.

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
        shuffle_switch = 'on'
    elif _shuffle is False:
        shuffle_switch = 'off'
        _batch_size = 1

    if data_distribution == "get_couette":
        _data_tag = 'Couette'
    elif data_distribution == "get_couette_eval":
        _data_tag = 'Couette_eval'
    elif data_distribution == "get_KVS":
        _data_tag = 'KVS'
    elif data_distribution == "get_KVS_eval":
        _data_tag = 'KVS_eval'
    elif data_distribution == "get_both":
        _data_tag = 'Couette and KVS'
    elif data_distribution == "get_random":
        _data_tag = 'random'

    print('------------------------------------------------------------')
    print('                      Loader Summary                        ')
    print('Cur. Loader\t : get_AE_loaders')
    print(f'Data Dist. \t= {_data_tag}')
    print(f'Batch size\t= {_batch_size}')
    print(f'Num worker\t= {_num_workers}')
    print(f'Shuffle\t\t= {shuffle_switch}')

    _data_train = []
    _data_valid = []
    _dataloaders_train = []
    _dataloaders_valid = []
    _directory = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/dataset_mlready/'

    if _data_tag == 'Couette':
        _train_files = glob.glob(f"{_directory}Couette/Training/.csv")
        _valid_files = glob.glob(
            f"{_directory}Couette/Validation/.csv")
    elif _data_tag == 'Couette_eval':
        _train_files = glob.glob(
            f"{_directory}KVS/Training/*bottom_0_oscil_2_5*.csv")
        _valid_files = glob.glob(
            f"{_directory}KVS/Validation/*bottom_0_oscil_3_0*.csv")
    elif _data_tag == 'KVS':
        _train_files = glob.glob(f"{_directory}KVS/Training/*.csv")
        # print(_train_files)
        _valid_files = glob.glob(f"{_directory}KVS/Validation/*.csv")
        # print(_valid_files)
    elif _data_tag == 'KVS_eval':
        _train_files = glob.glob(f"{_directory}KVS/Training/*20000_NW*.csv")
        _valid_files = glob.glob(f"{_directory}KVS/Validation/*28000_SW*.csv")
    elif _data_tag == 'Couette and KVS':
        _train_files = glob.glob(f"{_directory}Couette/Training/*.csv") + \
                                 glob.glob(
                                     f"{_directory}KVS/Training/*.csv")
        _valid_files = glob.glob(f"{_directory}Couette/Validation/*.csv") + \
            glob.glob(f"{_directory}KVS/Validation/*.csv")
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
            _dataset = MyMamicoDataset_AE(_data)
            _dataloader = DataLoader(
                dataset=_dataset,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_train.append(_dataloader)

        for _data in _data_valid:
            _dataset = MyMamicoDataset_AE(_data)
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
    print('successful data tag')

    if _shuffle is True:
        _data_train = mlready2dataset_mp(_train_files)
        _data_valid = mlready2dataset_mp(_valid_files)
        _data_train_stack = np.vstack(_data_train)
        print('Training stack shape:', _data_train_stack.shape)
        _data_valid_stack = np.vstack(_data_valid)

        _dataset_train = MyMamicoDataset_AE(_data_train_stack)
        _dataloader_train = DataLoader(
            dataset=_dataset_train,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )

        _dataset_valid = MyMamicoDataset_AE(_data_valid_stack)
        _dataloader_valid = DataLoader(
            dataset=_dataset_valid,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )

        print(f'Num Train Loaders = {len([_dataloader_train])}')
        print(f'Num Valid Loaders = {len([_dataloader_valid])}')
        return [_dataloader_train], [_dataloader_valid]

    else:
        for _file in _train_files:
            print(_file)
            _data_train = mlready2dataset(_file)
            _dataset_train = MyMamicoDataset_AE(_data_train)
            _dataloader_train = DataLoader(
                dataset=_dataset_train,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_train.append(_dataloader_train)
        for _file in _valid_files:
            print(_file)
            _data_valid = mlready2dataset(_file)
            _dataset_valid = MyMamicoDataset_AE(_data_valid)
            _dataloader_valid = DataLoader(
                dataset=_dataset_valid,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_valid.append(_dataloader_valid)

        print(f'Num Train Loaders = {len(_dataloaders_train)}')
        print(f'Num Valid Loaders = {len(_dataloaders_valid)}')
        return _dataloaders_train, _dataloaders_valid


def get_RNN_loaders(data_distribution, batch_size=32, shuffle=True, num_workers=1):
    """The get_RNN_loaders retrieves the loaders of PyTorch-type DataLoader
    to automatically feed datasets to the RNN model.

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
        shuffle_switch = 'on'
    elif _shuffle is False:
        shuffle_switch = 'off'
        _batch_size = 1

    if data_distribution == "get_KVS":
        _data_tag = 'KVS'

    elif data_distribution == "get_KVS_eval":
        _data_tag = 'KVS_eval'
    elif data_distribution == "get_random":
        _data_tag = 'random'

    print('------------------------------------------------------------')
    print('                      Loader Summary                        ')
    print('Cur. Loader\t : get_RNN_loaders')
    print(f'Data Dist. \t= {_data_tag}')
    print(f'Batch size\t= {_batch_size}')
    print(f'Num worker\t= {_num_workers}')
    print(f'Shuffle\t\t= {shuffle_switch}')

    _data_train_x = []
    _data_train_y = []
    _data_train_z = []

    _data_valid_x = []
    _data_valid_y = []
    _data_valid_z = []

    _dataloaders_train_x = []
    _dataloaders_train_y = []
    _dataloaders_train_z = []

    _dataloaders_valid_x = []
    _dataloaders_valid_y = []
    _dataloaders_valid_z = []

    _directory = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/dataset_mlready/'

    if _data_tag == 'KVS':
        _train_files_x = glob.glob(
            f"{_directory}KVS/Latentspace/Training/*.csv")
        _valid_files_x = glob.glob(
            f"{_directory}KVS/Latentspace/Validation/*.csv")

        for idx, _file in enumerate(_train_files_x):
            _data_train_x = mlready2latentspace(_train_files_x[idx])
            _dataset_x = MyMamicoDataset_RNN(_data_train_x)
            _dataloader_x = DataLoader(
                dataset=_dataset_x,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_train_x.append(_dataloader_x)
        for idx, _file in enumerate(_valid_files_x):
            _data_valid_x = mlready2latentspace(_valid_files_x[idx])
            _dataset_x = MyMamicoDataset_RNN(_data_valid_x)
            _dataloader_x = DataLoader(
                dataset=_dataset_x,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_valid_x.append(_dataloader_x)
        return _dataloaders_train_x, _dataloaders_valid_x
    elif _data_tag == 'KVS_eval':
        _train_files_x = glob.glob(
            f"{_directory}KVS/Latentspace/Training/*20000_NW_x.csv")
        _train_files_y = glob.glob(
            f"{_directory}KVS/Latentspace/Training/*20000_NW_y.csv")
        _train_files_z = glob.glob(
            f"{_directory}KVS/Latentspace/Training/*20000_NW_z.csv")
        _valid_files_x = glob.glob(
            f"{_directory}KVS/Latentspace/Validation/*28000_SW_x.csv")
        _valid_files_y = glob.glob(
            f"{_directory}KVS/Latentspace/Validation/*28000_SW_y.csv")
        _valid_files_z = glob.glob(
            f"{_directory}KVS/Latentspace/Validation/*28000_SW_z.csv")

    elif _data_tag == 'random':
        print('Loading ---> RANDOM <--- training datasets as loader.')
        _data_x = np.random.rand(1000, 32, 2, 2, 2)
        _data_train_x.append(_data_x)
        _data_y = np.random.rand(1000, 32, 2, 2, 2)
        _data_train_y.append(_data_y)
        _data_z = np.random.rand(1000, 32, 2, 2, 2)
        _data_train_z.append(_data_z)
        print('Completed loading ---> RANDOM <--- training datasets.')

        print('Loading ---> RANDOM <--- validation datasets as loader.')
        _data_x = np.random.rand(1000, 32, 2, 2, 2)
        _data_valid_x.append(_data_x)
        _data_y = np.random.rand(1000, 32, 2, 2, 2)
        _data_valid_y.append(_data_y)
        _data_z = np.random.rand(1000, 32, 2, 2, 2)
        _data_valid_z.append(_data_z)
        print('Completed loading ---> RANDOM <--- validation datasets.')

        for idx, _data in enumerate(_data_train_x):
            _dataset_x = MyMamicoDataset_RNN(_data_train_x[idx])
            _dataloader_x = DataLoader(
                dataset=_dataset_x,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_train_x.append(_dataloader_x)

            _dataset_y = MyMamicoDataset_RNN(_data_train_y[idx])
            _dataloader_y = DataLoader(
                dataset=_dataset_y,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_train_y.append(_dataloader_y)

            _dataset_z = MyMamicoDataset_RNN(_data_train_z[idx])
            _dataloader_z = DataLoader(
                dataset=_dataset_z,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_train_z.append(_dataloader_z)

        for idx, _data in enumerate(_data_valid_x):
            _dataset_x = MyMamicoDataset_RNN_verification(_data_valid_x[idx])
            _dataloader_x = DataLoader(
                dataset=_dataset_x,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_valid_x.append(_dataloader_x)

            _dataset_y = MyMamicoDataset_RNN_verification(_data_valid_y[idx])
            _dataloader_y = DataLoader(
                dataset=_dataset_y,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_valid_y.append(_dataloader_y)

            _dataset_z = MyMamicoDataset_RNN_verification(_data_valid_z[idx])
            _dataloader_z = DataLoader(
                dataset=_dataset_z,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_valid_z.append(_dataloader_z)

        print(f'Num Train Loaders = {len(_dataloaders_train_x)}')
        print(f'Num Valid Loaders = {len(_dataloaders_valid_x)}')
        return _dataloaders_train_x, _dataloaders_train_y, _dataloaders_train_z, _dataloaders_valid_x, _dataloaders_valid_y, _dataloaders_valid_z
    else:
        print('Invalid value for function parameter: data_distribution.')
        return

    print('successful data tag')

    for idx, _file in enumerate(_train_files_x):
        _data_train_x = mlready2latentspace(_train_files_x[idx])
        _dataset_x = MyMamicoDataset_RNN_verification(_data_train_x)
        _dataloader_x = DataLoader(
            dataset=_dataset_x,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )
        _dataloaders_train_x.append(_dataloader_x)

        _data_train_y = mlready2latentspace(_train_files_y[idx])
        _dataset_y = MyMamicoDataset_RNN_verification(_data_train_y)
        _dataloader_y = DataLoader(
            dataset=_dataset_y,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )
        _dataloaders_train_y.append(_dataloader_y)

        _data_train_z = mlready2latentspace(_train_files_z[idx])
        _dataset_z = MyMamicoDataset_RNN_verification(_data_train_z)
        _dataloader_z = DataLoader(
            dataset=_dataset_z,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )
        _dataloaders_train_z.append(_dataloader_z)

    for idx, _file in enumerate(_valid_files_x):
        _data_valid_x = mlready2latentspace(_valid_files_x[idx])
        _dataset_x = MyMamicoDataset_RNN_verification(_data_valid_x)
        _dataloader_x = DataLoader(
            dataset=_dataset_x,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )
        _dataloaders_valid_x.append(_dataloader_x)

        _data_valid_y = mlready2latentspace(_valid_files_y[idx])
        _dataset_y = MyMamicoDataset_RNN_verification(_data_valid_y)
        _dataloader_y = DataLoader(
            dataset=_dataset_y,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )
        _dataloaders_valid_y.append(_dataloader_y)

        _data_valid_z = mlready2latentspace(_valid_files_z[idx])
        _dataset_z = MyMamicoDataset_RNN_verification(_data_valid_z)
        _dataloader_z = DataLoader(
            dataset=_dataset_z,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )
        _dataloaders_valid_z.append(_dataloader_z)

    return _dataloaders_train_x, _dataloaders_train_y, _dataloaders_train_z, _dataloaders_valid_x, _dataloaders_valid_y, _dataloaders_valid_z


def get_Hybrid_loaders(data_distribution, batch_size=1, shuffle=False, num_workers=1):
    """The get_Hybrid_loaders retrieves the loaders of PyTorch-type DataLoader
    to automatically feed datasets to the Hybrid model.

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
        shuffle_switch = 'on'
    elif _shuffle is False:
        shuffle_switch = 'off'
        _batch_size = 1

    if data_distribution == "get_KVS":
        _data_tag = 'KVS'
    elif data_distribution == "get_KVS_eval":
        _data_tag = 'KVS_eval'

    print('------------------------------------------------------------')
    print('                      Loader Summary                        ')
    print('Cur. Loader\t : get_AE_loaders')
    print(f'Data Dist. \t= {_data_tag}')
    print(f'Batch size\t= {_batch_size}')
    print(f'Num worker\t= {_num_workers}')
    print(f'Shuffle\t\t= {shuffle_switch}')

    _data_train = []
    _data_valid = []
    _dataloaders_train = []
    _dataloaders_valid = []
    _directory = '/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/dataset_mlready/'

    if _data_tag == 'KVS':
        _train_files = glob.glob(f"{_directory}KVS/Training/*.csv")
        _valid_files = glob.glob(f"{_directory}KVS/Validation/*.csv")
    elif _data_tag == 'KVS_eval':
        _train_files = glob.glob(f"{_directory}KVS/Training/*20000_NW*.csv")
        print(_train_files)
        _valid_files = glob.glob(f"{_directory}KVS/Validation/*20000_NE*.csv")
    else:
        print('Invalid value for function parameter: data_distribution.')
        return

    print('successful data tag')

    if _shuffle is True:
        _data_train = mlready2dataset_mp(_train_files)
        _data_valid = mlready2dataset_mp(_valid_files)
        _data_train_stack = np.vstack(_data_train)
        print('Training stack shape:', _data_train_stack.shape)
        _data_valid_stack = np.vstack(_data_valid)

        _dataset_train = MyMamicoDataset_AE(_data_train_stack)
        _dataloader_train = DataLoader(
            dataset=_dataset_train,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )

        _dataset_valid = MyMamicoDataset_AE(_data_valid_stack)
        _dataloader_valid = DataLoader(
            dataset=_dataset_valid,
            batch_size=_batch_size,
            shuffle=_shuffle,
            num_workers=_num_workers
            )

        print(f'Num Train Loaders = {len([_dataloader_train])}')
        print(f'Num Valid Loaders = {len([_dataloader_valid])}')
        return [_dataloader_train], [_dataloader_valid]

    else:
        for _file in _train_files:
            print(_file)
            _data_train = mlready2dataset(_file)
            _dataset_train = MyMamicoDataset_Hybrid(_data_train)
            _dataloader_train = DataLoader(
                dataset=_dataset_train,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_train.append(_dataloader_train)
        for _file in _valid_files:
            print(_file)
            _data_valid = mlready2dataset(_file)
            _dataset_valid = MyMamicoDataset_Hybrid(_data_valid)
            _dataloader_valid = DataLoader(
                dataset=_dataset_valid,
                batch_size=_batch_size,
                shuffle=_shuffle,
                num_workers=_num_workers
                )
            _dataloaders_valid.append(_dataloader_valid)

        print(f'Num Train Loaders = {len(_dataloaders_train)}')
        print(f'Num Valid Loaders = {len(_dataloaders_valid)}')
        return _dataloaders_train, _dataloaders_valid


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


if __name__ == '__main__':

    pass
