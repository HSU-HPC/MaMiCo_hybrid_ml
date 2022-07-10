import torch
import numpy as np
import time
import csv
import torch.multiprocessing as mp
import concurrent.futures
from dataset import MyMamicoDataset, MyMamicoDataset_UNET_AE, MyMamicoDataset_RNN
from torch.utils.data import DataLoader
from model import UNET_AE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clean_mamico_data(directory, filename):
    #
    # This function is used to clean the MaMiCo generated
    # csv file. In other words, to remove the comma delimiter
    # and ensure a semicolon delimiter.
    #
    text = open(f"{directory}/{filename}", "r")
    text = ''.join([i for i in text]) \
        .replace(",", ";")
    x = open(f"{directory}/clean_{filename}", "w")
    x.writelines(text)
    x.close()
    pass


def mamico_csv2dataset(file_name):
    #
    # This function reads from a MaMiCo generatd csv file.
    # Currently, proper functionality is hardcoded for simulations
    # containing 1000 timesteps.
    #
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata'
    dataset = np.zeros((1000, 3, 26, 26, 26))

    with open(f'{_directory}/{file_name}') as csvfile:
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
    #
    # This function reads from a MaMiCo generatd csv file.
    # Currently, proper functionality is hardcoded for simulations
    # containing 1000 timesteps.
    #
    start = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(mamico_csv2dataset, file_names)

    duration = time.time() - start
    print(f'Loading Data via Multiprocessing takes: {duration:.3f} secs')
    return results


def dataset2csv(dataset, dataset_name,  model_descriptor=0, counter=''):
    #
    # This function reads from a MaMiCo generatd csv file.
    # Currently, proper functionality is hardcoded for simulations
    # containing 1000 timesteps.
    #
    # 1) Convert 3D array to 2D array
    dataset_reshaped = dataset.reshape(dataset.shape[0], -1)
    # 2) Save 2D array to file
    if model_descriptor == 0:
        name = f'{dataset_name}'
    else:
        name = f'{dataset_name}_{counter}_model_{model_descriptor}'
    np.savetxt(f'{name}.csv', dataset_reshaped)


def csv2dataset(filename, output_shape=0):
    # directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata'
    dataset = np.loadtxt(f'{filename}')
    print('Shape of dataset: ', dataset.shape)

    if output_shape == 0:
        return dataset

    t, c, d, h, w = output_shape

    # 4) Revert 2D array to 3D array
    original_dataset = dataset.reshape(t, c, d, h, w)
    return original_dataset


def csv2dataset_mp(filenames, output_shape=0):
    start = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(csv2dataset, filenames)

    duration = time.time() - start
    print(f'Loading Data via Multiprocessing takes: {duration:.3f} secs')
    return results


def get_UNET_AE_loaders(file_names=0, num_workers=12):
    #
    # This function creates the dataloaders needed to automatically
    # feed the neural networks with the input dataset. In particular,
    # this function vields a dataloader for each specified mamico generated
    # csv file. As for num_workers, the rule of thumb is = 4 * num_GPU.
    # Note that the variable file_names acts as a flag such that
    # if == 0 : load data via mp and return one train and valid loader
    # if == -1: load data via mp and return multiple train and valid loaders
    # else: load random data for testing
    _batch_size = 32
    _num_workers = 1
    print('------------------------------------------------------------')
    print(f'Batch size = {_batch_size}        Num worker = {_num_workers}')

    data_train = []
    data_valid = []
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata'

    if file_names == 0 or file_names == -1:
        _train_files = [
            'clean_couette_test_combined_domain_0_5_top.csv',
            'clean_couette_test_combined_domain_0_5_middle.csv',
            'clean_couette_test_combined_domain_0_5_bottom.csv',
            'clean_couette_test_combined_domain_1_0_top.csv',
            'clean_couette_test_combined_domain_1_0_middle.csv',
            'clean_couette_test_combined_domain_1_0_bottom.csv',
            'clean_couette_test_combined_domain_2_0_top.csv',
            'clean_couette_test_combined_domain_2_0_middle.csv',
            'clean_couette_test_combined_domain_2_0_bottom.csv',
            'clean_couette_test_combined_domain_4_0_top.csv',
            'clean_couette_test_combined_domain_4_0_middle.csv',
            'clean_couette_test_combined_domain_4_0_bottom.csv',
        ]

        _valid_files = [
            'clean_couette_test_combined_domain_3_0_top.csv',
            'clean_couette_test_combined_domain_3_0_middle.csv',
            'clean_couette_test_combined_domain_3_0_bottom.csv',
            'clean_couette_test_combined_domain_5_0_top.csv',
            'clean_couette_test_combined_domain_5_0_middle.csv',
            'clean_couette_test_combined_domain_5_0_bottom.csv'
        ]

        start_time = time.time()
        print('Loading training data.')
        data_train = mamico_csv2dataset_mp(_train_files)
        duration = time.time() - start_time
        print(
            f'Completed loading training data. Duration: {duration:.3f}')

        start_time = time.time()
        print('Loading validation data.')
        data_valid = mamico_csv2dataset_mp(_valid_files)
        duration = time.time() - start_time
        print(
            f'Completed loading validation data. Duration: {duration:.3f}')

        if file_names == -1:
            dataloaders_train = []
            dataloaders_valid = []

            for dataset in data_train:
                dataset = MyMamicoDataset_UNET_AE(dataset)
                dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=_batch_size,
                    shuffle=False,
                    num_workers=_num_workers
                    )
                dataloaders_train.append(dataloader)

            for dataset in data_valid:
                dataset = MyMamicoDataset_UNET_AE(dataset)
                dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=_batch_size,
                    shuffle=False,
                    num_workers=_num_workers
                    )
                dataloaders_valid.append(dataloader)
            return dataloaders_train, dataloaders_valid

    elif file_names == -2:
        _train_files = [
            'clean_kvs_10K_NE_combined_domain.csv',
            'clean_kvs_10K_NW_combined_domain.csv',
            'clean_kvs_10K_SE_combined_domain.csv',
            'clean_kvs_10K_SW_combined_domain.csv',
            'clean_kvs_20K_NE_combined_domain.csv',
            'clean_kvs_20K_NW_combined_domain.csv',
            'clean_kvs_20K_SE_combined_domain.csv',
            'clean_kvs_20K_SW_combined_domain.csv',
            'clean_kvs_30K_NE_combined_domain.csv',
            'clean_kvs_30K_NW_combined_domain.csv',
            'clean_kvs_30K_SE_combined_domain.csv',
            'clean_kvs_30K_SW_combined_domain.csv',
        ]

        _valid_files = [
            'clean_kvs_40K_NE_combined_domain.csv',
            'clean_kvs_40K_NW_combined_domain.csv',
            'clean_kvs_40K_SE_combined_domain.csv',
            'clean_kvs_40K_SW_combined_domain.csv',
        ]

        start_time = time.time()
        print('Loading training data.')
        data_train = mamico_csv2dataset_mp(_train_files)
        duration = time.time() - start_time
        print(
            f'Completed loading training data. Duration: {duration:.3f}')

        start_time = time.time()
        print('Loading validation data.')
        data_valid = mamico_csv2dataset_mp(_valid_files)
        duration = time.time() - start_time
        print(
            f'Completed loading validation data. Duration: {duration:.3f}')

        dataloaders_train = []
        dataloaders_valid = []

        for dataset in data_train:
            dataset = MyMamicoDataset_UNET_AE(dataset)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=_batch_size,
                shuffle=True,
                num_workers=_num_workers
                )
            dataloaders_train.append(dataloader)

        for dataset in data_valid:
            dataset = MyMamicoDataset_UNET_AE(dataset)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=_batch_size,
                shuffle=True,
                num_workers=_num_workers
                )
            dataloaders_valid.append(dataloader)
        return dataloaders_train, dataloaders_valid

    else:
        print('Loading ---> RANDOM <--- training datasets as loader.')
        for i in range(3):
            data = np.random.rand(1000, 3, 26, 26, 26)
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            data_train.append(data)
        print('Completed loading ---> RANDOM <--- training datasets.')

        print('Loading ---> RANDOM <--- validation datasets as loader.')
        for i in range(1):
            data = np.random.rand(1000, 3, 26, 26, 26)
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            data_valid.append(data)
        print('Completed loading ---> RANDOM <--- validation datasets.')

        dataloaders_train = []
        dataloaders_valid = []

        for dataset in data_train:
            dataset = MyMamicoDataset_UNET_AE(dataset)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=32,
                shuffle=True,
                num_workers=num_workers
                )
            dataloaders_train.append(dataloader)

        for dataset in data_valid:
            dataset = MyMamicoDataset_UNET_AE(dataset)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=32,
                shuffle=False,
                num_workers=num_workers
                )
            dataloaders_valid.append(dataloader)
        return dataloaders_train, dataloaders_valid

    data_valid_stack = np.vstack(data_valid)
    dataset_valid = MyMamicoDataset_UNET_AE(data_valid_stack)
    dataloader_valid = DataLoader(
        dataset=dataset_valid,
        batch_size=_batch_size,
        shuffle=False,
        num_workers=_num_workers
        )

    data_train_stack = np.vstack(data_train)
    dataset_train = MyMamicoDataset_UNET_AE(data_train_stack)
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=_batch_size,
        shuffle=False,
        num_workers=_num_workers
        )

    return [dataloader_train], [dataloader_valid]


def get_RNN_loaders(file_names=0, sequence_length=15, num_workers=4):
    #
    # This function creates the dataloaders needed to automatically
    # feed the neural networks with the input dataset. In particular,
    # this function vields a dataloader for each specified mamico generated
    # csv file. As for num_workers, the rule of thumb is = 4 * num_GPU.
    # Note that the variable file_names acts as a flag such that
    # if == 0 : load data via mp and return one train and valid loader
    # if == -1: load data via mp and return multiple train and valid loaders
    # else: load random data for testing
    _num_workers = 1
    _batch_size = 16

    data_train = []
    data_valid = []
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata/'

    if file_names == 0:
        _train_files = [
            f'{_directory}Latentspace_Dataset_0_5_B.csv',
            f'{_directory}Latentspace_Dataset_0_5_M.csv',
            f'{_directory}Latentspace_Dataset_0_5_T.csv',
            f'{_directory}Latentspace_Dataset_1_0_B.csv',
            f'{_directory}Latentspace_Dataset_1_0_M.csv',
            f'{_directory}Latentspace_Dataset_1_0_T.csv',
            f'{_directory}Latentspace_Dataset_2_0_B.csv',
            f'{_directory}Latentspace_Dataset_2_0_M.csv',
            f'{_directory}Latentspace_Dataset_2_0_T.csv',
            f'{_directory}Latentspace_Dataset_4_0_B.csv',
            f'{_directory}Latentspace_Dataset_4_0_M.csv',
            f'{_directory}Latentspace_Dataset_4_0_T.csv',
        ]

        _valid_files = [
            f'{_directory}Latentspace_Dataset_3_0_B.csv',
            f'{_directory}Latentspace_Dataset_3_0_M.csv',
            f'{_directory}Latentspace_Dataset_3_0_T.csv',
            f'{_directory}Latentspace_Dataset_5_0_B.csv',
            f'{_directory}Latentspace_Dataset_5_0_M.csv',
            f'{_directory}Latentspace_Dataset_5_0_T.csv',
        ]
        start_time = time.time()
        print('Loading training data.')
        data_train = csv2dataset_mp(_train_files)
        duration = time.time() - start_time
        print(
            f'Completed loading training data. Duration: {duration:.3f}')

        start_time = time.time()
        print('Loading validation data.')
        data_valid = csv2dataset_mp(_valid_files)
        duration = time.time() - start_time
        print(f'Completed loading validation data. Duration: {duration:.3f}')

    elif file_names == -2:
        _train_files = [
            f'{_directory}Latentspace_Dataset_kvs_10K_NE.csv',
            f'{_directory}Latentspace_Dataset_kvs_10K_NW.csv',
            f'{_directory}Latentspace_Dataset_kvs_10K_SE.csv',
            f'{_directory}Latentspace_Dataset_kvs_10K_SW.csv',
            f'{_directory}Latentspace_Dataset_kvs_20K_NE.csv',
            f'{_directory}Latentspace_Dataset_kvs_20K_NW.csv',
            f'{_directory}Latentspace_Dataset_kvs_20K_SE.csv',
            f'{_directory}Latentspace_Dataset_kvs_20K_SW.csv',
            f'{_directory}Latentspace_Dataset_kvs_30K_NE.csv',
            f'{_directory}Latentspace_Dataset_kvs_30K_NW.csv',
            f'{_directory}Latentspace_Dataset_kvs_30K_SE.csv',
            f'{_directory}Latentspace_Dataset_kvs_30K_SW.csv',
        ]

        _valid_files = [
            f'{_directory}Latentspace_Dataset_kvs_40K_NE.csv',
            f'{_directory}Latentspace_Dataset_kvs_20K_NW.csv',
            f'{_directory}Latentspace_Dataset_kvs_40K_SE.csv',
            f'{_directory}Latentspace_Dataset_kvs_40K_SW.csv',
        ]
        start_time = time.time()
        print('Loading training data.')
        data_train = csv2dataset_mp(_train_files)
        duration = time.time() - start_time
        print(
            f'Completed loading training data. Duration: {duration:.3f}')

        start_time = time.time()
        print('Loading validation data.')
        data_valid = csv2dataset_mp(_valid_files)
        duration = time.time() - start_time
        print(f'Completed loading validation data. Duration: {duration:.3f}')
    else:
        print('Loading ---> RANDOM <--- training datasets as loader.')
        for i in range(5):
            data = np.random.rand(1000, 256)
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            data_train.append(data)
        print('Completed loading ---> RANDOM <--- training datasets.')

        print('Loading ---> RANDOM <--- validation datasets as loader.')
        for i in range(3):
            data = np.random.rand(1000, 256)
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            data_valid.append(data)
        print('Completed loading ---> RANDOM <--- validation datasets.')

    dataloaders_train = []
    dataloaders_valid = []

    for data in data_train:
        _dataset = MyMamicoDataset_RNN(data, sequence_length)
        _dataloader = DataLoader(
            dataset=_dataset,
            batch_size=_batch_size,
            shuffle=False,
            num_workers=_num_workers
        )
        dataloaders_train.append(_dataloader)

    for data in data_valid:
        _dataset = MyMamicoDataset_RNN(data, sequence_length)
        _dataloader = DataLoader(
            dataset=_dataset,
            batch_size=_batch_size,
            shuffle=False,
            num_workers=_num_workers
        )
        dataloaders_valid.append(_dataloader)
    print('------------------------------------------------------------')
    print(f'Batch size = {_batch_size}        Num worker = {_num_workers}')

    return dataloaders_train, dataloaders_valid


def get_Hybrid_loaders(file_names=0, num_workers=4):
    #
    # This function creates the dataloaders needed to automatically
    # feed the neural networks with the input dataset. In particular,
    # this function vields a dataloader for each specified mamico generated
    # csv file. As for num_workers, the rule of thumb is = 4 * num_GPU.
    # Note that the variable file_names acts as a flag such that
    # if == 0 : load data via mp and return one train and valid loader
    # if == -1: load data via mp and return multiple train and valid loaders
    # else: load random data for testing

    data_train = []
    data_valid = []
    _batch_size = 1
    _num_workers = 1

    if file_names == 0 or file_names == -1:
        _train_files = [
            'clean_couette_test_combined_domain_0_5_top.csv',
            'clean_couette_test_combined_domain_0_5_middle.csv',
            'clean_couette_test_combined_domain_0_5_bottom.csv',
            'clean_couette_test_combined_domain_1_0_top.csv',
            'clean_couette_test_combined_domain_1_0_middle.csv',
            'clean_couette_test_combined_domain_1_0_bottom.csv',
            'clean_couette_test_combined_domain_2_0_top.csv',
            'clean_couette_test_combined_domain_2_0_middle.csv',
            'clean_couette_test_combined_domain_2_0_bottom.csv',
            'clean_couette_test_combined_domain_4_0_top.csv',
            'clean_couette_test_combined_domain_4_0_middle.csv',
            'clean_couette_test_combined_domain_4_0_bottom.csv',
        ]

        _valid_files = [
            'clean_couette_test_combined_domain_3_0_top.csv',
            'clean_couette_test_combined_domain_3_0_middle.csv',
            'clean_couette_test_combined_domain_3_0_bottom.csv',
            'clean_couette_test_combined_domain_5_0_top.csv',
            'clean_couette_test_combined_domain_5_0_middle.csv',
            'clean_couette_test_combined_domain_5_0_bottom.csv'
        ]
        start_time = time.time()
        print('Loading training data.')
        data_train = mamico_csv2dataset_mp(_train_files)
        duration = time.time() - start_time
        print(
            f'Completed loading training data. Duration: {duration:.3f}')

        start_time = time.time()
        print('Loading validation data.')
        data_valid = mamico_csv2dataset_mp(_valid_files)
        duration = time.time() - start_time
        print(
            f'Completed loading validation data. Duration: {duration:.3f}')

        if file_names == -1:
            dataloaders_train = []
            dataloaders_valid = []

            for dataset in data_train:
                dataset = MyMamicoDataset(dataset)
                dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=num_workers
                    )
                dataloaders_train.append(dataloader)

            for dataset in data_valid:
                dataset = MyMamicoDataset(dataset)
                dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=num_workers
                    )
                dataloaders_valid.append(dataloader)
            return dataloaders_train, dataloaders_valid
    elif file_names == -2:
        _train_files = [
            'clean_kvs_10K_NE_combined_domain.csv',
            'clean_kvs_10K_NW_combined_domain.csv',
            'clean_kvs_10K_SE_combined_domain.csv',
            'clean_kvs_10K_SW_combined_domain.csv',
            'clean_kvs_20K_NE_combined_domain.csv',
            'clean_kvs_20K_NW_combined_domain.csv',
            'clean_kvs_20K_SE_combined_domain.csv',
            'clean_kvs_20K_SW_combined_domain.csv',
            'clean_kvs_30K_NE_combined_domain.csv',
            'clean_kvs_30K_NW_combined_domain.csv',
            'clean_kvs_30K_SE_combined_domain.csv',
            'clean_kvs_30K_SW_combined_domain.csv',
        ]

        _valid_files = [
            'clean_kvs_40K_NE_combined_domain.csv',
            'clean_kvs_40K_NW_combined_domain.csv',
            'clean_kvs_40K_SE_combined_domain.csv',
            'clean_kvs_40K_SW_combined_domain.csv',
        ]

        start_time = time.time()
        print('Loading training data.')
        data_train = mamico_csv2dataset_mp(_train_files)
        duration = time.time() - start_time
        print(
            f'Completed loading training data. Duration: {duration:.3f}')

        start_time = time.time()
        print('Loading validation data.')
        data_valid = mamico_csv2dataset_mp(_valid_files)
        duration = time.time() - start_time
        print(
            f'Completed loading validation data. Duration: {duration:.3f}')

        dataloaders_train = []
        dataloaders_valid = []

        for dataset in data_train:
            dataset = MyMamicoDataset_UNET_AE(dataset)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=_batch_size,
                shuffle=False,
                num_workers=_num_workers
                )
            dataloaders_train.append(dataloader)

        for dataset in data_valid:
            dataset = MyMamicoDataset_UNET_AE(dataset)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=_batch_size,
                shuffle=False,
                num_workers=_num_workers
                )
            dataloaders_valid.append(dataloader)
        return dataloaders_train, dataloaders_valid
    else:
        print('Loading ---> RANDOM <--- training datasets as loader.')
        for i in range(5):
            data = np.random.rand(32, 3, 26, 26, 26)
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            data_train.append(data)
        print('Completed loading ---> RANDOM <--- training datasets.')

        print('Loading ---> RANDOM <--- validation datasets as loader.')
        for i in range(3):
            data = np.random.rand(32, 3, 26, 26, 26)
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            data_valid.append(data)
        print('Completed loading ---> RANDOM <--- validation datasets.')

    data_valid_stack = np.vstack(data_valid)
    dataset_valid = MyMamicoDataset(data_valid_stack)
    dataloader_valid = DataLoader(
        dataset=dataset_valid,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers
        )

    data_train_stack = np.vstack(data_train)
    dataset_train = MyMamicoDataset(data_train_stack)
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers
        )

    return dataloader_train, dataloader_valid


def get_mamico_loaders(file_names=0, num_workers=4):
    #
    # This function creates the dataloaders needed to automatically
    # feed the neural networks with the input dataset. In particular,
    # this function vields a dataloader for each specified mamico generated
    # csv file. As for num_workers, the rule of thumb is = 4 * num_GPU.
    #
    dataloaders_train = []
    dataloaders_valid = []

    if file_names == 0:
        _directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata'
        _train_files = [
            'clean_couette_test_combined_domain_0_5_top.csv',
            'clean_couette_test_combined_domain_0_5_middle.csv',
            'clean_couette_test_combined_domain_0_5_bottom.csv',
            'clean_couette_test_combined_domain_1_0_top.csv',
            'clean_couette_test_combined_domain_1_0_middle.csv',
            'clean_couette_test_combined_domain_1_0_bottom.csv',
            'clean_couette_test_combined_domain_2_0_top.csv',
            'clean_couette_test_combined_domain_2_0_middle.csv',
            'clean_couette_test_combined_domain_2_0_bottom.csv',
            'clean_couette_test_combined_domain_3_0_top.csv',
            'clean_couette_test_combined_domain_3_0_middle.csv',
            'clean_couette_test_combined_domain_3_0_bottom.csv',
            'clean_couette_test_combined_domain_4_0_top.csv',
            'clean_couette_test_combined_domain_4_0_middle.csv',
            'clean_couette_test_combined_domain_4_0_bottom.csv',
            'clean_couette_test_combined_domain_5_0_top.csv',
            'clean_couette_test_combined_domain_5_0_middle.csv',
            'clean_couette_test_combined_domain_5_0_bottom.csv'
        ]

        _valid_files = [
            'clean_couette_test_combined_domain_3_0_top.csv',
            # 'clean_couette_test_combined_domain_3_0_middle.csv',
            # 'clean_couette_test_combined_domain_3_0_bottom.csv',
            # 'clean_couette_test_combined_domain_5_0_top.csv',
            # 'clean_couette_test_combined_domain_5_0_middle.csv',
            # 'clean_couette_test_combined_domain_5_0_bottom.csv'
        ]

        for file_name in _valid_files:
            start_time = time.time()
            print(f'Loading validation dataset as loader: {file_name}')
            # dataset = csv2dataset(f'{_directory}/{file_name}', (1000, 3, 26, 26, 26))
            dataset = mamico_csv2dataset(f'{file_name}')
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            dataset = MyMamicoDataset(dataset)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers
                )
            dataloaders_valid.append(dataloader)
            duration = time.time() - start_time
            print(
                f'Completed loading validation dataset. Duration: {duration:.3f}')

        for file_name in _train_files:
            print(f'Loading training dataset as loader: {file_name}')
            # dataset = csv2dataset(f'{_directory}/{file_name}', (1000, 3, 26, 26, 26))
            dataset = mamico_csv2dataset(f'{file_name}')
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            dataset = MyMamicoDataset(dataset)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers
                )
            dataloaders_train.append(dataloader)
            print('Completed loading training dataset.')
    else:
        for i in range(5):
            print('Loading ---> RANDOM <--- training dataset as loader.')
            dataset = np.random.rand(25, 3, 26, 26, 26)
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            dataset = MyMamicoDataset(dataset)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers
                )
            dataloaders_train.append(dataloader)
            print('Completed loading ---> RANDOM <--- training dataset.')
        for i in range(3):
            print('Loading ---> RANDOM <--- validation dataset as loader.')
            dataset = np.random.rand(25, 3, 26, 26, 26)
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            dataset = MyMamicoDataset(dataset)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers
                )
            dataloaders_valid.append(dataloader)
            print('Completed loading ---> RANDOM <--- validation dataset.')

    return dataloaders_train, dataloaders_valid


def get_UNET_AE_bottleneck():
    _model = UNET_AE(
        device=device,
        in_channels=3,
        out_channels=3,
        features=[4, 8, 16],
        activation=torch.nn.ReLU(inplace=True)
    ).to(device)

    #TO DO - Check proper model to load
    _model.load_state_dict(torch.load(
        '/home/lerdo/lerdo_HPC_Lab_Project/MD_U-Net/3_Constituent_Hybrid_approach/Results/0_UNET_AE/Model_UNET_AE_0_001'))
    _model.eval()
    _file_prefix = 'Bottleneck_Dataset'
    _out_file_names = [
        '_0_5_T',
        '_0_5_M',
        '_0_5_B',
        '_1_0_T',
        '_1_0_M',
        '_1_0_B',
        '_2_0_T',
        '_2_0_M',
        '_2_0_B',
        '_4_0_T',
        '_4_0_M',
        '_4_0_B',
        '_3_0_T',
        '_3_0_M',
        '_3_0_B',
        '_5_0_T',
        '_5_0_M',
        '_5_0_B',
    ]
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata'
    _in_file_names = [
        'clean_couette_test_combined_domain_0_5_top.csv',
        'clean_couette_test_combined_domain_0_5_middle.csv',
        'clean_couette_test_combined_domain_0_5_bottom.csv',
        'clean_couette_test_combined_domain_1_0_top.csv',
        'clean_couette_test_combined_domain_1_0_middle.csv',
        'clean_couette_test_combined_domain_1_0_bottom.csv',
        'clean_couette_test_combined_domain_2_0_top.csv',
        'clean_couette_test_combined_domain_2_0_middle.csv',
        'clean_couette_test_combined_domain_2_0_bottom.csv',
        'clean_couette_test_combined_domain_4_0_top.csv',
        'clean_couette_test_combined_domain_4_0_middle.csv',
        'clean_couette_test_combined_domain_4_0_bottom.csv',
        'clean_couette_test_combined_domain_3_0_top.csv',
        'clean_couette_test_combined_domain_3_0_middle.csv',
        'clean_couette_test_combined_domain_3_0_bottom.csv',
        'clean_couette_test_combined_domain_5_0_top.csv',
        'clean_couette_test_combined_domain_5_0_middle.csv',
        'clean_couette_test_combined_domain_5_0_bottom.csv'
    ]

    train_loader, _ = get_mamico_loaders()

    for i in range(18):
        _start_time = time.time()
        print(f'Loading dataset: {_in_file_names[i]}')
        # dataset = csv2dataset(f'{_directory}/{file_name}', (1000, 3, 26, 26, 26))
        _dataset = mamico_csv2dataset(f'{_in_file_names[i]}')
        # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
        _torch_dataset = torch.from_numpy(_dataset)
        _duration = time.time() - _start_time
        print(f'Completed loading dataset. Duration: {_duration:.3f}')

        _start_time = time.time()
        print(f'Loading bottleneck: {_in_file_names[i]}')
        bottleneck, _ = _model(_torch_dataset, y='get_bottleneck')
        latent_space = bottleneck.cpu().detach().numpy()
        _duration = time.time() - _start_time
        print(f'Completed loading bottleneck. Duration: {_duration:.3f}')
        print('Shape of latent space: ', latent_space.shape)
        _start_time = time.time()
        print(f'Saving latent space: {_file_prefix}{_out_file_names[i]}')
        dataset2csv(f'{_file_prefix}{_out_file_names[i]}', latent_space)
        _duration = time.time() - _start_time
        print(f'Completed loading bottleneck. Duration: {_duration:.3f}')


def losses2file(losses, filename):
    np.savetxt(f"{filename}.csv", losses, delimiter=", ", fmt='% s')


def checkSaveLoad():
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
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata'
    _filenames = ['kvs_20K_NE_combined_domain.csv']

    '''[
        'kvs_10K_NE_combined_domain.csv',
        'kvs_10K_NW_combined_domain.csv',
        'kvs_10K_SE_combined_domain.csv',
        'kvs_10K_SW_combined_domain.csv',
        'kvs_20K_NW_combined_domain.csv',
        'kvs_20K_SE_combined_domain.csv',
        'kvs_20K_SW_combined_domain.csv',
        'kvs_30K_NE_combined_domain.csv',
        'kvs_30K_NW_combined_domain.csv',
        'kvs_30K_SE_combined_domain.csv',
        'kvs_30K_SW_combined_domain.csv',
        'kvs_40K_NE_combined_domain.csv',
        'kvs_40K_NW_combined_domain.csv',
        'kvs_40K_SE_combined_domain.csv',
        'kvs_40K_SW_combined_domain.csv',
    ]'''

    processes = []
    counter = 1
    start = time.time()
    for idx, filename in enumerate(_filenames):
        p = mp.Process(
            target=clean_mamico_data,
            args=(_directory, filename,)
        )
        p.start()
        processes.append(p)
        print(f'Creating Process Number: {counter}')
        counter += 1

    for process in processes:
        process.join()
        print('Joining Process')

    duration = time.time() - start
    print(f'Loading Data via Multiprocessing takes: {duration:.3f} secs')
