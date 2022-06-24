import torch
import numpy as np
import time
import csv
from dataset import MyMamicoDataset, MyMamicoDataset_UNET_AE
from torch.utils.data import DataLoader


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
            if(len(a) > 7):
                dataset[int(a[0])-1, 0, int(a[1])-1, int(a[2])
                        - 1, int(a[3])-1] = float(a[4])
                dataset[int(a[0])-1, 1, int(a[1])-1, int(a[2])
                        - 1, int(a[3])-1] = float(a[5])
                dataset[int(a[0])-1, 2, int(a[1])-1, int(a[2])
                        - 1, int(a[3])-1] = float(a[6])

    return dataset


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
    dataset = np.loadtxt(f'{filename}')

    if output_shape == 0:
        return dataset

    t, c, d, h, w = output_shape

    # 4) Revert 2D array to 3D array
    original_dataset = dataset.reshape(t, c, d, h, w)
    return original_dataset


def get_UNET_AE_loaders(file_names=0, num_workers=4):
    #
    # This function creates the dataloaders needed to automatically
    # feed the neural networks with the input dataset. In particular,
    # this function vields a dataloader for each specified mamico generated
    # csv file. As for num_workers, the rule of thumb is = 4 * num_GPU.
    #
    data_train = []
    data_valid = []

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

        for file_name in _valid_files:
            start_time = time.time()
            print(f'Loading validation data: {file_name}')
            # dataset = csv2dataset(f'{_directory}/{file_name}', (1000, 3, 26, 26, 26))
            data = mamico_csv2dataset(f'{file_name}')
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            data_valid.append(data)
            duration = time.time() - start_time
            print(
                f'Completed loading validation data. Duration: {duration:.3f}')

        for file_name in _train_files:
            start_time = time.time()
            print(f'Loading training data: {file_name}')
            # dataset = csv2dataset(f'{_directory}/{file_name}', (1000, 3, 26, 26, 26))
            data = mamico_csv2dataset(f'{file_name}')
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            data_train.append(data)
            duration = time.time() - start_time
            print(
                f'Completed loading training data. Duration: {duration:.3f}')

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
    dataset_valid = MyMamicoDataset_UNET_AE(data_valid_stack)
    dataloader_valid = DataLoader(
        dataset=dataset_valid,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers
        )

    data_train_stack = np.vstack(data_train)
    dataset_train = MyMamicoDataset_UNET_AE(data_train_stack)
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=32,
        shuffle=True,
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
            #'clean_couette_test_combined_domain_0_5_middle.csv',
            #'clean_couette_test_combined_domain_0_5_bottom.csv',
            #'clean_couette_test_combined_domain_1_0_top.csv',
            #'clean_couette_test_combined_domain_1_0_middle.csv',
            #'clean_couette_test_combined_domain_1_0_bottom.csv',
            #'clean_couette_test_combined_domain_2_0_top.csv',
            #'clean_couette_test_combined_domain_2_0_middle.csv',
            #'clean_couette_test_combined_domain_2_0_bottom.csv',
            #'clean_couette_test_combined_domain_4_0_top.csv',
            #'clean_couette_test_combined_domain_4_0_middle.csv',
            #'clean_couette_test_combined_domain_4_0_bottom.csv',
        ]

        _valid_files = [
            'clean_couette_test_combined_domain_3_0_top.csv',
            #'clean_couette_test_combined_domain_3_0_middle.csv',
            #'clean_couette_test_combined_domain_3_0_bottom.csv',
            #'clean_couette_test_combined_domain_5_0_top.csv',
            #'clean_couette_test_combined_domain_5_0_middle.csv',
            #'clean_couette_test_combined_domain_5_0_bottom.csv'
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


def checkUserModelSpecs(user_input):
    # BRIEF: This allows to verify that the user command line arguments are
    # valid and adhere to the coding convention.
    #
    # PARAMETERS:
    #
    # _model_name, _rnn_layer, _hid_size, _learning_rate = user_input. Note that
    # each argument must be an integer value between 1 and 3. The following list
    # describes the corresponding meaning:
    #
    # _model_name -    1) Hybrid_MD_RNN_UNET
    #                  2) Hybrid_MD_GRU_UNET
    #                  3) Hybrid_MD_LSTM_UNET
    #
    # _rnn_layer -     1) 2
    #                  2) 3
    #                  3) 4
    #
    # _hid_size -      1) 256
    #                  2) 512
    #                  3) 768
    #
    # _learning_rate - 1) 0.0001
    #                  2) 0.00005
    #                  3) 0.00001

    _model_names = ['Hybrid_MD_RNN_UNET',
                    'Hybrid_MD_GRU_UNET', 'Hybrid_MD_LSTM_UNET']
    # @_model_names - model names as strings for later file naming
    _rnn_layers = [2, 3, 4]
    # @_rnn_layers - container to hold the number of rnn layers deemed worth testing
    _hid_sizes = [256, 512, 768]
    # @_hid_sizes - container to hold the number of nodes per hidden layer deemed worth testing
    _learning_rates = [0.0001, 0.00005, 0.00001]
    # @_learning_rates - container to hold the learning rates deemed worth testing

    for i in range(len(user_input)):
        user_input[i] = int(user_input[i])

    if(len(user_input) < 4):
        print("Not enough arguments. Four are required.")
        return False

    if(len(user_input) > 4):
        print("Too many arguments. Four are required.")
        return False

    _model_name, _rnn_layer, _hid_size, _learning_rate = user_input
    _valid_input = True

    if (_model_name < 1 or _model_name > 3):
        print("You've chosen an invalid model.")
        _valid_input = False
    if (_rnn_layer < 1 or _rnn_layer > 3):
        print("You've chosen an invalid amount of RNN layers.")
        _valid_input = False

    if (_hid_size < 1 or _hid_size > 3):
        print("You've chosen an invalid amount of nodes per RNN layer.")
        _valid_input = False

    if (_learning_rate < 1 or _learning_rate > 3):
        print("You've chosen an invalid learning rate.")
        _valid_input = False

    if(_valid_input):
        print('------------------------------------------------------------')
        print('                       Model Summary')
        print('------------------------------------------------------------')
        print(f'Model Name: {_model_names[_model_name-1]}')
        print(f'RNN Layers: {_rnn_layers[_rnn_layer-1]}')
        print(f'Size hidden Layer: {_hid_sizes[_hid_size-1]}')
        print(f'Learning Rate: {_learning_rates[_learning_rate-1]}')
        print('------------------------------------------------------------')

    return _valid_input


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


if __name__ == "__main__":
    pass
