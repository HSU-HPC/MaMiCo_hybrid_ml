import torch
import numpy as np
import csv
from dataset import MyMamicoDataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


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
            if(len(a) > 7):
                dataset[int(a[0])-1, 0, int(a[1])-1, int(a[2])
                        - 1, int(a[3])-1] = float(a[4])
                dataset[int(a[0])-1, 1, int(a[1])-1, int(a[2])
                        - 1, int(a[3])-1] = float(a[5])
                dataset[int(a[0])-1, 2, int(a[1])-1, int(a[2])
                        - 1, int(a[3])-1] = float(a[6])

    return dataset


'''
def dataset2csv(dataset, model_descriptor):
    #
    # This function reads from a MaMiCo generatd csv file.
    # Currently, proper functionality is hardcoded for simulations
    # containing 1000 timesteps.
    #
    # 1) Convert 3D array to 2D array
    dataset_reshaped = dataset.reshape(dataset.shape[0], -1)
    # 2) Save 2D array to file
    name = f'{string_file_name}_{t}_{c}_{d}_{h}_{w}'
    np.savetxt(f'{name}.csv', input_reshaped)




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
'''


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
            'clean_couette_test_combined_domain_0_5_top.csv'
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
            'clean_couette_test_combined_domain_3_0_top.csv'
            #'clean_couette_test_combined_domain_3_0_middle.csv',
            #'clean_couette_test_combined_domain_3_0_bottom.csv',
            #'clean_couette_test_combined_domain_5_0_top.csv',
            #'clean_couette_test_combined_domain_5_0_middle.csv',
            #'clean_couette_test_combined_domain_5_0_bottom.csv',
        ]

        for file_name in _valid_files:
            print(f'Loading validation dataset as loader: {file_name}')
            dataset = mamico_csv2dataset(f'{_directory}/{file_name}')
            # print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
            dataset = MyMamicoDataset(dataset)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers
                )
            dataloaders_valid.append(dataloader)
            print('Completed loading validation dataset.')

        for file_name in _train_files:
            print(f'Loading training dataset as loader: {file_name}')
            dataset = mamico_csv2dataset(f'{_directory}/{file_name}')
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
    np.savetxt(f"Losses_{filename}.csv", losses, delimiter=", ", fmt='% s')


if __name__ == "__main__":
    get_mamico_loaders(file_names=2)
    '''
    file_names = [
        #'couette_test_combined_domain_0_5_top.csv',
        #'couette_test_combined_domain_0_5_middle.csv',
        #'couette_test_combined_domain_0_5_bottom.csv',
        #'couette_test_combined_domain_1_0_top.csv',
        #'couette_test_combined_domain_1_0_middle.csv',
        #'couette_test_combined_domain_1_0_bottom.csv',
        #'couette_test_combined_domain_1_5_top.csv',
        #'couette_test_combined_domain_1_5_middle.csv',
        #'couette_test_combined_domain_1_5_bottom.csv',
        'couette_test_combined_domain_2_0_top.csv',
        'couette_test_combined_domain_2_0_middle.csv',
        'couette_test_combined_domain_2_0_bottom.csv',
        'couette_test_combined_domain_3_0_top.csv',
        'couette_test_combined_domain_3_0_middle.csv',
        'couette_test_combined_domain_3_0_bottom.csv',
        'couette_test_combined_domain_4_0_top.csv',
        'couette_test_combined_domain_4_0_middle.csv',
        'couette_test_combined_domain_4_0_bottom.csv',
        'couette_test_combined_domain_5_0_top.csv',
        'couette_test_combined_domain_5_0_middle.csv',
        'couette_test_combined_domain_5_0_bottom.csv'
    ]
    _directory = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata'
    for name in file_names:
        clean_mamico_data(_directory, filename=name)
        pass
        '''
