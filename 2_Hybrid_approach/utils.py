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


def clean_mamico_data(filename):
    #
    # This function is used to clean the MaMiCo generated
    # csv file. In other words, to remove the comma delimiter
    # and ensure a semicolon delimiter.
    #
    text = open(f"{filename}.csv", "r")
    text = ''.join([i for i in text]) \
        .replace(",", ";")
    x = open(f"clean_{filename}.csv", "w")
    x.writelines(text)
    x.close()
    pass


def mamico_csv2dataset(file_name):
    #
    # This function reads from a MaMiCo generatd csv file.
    # Currently, proper functionality is hardcoded for simulations
    # containing 1000 timesteps.
    #

    dataset = np.zeros((1000, 3, 26, 26, 26))

    with open(f'{file_name}') as csvfile:
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


def get_mamico_loaders(file_names=0, num_workers=4):
    #
    # This function creates the dataloaders needed to automatically
    # feed the neural networks with the input dataset. In particular,
    # this function vields a dataloader for each specified mamico generated
    # csv file. As for num_workers, the rule of thumb is = 4 * num_GPU.
    #
    dataloaders = []

    if file_names == 0:
        prefix = '/home/lerdo/lerdo_HPC_Lab_Project/Trainingdata'
        file_names = [f'{prefix}/couette_test_combined_domain_0_5.csv']
        '''
            f'{prefix}/couette_test_combined_domain_1_0.csv',
            f'{prefix}/couette_test_combined_domain_1_5.csv',
            f'{prefix}/couette_test_combined_domain_2_0.csv',
            f'{prefix}/couette_test_combined_domain_3_0.csv',
            f'{prefix}/couette_test_combined_domain_4_0.csv',
            f'{prefix}/couette_test_combined_domain_5_0.csv',
            f'{prefix}/couette_test_combined_domain_0_5.csv',
            f'{prefix}/kvs_test_combined_domain.csv'
        ]'''

    for file_name in file_names:
        dataset = mamico_csv2dataset(file_name)
        print("Utils.py - Sanity Check - Dimension of loaded dataset: ", dataset.shape)
        dataset = MyMamicoDataset(dataset)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers
            )
        dataloaders.append(dataloader)

    return dataloaders


def userModelSpecs():
    '''
    invalid = True
    while invalid:
        print("Please choose which hybrid model to train:")
        print("1) Hybrid_MD_RNN_UNET")
        print("2) Hybrid_MD_GRU_UNET")
        print("3) Hybrid_MD_LSTM_UNET \n")

        model_name = input()

        if(model_name == "1"):
            print("You've chosen: Hybrid_MD_RNN_UNET \n")
            _model_name = 'Hybrid_MD_RNN_UNET'
            invalid = False
        elif(model_name == "2"):
            print("You've chosen: Hybrid_MD_GRU_UNET \n")
            _model_name = 'Hybrid_MD_GRU_UNET'
            invalid = False
        elif(model_name == "3"):
            print("You've chosen: Hybrid_MD_LSTM_UNET \n")
            _model_name = 'Hybrid_MD_LSTM_UNET'
            invalid = False
        else:
            print("Invalid input \n")

    invalid = True
    while invalid:
        print("Please choose how many RNN layers this model should contain:")
        print("1) 2 ")
        print("2) 3")
        print("3) 4 \n")

        rnn_layers = input()

        if(rnn_layers == "1"):
            print("You've chosen: 2 layers. \n")
            _rnn_layers = 2
            invalid = False
        elif(rnn_layers == "2"):
            print("You've chosen: 3 layers. \n")
            _rnn_layers = 3
            invalid = False
        elif(rnn_layers == "3"):
            print("You've chosen: 4 layers. \n")
            _rnn_layers = 4
            invalid = False
        else:
            print("Invalid input \n")

    invalid = True
    while invalid:
        print("Please choose the size of the RNN hidden layers:")
        print("1) 256 ")
        print("2) 512")
        print("3) 768 \n")

        hid_size = input()

        if(hid_size == "1"):
            print("You've chosen: 256 nodes per hidden layer. \n")
            _hid_size = 256
            invalid = False
        elif(hid_size == "2"):
            print("You've chosen: 512 nodes per hidden layer. \n")
            _hid_size = 512
            invalid = False
        elif(hid_size == "3"):
            print("You've chosen: 768 nodes per hidden layer. \n")
            _hid_size = 768
            invalid = False
        else:
            print("Invalid input \n")

    invalid = True
    while invalid:
        print("Please choose the learning rate:")
        print("1) 0.0001 ")
        print("2) 0.00005")
        print("3) 0.00001 \n")

        learning_rate = input()

        if(learning_rate == "1"):
            print("You've chosen: learning_rate = 0.0001. \n")
            _learning_rate = 0.0001
            invalid = False
        elif(learning_rate == "2"):
            print("You've chosen: learning_rate = 0.00005. \n")
            _learning_rate = 0.00005
            invalid = False
        elif(learning_rate == "3"):
            print("You've chosen: learning_rate = 0.00001. \n")
            _learning_rate = 0.00001
            invalid = False
        else:
            print("Invalid input \n")
    '''

    _model_name = "Hybrid_MD_GRU_UNET"
    _rnn_layers = 2
    _hid_size = 256
    _learning_rate = 0.0001

    model_name = 2
    rnn_layers = 1
    hid_size = 1
    learning_rate = 1

    print('------------------------------------------------------------')
    print('                       Model Summary')
    print('------------------------------------------------------------')
    print(f'Model Name: {_model_name}')
    print(f'RNN Layers: {_rnn_layers}')
    print(f'Size hidden Layer: {_hid_size}')
    print(f'Learning Rate: {_learning_rate}')
    print('------------------------------------------------------------')

    # Consider if returning a list or a dictionary is more beneficial
    return [int(model_name)-1, int(rnn_layers)-1, int(hid_size)-1, int(learning_rate)-1]


def losses2file(losses, filename):
    np.savetxt(f"Losses_{filename}.csv", losses, delimiter=", ", fmt='% s')


if __name__ == "__main__":
    results = userModelSpecs()
    for value in results:
        print(f'{str(value)}')
    pass
