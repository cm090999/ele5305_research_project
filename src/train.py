import torch
import os
import copy
import time
import logging
import sys

from dataset import BirdCLEF2023_Dataset
from model import Mel_Classifier

def create_logger(final_output_path):
    log_file = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file),
                        format=head)
    clogger = logging.getLogger()
    clogger.setLevel(logging.INFO)
    # add handler
    # print to stdout and log file
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    clogger.addHandler(ch)
    return clogger

def train_net(net, trainloader, valloader, logging, criterion, optimizer, scheduler, epochs=1, patience = 3, savePth = 'project2_weights.pth', print_every_samples = 20, device = 'cpu'):

    logging.info('Using device: {}'.format(device))
    net.to(device)

    validation_loss_list = []
    training_loss_list = []
    validation_accuracy_list = []
    training_accuracy_list = []

    best_state_dictionary = None
    best_validation_accuracy = 0.0
    inertia = 0
    for epoch in range(epochs):

        training_loss = 0.0
        training_accuracy = 0.0
        running_loss = 0.0
        net = net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data['mel_spec'], data['primary_label_tensor']
            if device == 'cuda':
                inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if device == 'cuda':
                loss = loss.cpu()

            # print statistics and write to log
            running_loss += loss.item()
            training_loss += loss.item()
            if i % print_every_samples == print_every_samples - 1:    # print every 2000 mini-batches
                logging.info('[%d, %5d / %5d] Training loss: %.3f' %
                      (epoch + 1, i + 1, len(trainloader), running_loss / print_every_samples))
                running_loss = 0.0

            training_accuracy += (outputs.argmax(1) == labels.argmax(1)).sum().item()

        if type(scheduler).__name__ != 'NoneType':
            scheduler.step()

        training_loss = training_loss / len(trainloader.dataset)
        training_loss_list.append(training_loss)
        training_accuracy = 100 * training_accuracy / len(trainloader.dataset)
        training_accuracy_list.append(training_accuracy)

        running_loss = 0.0
        val_loss = 0.0
        correct = 0
        net = net.eval()
        for i, data in enumerate(valloader, 0):
            # get the inputs
            inputs, labels = data['mel_spec'], data['primary_label_tensor']
            if device == 'cuda':
                inputs, labels = inputs.cuda(), labels.cuda()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            if device == 'cuda':
                loss = loss.cpu()

            # print statistics and write to log
            running_loss += loss.item()
            val_loss += loss.item()
            if i % print_every_samples == print_every_samples - 1:  # print every 2000 mini-batches
                logging.info('[%d, %5d / %5d] Validation loss: %.3f' %
                             (epoch + 1, i + 1, len(valloader), running_loss / print_every_samples))
                running_loss = 0.0
            correct += (outputs.argmax(1) == labels.argmax(1)).sum().item()

        val_loss = val_loss / len(valloader.dataset)
        validation_loss_list.append(val_loss)
        val_accuracy = 100 * correct / len(valloader.dataset)
        validation_accuracy_list.append(val_accuracy)

        save_weights = os.path.join(savePth,'model_weights.pth')
        if val_accuracy > best_validation_accuracy:
            best_state_dictionary = copy.deepcopy(net.state_dict())
            # save network
            torch.save(best_state_dictionary, save_weights)
            inertia = 0
        else:
            inertia += 1
            if inertia == patience:
                if best_state_dictionary is None:
                    raise Exception("State dictionary should have been updated at least once")
                break
        print(f"Validation accuracy: {val_accuracy}")

    logging.info('Finished Training')

    output = {'validation_loss': validation_loss_list,
              'validation_accuracy': validation_accuracy_list,
              'training_loss': training_loss_list,
              'training_accuracy': training_accuracy_list}
    
    return output

if __name__ == '__main__':

    logger = create_logger(final_output_path='')

    dataset_path = 'birdclef-2023'
    dataset_ = BirdCLEF2023_Dataset(data_path=dataset_path)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset_, (0.8, 0.2))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 2, shuffle = True, num_workers = 2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 2, shuffle = True, num_workers = 2)

    network = Mel_Classifier()
    for param in network.dinov2_vits14.parameters():
        param.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, network.parameters()), lr=0.0004, momentum=0.9)  # adjust optimizer settings

    train_net(net = network
              ,trainloader = train_loader
              ,valloader = val_loader
              ,logging=logger
              ,criterion = criterion
              ,optimizer = optimizer
              ,scheduler = None
              ,epochs=1
              ,device = 'cpu')


# TODO data augmentation on mel spec
# TODO data augmentation on wave fiel
# TODO class uniform sampler
