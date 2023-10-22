import torch

from dataset import BirdCLEF2023_Dataset
from model import Mel_Classifier

def train_net(net, trainloader, valloader, criterion, optimizer, scheduler, epochs=1, device = 'cpu'):

    best_state_dictionary = None
    best_validation_accuracy = 0.0
    inertia = 0
    for epoch in range(epochs):  # loop over the dataset multiple times, only 1 time by default

        running_loss = 0.0
        net = net.train()
        for i, data in enumerate(trainloader, 0):
            print('Batch: ' + str(i))
            # get the inputs
            inputs, labels = data
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
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] Training loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        if type(scheduler).__name__ != 'NoneType':
            scheduler.step()

        running_loss = 0.0
        correct = 0
        net = net.eval()
        for i, data in enumerate(valloader, 0):
            # get the inputs
            inputs, labels = data
            if device == 'cuda':
                inputs, labels = inputs.cuda(), labels.cuda()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            if device == 'cuda':
                loss = loss.cpu()

            # print statistics and write to log
            running_loss += loss.item()
            if i % 20 == 19:  # print every 2000 mini-batches
                print('[%d, %5d] Validation loss: %.3f' %
                             (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
            correct += (outputs.argmax(1) == labels).sum().item()

        val_accuracy = 100 * correct / len(valloader)

        if val_accuracy > best_validation_accuracy:
            best_state_dictionary = net.state_dict()
            inertia = 0
        else:
            inertia += 1
            if inertia == 3:
                if best_state_dictionary is None:
                    raise Exception("State dictionary should have been updated at least once")
                break
        print(f"Validation accuracy: {val_accuracy}")

    # save network
    torch.save(best_state_dictionary, 'project2_modified.pth')
    # write finish to the file
    print('Finished Training')

if __name__ == '__main__':

    dataset_path = 'birdclef-2023'
    dataset_ = BirdCLEF2023_Dataset(data_path=dataset_path)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset_, (0.8, 0.2))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = True, num_workers = 2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 1, shuffle = True, num_workers = 2)

    network = Mel_Classifier()
    for param in network.dinov2_vits14.parameters():
        param.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, network.parameters()), lr=0.0004, momentum=0.9)  # adjust optimizer settings

    train_net(net = network
              ,trainloader = train_loader
              ,valloader = val_loader
              ,criterion = criterion
              ,optimizer = optimizer
              ,scheduler = None
              ,epochs=1
              ,device = 'cpu')

