import torch

from other_modules import utils
from dataset import DefectDataset
from other_modules.engine import train_one_epoch, evaluate
from model import get_model
from transform import get_transform


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cpu')

    # our dataset has three classes only - background, holes, notch
    num_classes = 2 + 1
    # use our dataset and defined transformations
    train_data_dir = 'data/images'
    train_coco = 'data/annotations.json'

    dataset = DefectDataset(root=train_data_dir,
                            annotation=train_coco,
                            transforms=get_transform(train=True)
                            )
    dataset_test = DefectDataset(root=train_data_dir,
                                 annotation=train_coco,
                                 transforms=get_transform(train=False)
                                 )

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-7])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-7:])

    train_batch_size = 2
    test_batch_size = 1

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=test_batch_size,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 35

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
    return model


if __name__ == '__main__':
    model = main()
