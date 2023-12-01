import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import logging

# from models.vgg import VGG
from models.resnet18 import ResNet18
from utils.dataset import load_datasets
from utils.terminal import MetricMonitor
from utils.logger import logger_info
from utils.dirs import mkdirs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = 'checkpoint/secret_model/resnet18.pt'
mkdirs('checkpoint/secret_model')
mkdirs('results/secret_model')
logger_name = 'secret_model'
logger_info(logger_name, log_path=os.path.join('results', logger_name, 'resnet18.log'))
logger = logging.getLogger(logger_name)
logger.info('#'*50)
logger.info('model: {:s}'.format('resnet18'))
logger.info('secret dataset: {:s}'.format('fashion-mnist'))


def train():
    model.train()
    correct = 0
    total = 0
    metric_monitor = MetricMonitor(float_precision=4)
    stream = tqdm(train_loader)
    for inputs, targets in stream:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Acc", correct/total)
        stream.set_description(
            "Epoch: {epoch}. Train.   {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )


def test():
    global best_acc
    model.eval()
    correct = 0
    total = 0
    metric_monitor = MetricMonitor(float_precision=4)
    stream = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(stream):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Acc", correct/total)
            stream.set_description(
                "Epoch: {epoch}. Test.    {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

    # Save model.
    acc = correct/total
    if acc > best_acc:
        best_acc = acc
        logger.info('Saving.., epoch: {}, test acc: {:.4f}'.format(epoch+1, acc))
        torch.save(model, model_save_path)


if __name__ == '__main__':
    lr = 1e-3
    ep = 2 # 120
    bs = 128 # batch size
    num_classes = 10 # fashion-mnist
    train_loader, test_loader = load_datasets('fmnist-10', bs)

    model = ResNet18(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss() 

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, [60, 90], gamma=0.1)

    best_acc = 0  # best test accuracy.

    for epoch in range(ep):
        train()
        test()
        scheduler.step()
    logger.info('best_acc: {:.5f}'.format(best_acc))


