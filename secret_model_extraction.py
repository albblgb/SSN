import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import logging

from models.vgg import VGG
from utils.dataset import load_datasets
from utils.terminal import MetricMonitor
from utils.logger import logger_info
from utils.dirs import mkdirs
from utils.proposed_mothod import secret_model_extraction, side_info_extraction, bn_running_embedding


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = 'checkpoint/extracted_seceret_model/vgg.pt'
mkdirs('checkpoint/extracted_seceret_model')
mkdirs('results/extracted_secret_model')
logger_name = 'extracted_secret_model'
logger_info(logger_name, log_path=os.path.join('results', logger_name, 'vgg.log'))
logger = logging.getLogger(logger_name)
logger.info('#'*50)
logger.info('model: {:s}'.format('vgg11'))
logger.info('secret dataset: {:s}'.format('fashion-mnist'))


def test(model, test_loader, model_save_path):
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
                "Test. {metric_monitor}".format(metric_monitor=metric_monitor)
            )

    # Save model.
    acc = correct/total
    logger.info('Saving.., test acc: {:.4f}'.format(acc))
    torch.save(model, model_save_path)


if __name__ == '__main__':

    lr = 1e-3
    bs = 128 # batch size
    num_classes = 10 # fashion-mnist
    stego_model_path = 'checkpoint/stego_model/vgg.pt' # pre-trained
    key = 10000

    logger.info('Loading secret datasets')
    _, secret_test_loader = load_datasets('fmnist-10', bs)

    logger.info('Loading stego model')
    stego_model = (torch.load(stego_model_path))

    logger.info('Extracting the B_stream and bn_running info form the stego model')
    B_stream, bn_running_binary = side_info_extraction(stego_model, key)

    logger.info('Extracting the secret sub-model composed by selected filters from the stego model')
    secret_model = secret_model_extraction(stego_model, B_stream)

    logger.info('Recovering the running means and vars of bn layers')      
    secret_model = bn_running_embedding(secret_model, bn_running_binary)
    secret_model = secret_model.to(device)

    logger.info('Testing the extracted secret sub-network')
    criterion = nn.CrossEntropyLoss()
    test(secret_model, secret_test_loader, model_save_path)



    



