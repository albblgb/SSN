import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import logging
import copy

from models.vgg import VGG
from utils.dataset import load_datasets
from utils.terminal import MetricMonitor
from utils.logger import logger_info
from utils.dirs import mkdirs
from utils.proposed_mothod import filter_selection, secret_model_extraction, secret_model_embedding, side_info_embedding, side_info_extraction
from utils.model import describe_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = 'checkpoint/stego_model/vgg.pt'
mkdirs('checkpoint/stego_model')
mkdirs('results/stego_model')
logger_name = 'stego_model'
logger_info(logger_name, log_path=os.path.join('results', logger_name, 'vgg.log'))
logger = logging.getLogger(logger_name)
logger.info('#'*50)
logger.info('model: {:s}'.format('vgg11'))
logger.info('secret dataset: {:s}'.format('fashion-mnist'))
logger.info('stego dataset: {:s}'.format('cifar10'))


def train(model, train_loader, sparse_masks=None):
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

        # partial optimization
        if sparse_masks != None:
            l_c = 0; l_b = 0
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.grad.data = torch.mul(m.weight.grad.data, (1-sparse_masks[l_c]))
                    l_c += 1
                elif isinstance(m, nn.Linear):
                    m.weight.grad.data = torch.mul(m.weight.grad.data, (1-sparse_masks[len(sparse_masks)-1]))

        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Acc", correct/total)
        stream.set_description(
            "Epoch: {epoch}. Train.   {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )


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
    ep = 2
    bs = 128 # batch size
    num_classes = 10 # fashion-mnist
    prop = 0.9 # The proportion of the selected filters 
    secret_model_path = 'checkpoint/secret_model/vgg.pt' # pre-trained
    key = 10000

    logger.info('Loading secret dnn model')
    secret_model = (torch.load(secret_model_path))
    
    logger.info('Loading secret and stego datasets')
    secret_train_loader, secret_test_loader = load_datasets('fmnist-10', bs)
    stego_train_loader, stego_test_loader = load_datasets('cf-10', bs)

    logger.info('Selecting important filters')
    B_stream, sparse_masks = filter_selection(secret_model, secret_test_loader, stego_test_loader, prop, device)

    logger.info('Extracting the secret sub-model composed by selected filters from the original secret model.')
    sub_model = secret_model_extraction(secret_model, B_stream)
    sub_model = sub_model.to(device)

    logger.info('Tuning the sub-model on the secret dataset.')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(sub_model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, [80, 120], gamma=0.1)
    best_acc = 0
    model_save_path = 'checkpoint/secret_sub_model/vgg.pt'
    mkdirs('checkpoint/secret_sub_model')
    for epoch in range(ep):
        train(sub_model, secret_train_loader)
        test(sub_model, secret_test_loader, model_save_path)
    tuned_sub_model = torch.load(model_save_path)

    logger.info('Embedding the tuned secret sub-model into a initialized stego-vgg11.')
    #  i.e., replace partial filters in stego-vgg11 according to sparse masks or B_stream
    model = VGG('VGG11', num_classes)
    stego_model = secret_model_embedding(model, tuned_sub_model, B_stream, sparse_masks)
    stego_model = stego_model.to(device)
    bn_masks = copy.deepcopy(B_stream)
    for i in range(len(bn_masks)):
        bn_masks[i] = bn_masks[i].to(device)
    for i in range(len(sparse_masks)):
        sparse_masks[i] = sparse_masks[i].to(device)
    logger.info('Activating the remaining filters (unreplaced ones) in stego-vgg11 on the stego dataset.')
    optimizer = optim.Adam(stego_model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, [80, 120], gamma=0.1)
    best_acc = 0
    model_save_path = 'checkpoint/stego_model/vgg.pt'
    mkdirs('checkpoint/stego_model')
    for epoch in range(ep):
        train(stego_model, stego_train_loader, sparse_masks)
        test(stego_model, stego_test_loader, model_save_path)


    logger.info('Embedding the B_stream into the stego model.')
    stego_model = torch.load(model_save_path)
    stego_model = side_info_embedding(key, stego_model, B_stream)
    torch.save(stego_model, model_save_path)




    # Extraction for sender
    logger.info('Loading stego model')
    stego_model = (torch.load(model_save_path))

    logger.info('Extracting the B_stream and bn_running info form the stego model')
    B_stream = side_info_extraction(stego_model, key)

    logger.info('Extracting the secret sub-model composed by selected filters from the stego model')
    secret_model = secret_model_extraction(stego_model, B_stream)
    secret_model = secret_model.to(device)

    logger.info('Testing the extracted secret sub-network')
    model_save_path = 'checkpoint/extracted_seceret_model/vgg.pt'
    mkdirs('checkpoint/extracted_seceret_model')
    test(secret_model, secret_test_loader, model_save_path)
        
    