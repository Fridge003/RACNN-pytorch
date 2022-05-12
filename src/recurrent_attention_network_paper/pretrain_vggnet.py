import imageio
import os
import numpy as np
import sys
import torch
import time
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

sys.path.append('.')  # noqa: E402
from src.recurrent_attention_network_paper.model import RACNN
from src.recurrent_attention_network_paper.CUB_loader import CUB200_loader
from torch.autograd import Variable

def eval(net, dataloader, device, writer, total_step):
    print(' :: Testing on test set ...')
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    for step, (inputs, labels) in enumerate(dataloader, 0):
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

        with torch.no_grad():
            logits = net(inputs)
            correct_top1 += torch.eq(logits.topk(max((1, 1)), 1, True, True)[1], labels.view(-1, 1)).sum().float().item()
            correct_top3 += torch.eq(logits.topk(max((1, 3)), 1, True, True)[1], labels.view(-1, 1)).sum().float().item()
            correct_top5 += torch.eq(logits.topk(max((1, 5)), 1, True, True)[1], labels.view(-1, 1)).sum().float().item()


        if step > 200:
            test_num = (step+1) * int(inputs.shape[0])
            print(f'\tAccuracy@top1 ({step}/{len(dataloader)}) = {correct_top1/test_num:.5%}')
            print(f'\tAccuracy@top3 ({step}/{len(dataloader)}) = {correct_top3/test_num:.5%}')
            print(f'\tAccuracy@top5 ({step}/{len(dataloader)}) = {correct_top5/test_num:.5%}')
            writer.add_scalar('Accuracy/val_top1_acc', correct_top1/test_num, total_step)
            writer.add_scalar('Accuracy/val_top3_acc', correct_top3/test_num, total_step)
            writer.add_scalar('Accuracy/val_top5_acc', correct_top5/test_num, total_step)
            return


def run(save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    state_dict = torchvision.models.vgg19_bn(pretrained=True).state_dict()
    state_dict.pop('classifier.6.weight')
    state_dict.pop('classifier.6.bias')
    net = torchvision.models.vgg19_bn(num_classes=200).to(device)
    state_dict['classifier.6.weight'] = net.state_dict()['classifier.6.weight']
    state_dict['classifier.6.bias'] = net.state_dict()['classifier.6.bias']
    net.load_state_dict(state_dict)
    cudnn.benchmark = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    trainset = CUB200_loader('../CUB_200_2011', split='train')
    testset = CUB200_loader('../CUB_200_2011', split='test')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, collate_fn=trainset.CUB_collate, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, collate_fn=testset.CUB_collate, num_workers=4)
    log_path = os.path.join(save_path, 'log')
    writer = SummaryWriter(log_path)

    # classes = [line.split(' ')[1] for line in open('../CUB_200_2011/classes.txt', 'r').readlines()]
    print(' :: Start training ...')

    total_step = -1
    for epoch in range(100):  # loop over the dataset multiple times
        losses = 0
        for step, (inputs, labels) in enumerate(trainloader, 0):
            total_step += 1
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss, total_step)

            losses += loss

            if step % 20 == 0 and step != 0:
                avg_loss = losses/20
                print(f':: loss @step({step:2d}/{len(trainloader)})-epoch{epoch}: {loss:.10f}\tavg_loss_20: {avg_loss:.10f}')
                losses = 0
        eval(net, testloader, device, writer, total_step)
        if epoch % 20 == 19:
            stamp = f'{epoch}-{int(time.time())}'
            torch.save(net, os.path.join(save_path, f'model-{stamp}.pt'))
            torch.save(optimizer.state_dict, os.path.join(save_path, f'optimizer-{stamp}.pt'))


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    save_path = 'vgg_pretrain_result'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # create the log folder if not exists
    if not os.path.exists(os.path.join(save_path, 'log')):
        os.makedirs(os.path.join(save_path, 'log'))

    run(save_path)
