import imageio
import os
import shutil
import sys
import torch
import time
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

sys.path.append('.')  # noqa: E402
from src.recurrent_attention_network_paper.model import RACNN
from src.recurrent_attention_network_paper.CUB_loader import CUB200_loader
from src.recurrent_attention_network_paper.pretrain_apn import random_sample, save_img, clean, build_gif



def avg(x): return sum(x)/len(x)


def train(net, dataloader, optimizer, epoch, _type, writer, total_step):

    assert _type in ['apn', 'backbone']
    losses = 0
    net.mode(_type)
    print(f' :: Switch to {_type}')  # switch loss type

    for step, (inputs, targets) in enumerate(dataloader, 0):
        loss = net.echo(inputs, targets, optimizer)
        losses += loss
        total_step += 1

        if step % 20 == 0 and step != 0:
            avg_loss = losses/20
            print(f':: loss @step({step:2d}/{len(dataloader)})-epoch{epoch}: {loss:.10f}\tavg_loss_20: {avg_loss:.10f}')
            if _type == 'backbone':
                writer.add_scalar('Train/backbone_loss', avg_loss, total_step)
            if _type == 'apn':
                writer.add_scalar('Train/apn_loss', avg_loss, total_step)
            losses = 0

    return total_step


def test(net, dataloader, writer, epoch):

    print(' :: Testing on test set ...')
    correct_summary = {'clsf-0': {'top-1': 0, 'top-5': 0}, 'clsf-1': {'top-1': 0, 'top-5': 0}, 'clsf-2': {'top-1': 0, 'top-5': 0}}

    for step, (inputs, labels) in enumerate(dataloader, 0):
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        with torch.no_grad():
            outputs, _, _, _ = net(inputs)
            for idx, logits in enumerate(outputs):
                correct_summary[f'clsf-{idx}']['top-1'] += torch.eq(logits.topk(max((1, 1)), 1, True, True)[1], labels.view(-1, 1)).sum().float().item()  # top-1
                correct_summary[f'clsf-{idx}']['top-5'] += torch.eq(logits.topk(max((1, 5)), 1, True, True)[1], labels.view(-1, 1)).sum().float().item()  # top-5

            if step > 300: # only use a portion of the test-dataset for testing
                for clsf in correct_summary.keys():
                    _summary = correct_summary[clsf]
                    for topk in _summary.keys():
                        cls_idx = clsf[-1] # 0 or 1 or 2
                        acc = _summary[topk]/float(((step+1)*int(inputs.shape[0])))
                        writer.add_scalar(f'Test-{cls_idx}/Accuracy-{topk}', acc, epoch)
                        print(f'\tAccuracy {clsf}@{topk} ({step}/{len(dataloader)}) = {acc:.5%}')
                return


def run(pretrained_model, save_path='./racnn_result'):

    print(f' :: Start training with {pretrained_model}')
    net = RACNN(num_classes=200).cuda()
    net.load_state_dict(torch.load(pretrained_model))
    cudnn.benchmark = True

    cls_params = list(net.b1.parameters()) + list(net.b2.parameters()) + list(net.b3.parameters()) + \
        list(net.classifier1.parameters()) + list(net.classifier2.parameters()) + list(net.classifier3.parameters())
    apn_params = list(net.apn1.parameters()) + list(net.apn2.parameters())

    cls_opt = optim.SGD(cls_params, lr=0.001, momentum=0.9)
    apn_opt = optim.SGD(apn_params, lr=0.001, momentum=0.9)

    trainset = CUB200_loader('../CUB_200_2011', split='train')
    testset = CUB200_loader('../CUB_200_2011', split='test')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, collate_fn=trainset.CUB_collate, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, collate_fn=testset.CUB_collate, num_workers=4)
    sample1, sample2 = random_sample(testloader)
    log_path = os.path.join(save_path, 'log')
    image_path = os.path.join(save_path, 'image')
    writer = SummaryWriter(log_path)
    total_step_1, total_step_2 = 0, 0

    for epoch in range(100):

        total_step_1 = train(net, trainloader, cls_opt, epoch, 'backbone', writer=writer, total_step=total_step_1)
        total_step_2 = train(net, trainloader, apn_opt, epoch, 'apn', writer=writer, total_step=total_step_2)
        test(net, testloader, writer=writer, epoch=epoch)

        # visualize cropped inputs
        _, _, _, resized_1 = net(sample1.unsqueeze(0))
        x1, x2 = resized_1[0].data, resized_1[1].data
        _, _, _, resized_2 = net(sample2.unsqueeze(0))
        x3, x4 = resized_2[0].data, resized_2[1].data

        save_img(x1, path=os.path.join(image_path, f'epoch_{epoch}@2x_1.jpg'))
        save_img(x2, path=os.path.join(image_path, f'epoch_{epoch}@4x_1.jpg'))
        save_img(x3, path=os.path.join(image_path, f'epoch_{epoch}@2x_2.jpg'))
        save_img(x4, path=os.path.join(image_path, f'epoch_{epoch}@4x_2.jpg'))


        # save model per 10 epoches
        if epoch % 10 == 9:
            stamp = f'e{epoch}{int(time.time())}'
            torch.save(net.state_dict, f'{save_path}/racnn-e{epoch}s{stamp}.pt')
            print(f' :: Saved model dict as:\t{save_path}/racnn-e{epoch}s{stamp}.pt')
            torch.save(cls_opt.state_dict(), f'{save_path}/clc_optimizer-e{epoch}s{stamp}.pt')
            torch.save(apn_opt.state_dict(), f'{save_path}/apn_optimizer-e{epoch}s{stamp}.pt')


if __name__ == "__main__":
    save_path = './racnn_result'
    clean(path=save_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run(pretrained_model='build/racnn_pretrained.pt', save_path=save_path)
    build_gif(pattern='@2x_1', gif_name='racnn_cub200', cache_path=save_path)
    build_gif(pattern='@4x_1', gif_name='racnn_cub200', cache_path=save_path)
    build_gif(pattern='@2x_2', gif_name='racnn_cub200', cache_path=save_path)
    build_gif(pattern='@4x_2', gif_name='racnn_cub200', cache_path=save_path)
