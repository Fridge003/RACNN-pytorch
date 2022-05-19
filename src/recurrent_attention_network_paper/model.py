import torchstat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import mobilenet
from torchvision.models import vgg19_bn

# add two boxes to the third layer
# control the size of the third scale, no more than half of the last scale
# control the distance between the two boxes of the third scale
# hyperparameters: 1. box length constraint  2.margin  3.the coefficient of box-dist-loss


def cut_outside_parts(box):
    # box is a tensor in the form of [tx, ty, tl]
    tx, ty, tl = box[0], box[1], box[2]

    # to ensure that the box is totally located in the image
    tx = tx if tx > tl else tl
    tx = tx if tx < 1-tl else 1-tl
    ty = ty if ty > tl else tl
    ty = ty if ty < 1-tl else 1-tl

    return tx-tl, tx+tl, ty+tl, ty-tl


class AttentionCropFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, images, locs):
        def h(_x): return 1 / (1 + torch.exp(-10 * _x.float())) # sigmoid function, here k = 10
        in_size = images.size()[2]
        unit = torch.stack([torch.arange(0, in_size)] * in_size)
        x = torch.stack([unit.t()] * 3)
        y = torch.stack([unit] * 3)
        if isinstance(images, torch.cuda.FloatTensor):
            x, y = x.cuda(), y.cuda()

        in_size = images.size()[2]
        ret = []
        for i in range(images.size(0)):
            tx, ty, tl = locs[i][0], locs[i][1], locs[i][2]
            # add constraints to tx, ty, tl
            if in_size == 448: # scale_1 to scale_2
                tl = tl if tl > (in_size / 3) else in_size / 3
            else:  # scale_2 to scale_3a/scale_3b
                tl = tl if tl > (in_size / 4) else in_size / 4
                tl = tl if tl < (in_size * 3 / 8) else in_size * 3 / 8
            tx = tx if tx > tl else tl
            tx = tx if tx < in_size-tl else in_size-tl
            ty = ty if ty > tl else tl
            ty = ty if ty < in_size-tl else in_size-tl

            w_off = int(tx-tl) if (tx-tl) > 0 else 0
            h_off = int(ty-tl) if (ty-tl) > 0 else 0
            w_end = int(tx+tl) if (tx+tl) < in_size else in_size
            h_end = int(ty+tl) if (ty+tl) < in_size else in_size

            mk = (h(x-w_off) - h(x-w_end)) * (h(y-h_off) - h(y-h_end))
            xatt = images[i] * mk

            xatt_cropped = xatt[:, w_off: w_end, h_off: h_end]
            before_upsample = Variable(xatt_cropped.unsqueeze(0))
            # upsample to 224 rather than 448
            xamp = F.upsample(before_upsample, size=(224, 224), mode='bilinear', align_corners=True)
            ret.append(xamp.data.squeeze())

        ret_tensor = torch.stack(ret)
        self.save_for_backward(images, ret_tensor)
        return ret_tensor

    @staticmethod
    def backward(self, grad_output):
        images, ret_tensor = self.saved_variables[0], self.saved_variables[1]
        in_size = 224
        ret = torch.Tensor(grad_output.size(0), 3).zero_()
        norm = -(grad_output * grad_output).sum(dim=1)
        x = torch.stack([torch.arange(0, in_size)] * in_size).t()
        y = x.t()
        long_size = (in_size/3*2)
        short_size = (in_size/3)
        mx = (x >= long_size).float() - (x < short_size).float()
        my = (y >= long_size).float() - (y < short_size).float()
        ml = (((x < short_size)+(x >= long_size)+(y < short_size)+(y >= long_size)) > 0).float()*2 - 1

        mx_batch = torch.stack([mx.float()] * grad_output.size(0))
        my_batch = torch.stack([my.float()] * grad_output.size(0))
        ml_batch = torch.stack([ml.float()] * grad_output.size(0))

        if isinstance(grad_output, torch.cuda.FloatTensor):
            mx_batch = mx_batch.cuda()
            my_batch = my_batch.cuda()
            ml_batch = ml_batch.cuda()
            ret = ret.cuda()

        ret[:, 0] = (norm * mx_batch).sum(dim=1).sum(dim=1)
        ret[:, 1] = (norm * my_batch).sum(dim=1).sum(dim=1)
        ret[:, 2] = (norm * ml_batch).sum(dim=1).sum(dim=1)
        return None, ret


class AttentionCropLayer(nn.Module):
    """
        Crop function sholud be implemented with the nn.Function.
        Detailed description is in 'Attention localization and amplification' part.
        Forward function will not be changed. backward function will not opearate with autograd, but manually implemented function
    """

    def forward(self, images, locs):
        return AttentionCropFunction.apply(images, locs)


class RACNN(nn.Module):
    def __init__(self, num_classes, img_scale=448):
        super(RACNN, self).__init__()

        self.b1 = vgg19_bn(num_classes=num_classes)
        self.b2 = vgg19_bn(num_classes=num_classes)
        self.b3 = vgg19_bn(num_classes=num_classes)
        self.fc3 = nn.Linear(1024, 512)
        self.classifier1 = nn.Linear(512, num_classes)
        self.classifier2 = nn.Linear(512, num_classes)
        self.classifier3 = nn.Linear(512, num_classes)
        self.feature_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.atten_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.crop_resize = AttentionCropLayer()
        self.apn1 = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3),
            nn.Sigmoid(),
        )
        self.apn2 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3),
            nn.Sigmoid(),
        )
        self.apn3 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3),
            nn.Sigmoid(),
        )
        self.echo = None

    def forward(self, x):
        batch_size = x.shape[0]
        rescale_tl = torch.tensor([1, 1, 0.5], requires_grad=False).cuda()
        # forward @scale-1
        # abandon the fc layer and the last CNN layer of the backbone model(VGG Net)
        feature_s1 = self.b1.features(x)  # torch.Size([b, 512, 14, 14])
        pool_s1 = self.feature_pool(feature_s1) # torch.Size([b, 512, 1, 1])
        _attention_s1 = self.apn1(feature_s1.view(-1, 512 * 14 * 14))
        attention_s1 = _attention_s1 * rescale_tl # t_l should be no more than half of the length
        resized_s1 = self.crop_resize(x, attention_s1 * x.shape[-1])  # torch.Size([b, 3, 224, 224])
        # forward @scale-2, produce two sets of attention features
        feature_s2 = self.b2.features(resized_s1)  # torch.Size([b, 512, 7, 7])
        pool_s2 = self.feature_pool(feature_s2)
        _attention_s2 = self.apn2(feature_s2.view(-1, 512 * 7 * 7))
        attention_s2 = _attention_s2 * rescale_tl
        resized_s2 = self.crop_resize(resized_s1, attention_s2 * resized_s1.shape[-1]) # torch.Size([b, 3, 224, 224])
        _attention_s3 = self.apn3(feature_s2.view(-1, 512 * 7 * 7))
        attention_s3 = _attention_s3 * rescale_tl
        resized_s3 = self.crop_resize(resized_s1, attention_s3 * resized_s1.shape[-1])
        # forward @scale-3a / @scale-3b
        feature_s3a = self.b3.features(resized_s2)
        feature_s3b = self.b3.features(resized_s3) # torch.Size([b, 512, 7, 7])
        pool_s3a = self.feature_pool(feature_s3a).view(-1, 512)
        pool_s3b = self.feature_pool(feature_s3b).view(-1, 512)
        pool_s3 = self.fc3(torch.cat((pool_s3a, pool_s3b), dim=-1))
        pred1 = self.classifier1(pool_s1.view(-1, 512))
        pred2 = self.classifier2(pool_s2.view(-1, 512))
        pred3 = self.classifier3(pool_s3.view(-1, 512))

        return [pred1, pred2, pred3], [feature_s1, feature_s2], [attention_s1, attention_s2, attention_s3], [resized_s1, resized_s2, resized_s3]

    def __get_weak_loc(self, features):
        ret = []   # search regions with the highest response value in conv5
        for i in range(len(features)):
            resize = 224 if i >= 1 else 448
            response_map_batch = F.interpolate(features[i], size=[resize, resize], mode="bilinear").mean(1)  # mean along channels
            ret_batch = []
            # response_map: resize * resize
            for response_map in response_map_batch:
                argmax_idx = response_map.argmax()
                ty = (argmax_idx % resize)
                argmax_idx = (argmax_idx - ty)/resize
                tx = (argmax_idx % resize)
                ret_batch.append([(tx*1.0/resize).clamp(min=0.25, max=0.75), (ty*1.0/resize).clamp(min=0.25, max=0.75), 0.25])  # tl = 0.25, fixed, might be modified
            ret.append(torch.Tensor(ret_batch))
        return ret

    @staticmethod
    def multitask_loss(logits, targets):
        loss = []
        for i in range(len(logits)):
            # logits : (num_backbones, batch_size, 200)
            # targets: (batch_size)
            loss.append(F.cross_entropy(logits[i], targets))
        loss = torch.sum(torch.stack(loss))
        return loss

    # the value of margin here can be modified
    @staticmethod
    def rank_loss(logits, targets, margin=0.05):
        preds = [F.softmax(x, dim=-1) for x in logits]
        set_pt = [[scaled_pred[batch_inner_id][target] for scaled_pred in preds] for batch_inner_id, target in enumerate(targets)]
        loss = 0
        for batch_inner_id, pts in enumerate(set_pt):
            loss += (pts[0] - pts[1] + margin).clamp(min=0)
            loss += (pts[1] - pts[2] + margin).clamp(min=0)
        return loss

    # control the cross area of the two boxes in the third layer
    @staticmethod
    def box_area_loss(attens):
        loss = torch.tensor(0.0).cuda()
        scale_3a, scale_3b = attens[1], attens[2]
        batch_size = attens[1].shape[0]
        for batch_id in range(0, batch_size):
            box3a, box3b = scale_3a[batch_id], scale_3b[batch_id]
            l1, r1, u1, d1 = cut_outside_parts(box3a) # left/right/up/low(down) bound of the box
            l2, r2, u2, d2 = cut_outside_parts(box3b)
            if d1 > u2 or d2 > u1 or l1 > r2 or l2 > r1:
                batch_loss = torch.tensor(0.0).cuda()
            else:
                inter_up = u1 if u1 < u2 else u2
                inter_down = d1 if d1 > d2 else d2
                inter_left = l1 if l1 > l2 else l2
                inter_right = r1 if r1 < r2 else r2
                batch_loss = (inter_up - inter_down) * (inter_right - inter_left).cuda()
            # print(batch_loss)
            loss += batch_loss
        return loss

    def __echo_pretrain_apn(self, inputs, optimizer):
        inputs = Variable(inputs).cuda()
        _, features, attens, _ = self.forward(inputs)
        weak_loc = self.__get_weak_loc(features)
        optimizer.zero_grad()
        weak_loss1 = F.smooth_l1_loss(attens[0], weak_loc[0].cuda())
        weak_loss2 = F.smooth_l1_loss(attens[1], weak_loc[1].cuda())
        weak_loss3 = F.smooth_l1_loss(attens[2], weak_loc[1].cuda())
        loss = weak_loss1 + weak_loss2 + weak_loss3
        loss.backward()
        optimizer.step()
        return loss.item()

    def __echo_backbone(self, inputs, targets, optimizer):
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        logits, _, _, _ = self.forward(inputs)
        optimizer.zero_grad()
        loss = self.multitask_loss(logits, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

    def __echo_apn(self, inputs, targets, optimizer):
        lambda_1 = 0.1
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        logits, _, attens, _ = self.forward(inputs)
        optimizer.zero_grad()
        loss = self.rank_loss(logits, targets) + lambda_1 * self.box_area_loss(attens)
        loss.backward()
        optimizer.step()
        return loss.item()

    def mode(self, mode_type):
        assert mode_type in ['pretrain_apn', 'apn', 'backbone']
        if mode_type == 'pretrain_apn':
            self.echo = self.__echo_pretrain_apn
            self.eval()
        if mode_type == 'backbone':
            self.echo = self.__echo_backbone
            self.train()
        if mode_type == 'apn':
            self.echo = self.__echo_apn
            self.eval()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = RACNN(num_classes=8).to(device)
    net.mode('pretrain_apn')
    optimizer = torch.optim.SGD(list(net.apn1.parameters()) + list(net.apn2.parameters()), lr=0.001, momentum=0.9)
    for i in range(50):
        inputs = torch.rand(2, 3, 448, 448)
        print(f':: loss @step{i} : {net.echo(inputs, optimizer)}')
