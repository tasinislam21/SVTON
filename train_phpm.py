import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.utils.data as data
import torch.optim as optim
import torchvision.utils
from torch.autograd import Variable
import torchgeometry as tgm
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from models import networks, dataset
import argparse

device = "cuda"
from distributed import (
    get_rank,
    synchronize,
)

class Args:
    batchSize = 8
    dataroot = 'data'
    datapairs = 'train_pairs.txt'
    phase = 'train'
    beta1 = 0.5
opt = Args


parser = argparse.ArgumentParser(description="Pose with Style trainer")
parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
args = parser.parse_args()
torch.distributed.init_process_group(backend="nccl", init_method="env://")
torch.cuda.set_device(args.local_rank)
synchronize()

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

train_dataset = dataset.BaseDataset(opt)
sampler = data_sampler(train_dataset, shuffle=True, distributed=True)
dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    sampler=sampler,
    shuffle=False)

if get_rank() == 0:
    if not os.path.isdir('checkpoint_phpm'):
        os.mkdir('checkpoint_phpm')
    writer = SummaryWriter('runs/phpm')

opt_train = Args

G1 = networks.PHPM(7, 4).to(device)
optimizerG = optim.Adam(G1.parameters(), lr=0.0002, betas=(opt_train.beta1, 0.999))

G1 = nn.parallel.DistributedDataParallel(
        G1,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False
    )

def extractChannel(label):
    up = nn.Upsample(size=(256, 192), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss.to(device)

    parse_pred = gauss(up(label))
    parse_pred = parse_pred.argmax(dim=1)[:, None]
    parse_old = torch.zeros(parse_pred.size(0), 14, 256, 192, dtype=torch.float).to(device)
    parse_old.scatter_(1, parse_pred, 1.0)
    labels = {
        0:  ['background',  [0]],
        1:  ['cloth',       [4]],
        2:  ['arm1',       [11]],
        3:  ['arm2',        [13]]
    }
    parse = torch.zeros(parse_pred.size(0), 4, 256, 192, dtype=torch.float).to(device)
    for j in range(len(labels)):
        for label in labels[j][1]:
            parse[:, j] += parse_old[:, label]
    return parse

def generate_discrete_label(inputs, label_nc, onehot=True):
    pred_batch = []
    size = inputs.size()
    for input in inputs:
        input = input.view(1, label_nc, size[2], size[3])
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)
    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_map = []
    for p in pred_batch:
        p = p.view(1, 256, 192)
        label_map.append(p)
    label_map = torch.stack(label_map, 0)
    if not onehot:
        return label_map.float().cuda()
    size = label_map.size()
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_().to(device)
    input_label = input_label.scatter_(1, label_map.data.long().to(device), 1.0)
    return input_label

def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise.to(device)

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, _,ht, wt = target.size()
    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    target = target.type(torch.int64)
    loss = F.cross_entropy(input, target)
    return loss

sigmoid = nn.Sigmoid()
step = 0

for epoch in range(20):
    for data in dataloader: #training
        mask_clothes = (data['label'] == 4).float().to(device)
        mask_hair = (data['label'] == 1).float().to(device)
        mask_bottom = (data['label'] == 8).float().to(device)
        mask_head = (data['label'] == 12).float().to(device)

        mask_fore = (data['label'] > 0).float().to(device)

        in_label = Variable(data['label'].to(device))
        in_edge = Variable(data['edge'].to(device))
        in_mask_clothes = Variable(mask_clothes.to(device))
        in_color = Variable(data['color'].to(device))
        in_image = Variable(data['image'].to(device))
        in_skeleton = Variable(data['skeleton'].to(device))
        in_mask_fore = Variable(mask_fore.to(device))
        in_blurry = Variable(data['blurry'].to(device))
        pre_clothes_mask = (in_edge > 0.5).float().to(device)
        img_fore = in_image * mask_fore
        in_img_fore = Variable(img_fore.to(device))
        shape = pre_clothes_mask.shape
        clothes = in_color*pre_clothes_mask
        shape = pre_clothes_mask.shape

        G1_in = torch.cat([in_blurry, clothes, in_skeleton], dim=1)
        arm_label = G1(G1_in)
        arm_label = sigmoid(arm_label)

        size = in_label.size()
        wanted_feature = (in_label * (1 - mask_hair) * (1 - mask_bottom) *
                          (1 - mask_head))
        # Despite removing some labels, cross-entropy will see this as having 14 channels
        oneHot_size = (size[0], 14, size[2], size[3])
        input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_().to(device)
        ground_truth_14 = input_label.scatter_(1, wanted_feature.data.long().to(device), 1.0)
        ground_truth_4 = extractChannel(ground_truth_14)  # changes 14 channel to 4
        ground_truth = generate_discrete_label(ground_truth_4.detach(), 4, False)
        CE_loss = cross_entropy2d(arm_label, ground_truth) * 10
        armlabel_map = generate_discrete_label(arm_label.detach(), 4, False)

        loss_G = CE_loss
        if get_rank() == 0:
            if step % 60 == 0:
                writer.add_scalar('loss_G', loss_G, step)

            if step % 450 == 0:
                writer.add_image('torso_label', torchvision.utils.make_grid(armlabel_map), step)
                writer.add_image('gt', torchvision.utils.make_grid(ground_truth), step)

        optimizerG.zero_grad()
        loss_G.backward()
        optimizerG.step()
        step += 1
    if get_rank() == 0:
        torch.save(G1.module.state_dict(), "checkpoint_phpm/phpm_"+str(epoch)+".pth")