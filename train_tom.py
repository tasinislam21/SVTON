import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import cv2
import torch.utils.data as data
from torch.autograd import Variable
from models import networks, dataset
import torch.optim as optim
import torchgeometry as tgm
import torchvision.transforms as transforms
import torchvision
import argparse

device = "cuda"
from distributed import (
    get_rank,
    synchronize,
)

mean_candidate = [0.74112587, 0.69617281, 0.68865463]
std_candidate = [0.2941623, 0.30806473, 0.30613222]

inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean_candidate, std_candidate)],
    std=[1/s for s in std_candidate]
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
    writer = SummaryWriter('runs/TOM')


def encode(label_map, size):
    label_nc = 14
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

def encode_input(label_map):
    size = label_map.size()
    oneHot_size = (size[0], 14, size[2], size[3])
    input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_().to(device)
    input_label = input_label.scatter_(1, label_map.data.long().to(device), 1.0)
    input_label = Variable(input_label)
    return input_label

def ger_average_color(mask, arms):
    color = torch.zeros(arms.shape)
    for i in range(arms.shape[0]):
        count = len(torch.nonzero(mask[i, :, :, :]))
        if count < 10:
            color[i, 0, :, :] = 0
            color[i, 1, :, :] = 0
            color[i, 2, :, :] = 0
        else:
            color[i, 0, :, :] = arms[i, 0, :, :].sum() / count
            color[i, 1, :, :] = arms[i, 1, :, :].sum() / count
            color[i, 2, :, :] = arms[i, 2, :, :].sum() / count
    return color

tom = networks.TOM(11, 3).to(device)
discriminator = networks.Discriminator(14).to(device)
optimizerG = optim.AdamW(tom.parameters(), lr=0.0002, betas=(opt.beta1, 0.999))
optimizerD = optim.AdamW(discriminator.parameters(), lr=0.0002, betas=(opt.beta1, 0.999))

tom = nn.parallel.DistributedDataParallel(
        tom,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False,
    )

discriminator = nn.parallel.DistributedDataParallel(
        discriminator,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False
    )

def discriminate(netD ,input_label, real_or_fake):
    input = torch.cat([input_label, real_or_fake], dim=1)
    return netD.forward(input)

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
        return label_map.float().to(device)
    size = label_map.size()
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_().to(device)
    input_label = input_label.scatter_(1, label_map.data.long().to(device), 1.0)
    return input_label

sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
step = 0

criterionVGG = networks.VGGLoss()
criterionGAN = networks.GANLoss(use_lsgan=False, tensor=torch.cuda.FloatTensor)
criterionFeat = torch.nn.L1Loss()

if not os.path.isdir('checkpoint_tom') and get_rank() == 0:
    os.mkdir('checkpoint_tom')

for epoch in range(100):
    for data in dataloader:
        step += 1
        t_mask = (data['label'] == 7).int()
        data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        mask_clothes = (data['label'] == 4).int()
        mask_hair = (data['label'] == 1).int()
        in_mask_hair_label = Variable(mask_hair.to(device))
        mask_bottom = (data['label'] == 8).int()
        in_mask_bottom_label = Variable(mask_bottom.to(device))
        mask_head = (data['label'] == 12).int()
        in_mask_head_label = Variable(mask_head.to(device))
        in_label = Variable(data['label'].to(device))
        size = in_label.size()

        segment = (in_label * (1 - in_mask_hair_label) * (1 - in_mask_bottom_label) *
                          (1 - in_mask_head_label)).transpose(0, 1).long()
        oneHot_size = (size[0], 14, size[2], size[3])
        input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_().to(device)
        segment_14 = input_label.scatter_(1, segment.data.long().to(device), 1.0)
        segment_4 = extractChannel(segment_14)  # changes 14 channel to 4
        segment = generate_discrete_label(segment_4.detach(), 4, False)

        in_image = Variable(data['image'].to(device))
        torso_label = (segment == 1).to(torch.float32).to(device)
        warped_garment = in_image * torso_label
        in_clothes = Variable(data['color'].to(device))
        mask = Variable(data['mask'].to(device))
        mask_fore = (in_label > 0).int()
        img_fore = (in_image * mask_fore).to(device)
        shape = segment.shape
        arm1_mask = (segment == 2).int().to(device)
        arm2_mask = (segment == 3).int().to(device)
        img_hole_hand = img_fore * (1 - torso_label) * (1 - arm1_mask) * (
                1 - arm2_mask) + img_fore * arm1_mask * (
                                1 - mask) + img_fore * arm2_mask * (1 - mask)
        skin_color = ger_average_color((arm1_mask + arm2_mask - arm2_mask * arm1_mask),
                    (arm1_mask + arm2_mask - arm2_mask * arm1_mask) * in_image)
        G_in = torch.cat([img_hole_hand, segment, warped_garment, skin_color.to(device), gen_noise(shape)], 1)
        G_in = G_in.float()
        tryon_sythesis = tom(G_in)
        tryon_sythesis = tanh(tryon_sythesis)

        input_pool = G_in
        real_pool = in_image
        fake_pool = tryon_sythesis
        D_pool = discriminator

        pred_fake = discriminate(D_pool, input_pool.detach(), fake_pool.detach())
        loss_D_fake = criterionGAN(pred_fake, False)
        pred_real = discriminate(D_pool, input_pool.detach(), real_pool.detach())
        loss_D_real = criterionGAN(pred_real, True)

        pred_fake2 = D_pool.forward(torch.cat((input_pool.detach(), fake_pool.detach()), dim=1))
        loss_G_GAN = criterionGAN(pred_fake2, True)

        loss_G_VGG = criterionVGG(fake_pool, real_pool) * 20
        L1_loss = criterionFeat(fake_pool, real_pool) * 0.2

        loss_D = (loss_D_fake + loss_D_real) * 10 # Make loss bigger to make life easier for the generator
        loss_G = loss_G_GAN + loss_G_VGG + L1_loss

        if step % 30 == 0 and get_rank() == 0:
            writer.add_scalar('loss_g', loss_G, step)
            writer.add_scalar('loss_d', loss_D, step)
        if step % 100 == 0 and get_rank() == 0:
            writer.add_image('generated synthesis', torchvision.utils.make_grid(inv_normalize(fake_pool)), step)
            writer.add_image('ground_truth', torchvision.utils.make_grid(inv_normalize(real_pool)), step)

        optimizerG.zero_grad()
        loss_G.backward()
        optimizerG.step()

        optimizerD.zero_grad()
        loss_D.backward()
        optimizerD.step()

        # if step % 5000 == 0:
        #     torch.save({
        #     'tom': tom.state_dict(),
        #     'disc': discriminator.state_dict(),
        #     'optimizer_tom': optimizerG.state_dict(),
        #     'optimizer_disc': optimizerD.state_dict()
        #     }, 'checkpoint_tom/model_'+str(step)+'.pth')
    torch.save(tom.state_dict(), 'checkpoint_tom/G3_epoch'+str(epoch)+'.pth')