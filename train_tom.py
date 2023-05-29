import os
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.utils.data as data
from torch.autograd import Variable
from models import networks, dataset
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import argparse
import numpy as np
import cv2

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

parser = argparse.ArgumentParser(description="Pose with Style trainer")
parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
parser.add_argument("--batchSize", type=int, default=8)
parser.add_argument("--dataroot", type=str, default="data")
parser.add_argument("--datapairs", type=str, default="train_pairs.txt")
parser.add_argument("--phase", type=str, default="train")
parser.add_argument("--beta1", type=float, default=0.5)
opt_train = parser.parse_args()


torch.distributed.init_process_group(backend="nccl", init_method="env://")
torch.cuda.set_device(opt_train.local_rank)
synchronize()

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

train_dataset = dataset.BaseDataset(opt_train)
sampler = data_sampler(train_dataset, shuffle=True, distributed=True)
dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt_train.batchSize,
    sampler=sampler,
    shuffle=False)

if get_rank() == 0:
    writer = SummaryWriter('runs/TOM')

def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise.to(device)

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
optimizerG = optim.Adam(tom.parameters(), lr=0.0002, betas=(opt_train.beta1, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(opt_train.beta1, 0.999))

tom = nn.parallel.DistributedDataParallel(
        tom,
        device_ids=[opt_train.local_rank],
        output_device=opt_train.local_rank,
        broadcast_buffers=False,
    )

discriminator = nn.parallel.DistributedDataParallel(
        discriminator,
        device_ids=[opt_train.local_rank],
        output_device=opt_train.local_rank,
        broadcast_buffers=False
    )

def discriminate(netD ,input_label, real_or_fake):
    input = torch.cat([input_label, real_or_fake], dim=1)
    return netD.forward(input)

sigmoid = nn.Sigmoid()
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
        mask_clothes = (data['label'] == 4).float().to(device)
        mask_hair = (data['label'] == 1).float()
        in_mask_hair_label = Variable(mask_hair.to(device))
        mask_bottom = (data['label'] == 8).int()
        in_mask_bottom_label = Variable(mask_bottom.to(device))
        mask_head = (data['label'] == 12).int()
        in_mask_head_label = Variable(mask_head.to(device))
        in_label = Variable(data['label'].to(device))
        arm1_mask = (in_label == 11).float()
        arm2_mask = (in_label == 13).float()
        mask_fore = (data['label'] > 0).float().to(device)
        in_image = data['image'].to(device)
        img_fore = in_image * mask_fore
        warped_garment = in_image * mask_clothes
        size = in_label.size()

        img_fore *= 1 - (mask_clothes + arm1_mask + arm2_mask)

        u1 = (arm1_mask * 2).to(device)
        u2 = (arm2_mask * 3).to(device)
        M_g = (u1 + u2 + mask_clothes)

        skin_color = ger_average_color((arm1_mask + arm2_mask - arm2_mask * arm1_mask),
                    (arm1_mask + arm2_mask - arm2_mask * arm1_mask) * in_image)


        G_in = torch.cat([img_fore, M_g, warped_garment, skin_color.to(device), gen_noise(arm1_mask.shape)], 1)
        G_in = G_in.float()
        tryon_sythesis = tom(G_in)

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

        if step % 40 == 0 and get_rank() == 0:
            writer.add_scalar('loss_g', loss_G, step)
            writer.add_scalar('loss_d', loss_D, step)
        if step % 500 == 0 and get_rank() == 0:
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
    if get_rank() == 0:
        torch.save(tom.module.state_dict(), 'checkpoint_tom/G3_epoch'+str(epoch)+'.pth')