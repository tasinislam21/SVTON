import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from models import networks, dataset
import torchvision
import argparse

device = "cuda"
from distributed import (
    get_rank,
    synchronize,
)

mean_candidate = [0.74112587, 0.69617281, 0.68865463]
std_candidate = [0.2941623, 0.30806473, 0.30613222]

mean_clothing = [0.73949153, 0.70635068, 0.71736564]
std_clothing = [0.34867646, 0.36374153, 0.35065262]

inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean_candidate, std_candidate)],
    std=[1/s for s in std_candidate]
)

clothing_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean_clothing, std_clothing)],
    std=[1/s for s in std_clothing]
)

class Args:
    batchSize = 8
    dataroot = 'data'
    datapairs = 'train_pairs.txt'
    phase = 'train'
    beta1 = 0.5
opt_train = Args

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

train_dataset = dataset.BaseDataset(opt_train)
sampler = data_sampler(train_dataset, shuffle=True, distributed=True)
train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt_train.batchSize,
            sampler=sampler,
            shuffle=False)

if get_rank() == 0:
    writer = SummaryWriter('runs/gmm')

gmm = networks.GMM(7, 3).to(device)
discriminator = networks.Discriminator(10).to(device)
optimizerG = optim.AdamW(gmm.parameters(), lr=0.0002, betas=(opt_train.beta1, 0.999))
optimizerD = optim.AdamW(discriminator.parameters(), lr=0.0002, betas=(opt_train.beta1, 0.999))

gmm = nn.parallel.DistributedDataParallel(
        gmm,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True
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

sigmoid = nn.Sigmoid()
tanh = torch.nn.Tanh()
l1loss = nn.L1Loss()

step = 0
criterionVGG = networks.VGGLoss()
criterionGAN = networks.GANLoss(use_lsgan=False, tensor=torch.cuda.FloatTensor)
criterionFeat = nn.L1Loss()

if get_rank() == 0:
    if not os.path.isdir('checkpoint_gmm'):
        os.mkdir('checkpoint_gmm')

for epoch in range(50):
    for data in train_dataloader:
        mask_clothes = (data['label'] == 4).float().cuda()
        in_image = data['image'].cuda()
        in_edge = data['edge'].cuda()
        in_color = data['color'].cuda()
        in_skeleton = data['skeleton'].cuda()
        pre_clothes_mask = (in_edge > 0.5).float().cuda()
        clothes = in_color*pre_clothes_mask
        fake_c, affine = gmm(clothes, mask_clothes, in_skeleton)
        fake_c = tanh(fake_c)

        input_pool = torch.cat([clothes, mask_clothes, in_skeleton],1)
        real_pool = in_image*mask_clothes
        fake_pool = fake_c*mask_clothes
        D_pool = discriminator

        loss_D_fake = 0
        loss_D_real = 0
        loss_G_VGG = 0

        pred_fake = discriminate(D_pool, input_pool.detach(), fake_pool.detach())
        loss_D_fake += criterionGAN(pred_fake, False)
        pred_real = discriminate(D_pool, input_pool.detach(), real_pool.detach())
        loss_D_real = criterionGAN(pred_real, True)
        pred_fake = D_pool.forward(torch.cat((input_pool.detach(), fake_pool.detach()), dim=1))
        loss_G_GAN = criterionGAN(pred_fake, True)

        loss_G_VGG += criterionVGG(fake_pool, real_pool) * 5
        L1_loss = criterionFeat(fake_pool, real_pool)
        L1_loss += criterionFeat(affine, real_pool)
        loss_D = loss_D_fake + loss_D_real
        loss_G = loss_G_GAN + loss_G_VGG + L1_loss

        optimizerG.zero_grad()
        loss_G.backward()
        optimizerG.step()

        optimizerD.zero_grad()
        loss_D.backward()
        optimizerD.step()

        if step % 300 == 0 and get_rank() == 0:
            writer.add_image('warped_garment', torchvision.utils.make_grid(inv_normalize(fake_c)), step)
            writer.add_image('affine', torchvision.utils.make_grid(clothing_normalize(affine)), step)
            writer.add_image('gt', torchvision.utils.make_grid(inv_normalize(real_pool)), step)

        step += 1
        if step % 20 == 0 and get_rank() == 0:
            writer.add_scalar('generator loss', loss_G, step)
            writer.add_scalar('discriminator loss', loss_D, step)
    if get_rank() == 0:
        torch.save(gmm.state_dict(), "checkpoint_gmm/gmm_"+str(epoch)+".pth")