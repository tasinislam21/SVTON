import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import cv2
import tqdm

from models import networks, dataset
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms

mean_candidate = [0.74112587, 0.69617281, 0.68865463]
std_candidate = [0.2941623, 0.30806473, 0.30613222]

inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean_candidate, std_candidate)],
    std=[1/s for s in std_candidate]
)

class Args:
    batchSize = 1
    dataroot = 'viton_hd_dataset'
    datapairs = 'shuffled_test_pairs.txt'
    phase = 'test'
opt = Args

t = dataset.BaseDataset(opt)
dataloader = torch.utils.data.DataLoader(
            t,
            batch_size=opt.batchSize,
            shuffle=False)

with torch.no_grad():
    phpm = networks.PHPM(7, 4)
    phpm.cuda()
    phpm.load_state_dict(torch.load('checkpoint/PHPM.pth'))
    phpm.eval()

with torch.no_grad():
    gmm = networks.GMM(7,3).cuda()
    gmm.load_state_dict(torch.load('checkpoint/GMM.pth'))
    gmm.eval()

with torch.no_grad():
    tom = networks.TOM(11, 3).cuda()
    tom.load_state_dict(torch.load('checkpoint/TOM.pth'))
    tom.eval()

def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise.cuda()

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
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
    return input_label

def ger_average_color(mask, arms):
    color = torch.zeros(arms.shape).cuda()
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

sigmoid = nn.Sigmoid()

for data in tqdm.tqdm(dataloader):
    h_name = data['name']

    label = data['label']
    mask_hair = (label == 2).float() * 1
    mask_clothes = (label == 5).float() * 4
    mask_bottom = (label == 9).float() * 8
    mask_face = (label == 13).float() * 12
    mask_arm1 = (label == 14).float() * 11
    mask_arm2 = (label == 15).float() * 13
    label = (mask_hair + mask_clothes + mask_bottom + mask_face + mask_arm1 + mask_arm2).float()

    mask_clothes = (label == 4).float()
    mask_fore = (label > 0).float()

    img_fore = data['image'] * mask_fore
    in_label = Variable(label.cuda())
    in_edge = Variable(data['edge'].cuda())
    in_img_fore = Variable(img_fore.cuda())
    in_mask_clothes = Variable(mask_clothes.cuda())
    in_color = Variable(data['color'].cuda())
    in_image = Variable(data['image'].cuda())
    in_mask_fore = Variable(mask_fore.cuda())
    in_skeleton = Variable(data['skeleton'].cuda())
    in_blurry = Variable(data['blurry'].cuda())
    pre_clothes_mask = (in_edge > 0.5).float()
    clothes = in_color*pre_clothes_mask
    shape = pre_clothes_mask.shape
    phpm_in = torch.cat([in_blurry, clothes, in_skeleton], dim=1)
    arm_label = phpm(phpm_in)
    arm_label = sigmoid(arm_label)
    armlabel_map = generate_discrete_label(arm_label.detach(), 4, False)
    fake_cl = (armlabel_map == 1).float()
    arm1_mask = (in_label == 11).float()
    arm2_mask = (in_label == 13).float()
    skin_color = ger_average_color((arm1_mask + arm2_mask - arm2_mask * arm1_mask),
                (arm1_mask + arm2_mask - arm2_mask * arm1_mask) * in_image)

    new_arm1_mask = (armlabel_map == 2).float()
    new_arm2_mask = (armlabel_map == 3).float()

    fake_c, warped = gmm(clothes, fake_cl, in_skeleton)
    fake_c *= fake_cl

    new_arms = new_arm1_mask + new_arm2_mask
    old_arms = arm1_mask + arm2_mask

    u = (new_arms * (1 - old_arms)).cuda()

    M_t = (armlabel_map == 1).float().cuda()
    img_fore *= (1 - mask_clothes)
    I_p = (1 - (u + M_t)) * img_fore.cuda()

    u1 = new_arm1_mask * (1 - arm1_mask) * 2
    u2 = new_arm2_mask * (1 - arm2_mask) * 3
    torso = (armlabel_map == 1).float()
    M_g = u1 + u2 + torso

    tom_in = torch.cat([I_p, M_g, fake_c, skin_color, gen_noise(shape)], 1)
    fake_image = tom(tom_in)
    fake_image = inv_normalize(fake_image)

    fake_image = fake_image[0].clone() * 255
    fake_image = fake_image.cpu().clamp(0, 255)
    fake_image = fake_image.detach().numpy().astype('uint8')
    fake_image = Image.fromarray(fake_image.transpose((1, 2, 0)))

    fake_image.save("viton_hd_result/unpaired_setting/"+h_name[0])