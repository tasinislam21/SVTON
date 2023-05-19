import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import cv2
from models import networks, dataset
from torch.autograd import Variable
from PIL import Image
import os

class Args:
    batchSize = 1
    #dataroot = '../../DeepFashion_Try_On/acgpn_dataset'
    #datapairs = 'short_long.txt'
    dataroot = 'data'
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
    gmm = networks.GMM(7,3)
    gmm.cuda()
    gmm.load_state_dict(torch.load('checkpoint/GMM.pth'))
    gmm.eval()

with torch.no_grad():
    tom = networks.TOM(11, 3)
    tom.cuda()
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

def encode(label_map, size):
    label_nc = 14
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

transform_A = dataset.get_transform(normalize=False)
tanh = torch.nn.Tanh()
sigmoid = nn.Sigmoid()
step = 0

if not os.path.isdir('result_name'):
    os.mkdir('result_name')
for data in dataloader:
    h_name = data['human_name']
    mask_clothes = (data['label'] == 4).float()
    mask_fore = (data['label'] > 0).float()
    img_fore = data['image'] * mask_fore
    in_label = Variable(data['label'].cuda())
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
    dis_label = generate_discrete_label(arm_label.detach(), 4)
    fake_cl = (armlabel_map == 1).float()
    arm1_mask = (in_label == 11).float()
    arm2_mask = (in_label == 13).float()
    skin_color = ger_average_color((arm1_mask + arm2_mask - arm2_mask * arm1_mask),
                (arm1_mask + arm2_mask - arm2_mask * arm1_mask) * in_image)
    new_arm1_mask = (armlabel_map == 2).float()
    new_arm2_mask = (armlabel_map == 3).float()

    arm1_occ = in_mask_clothes * new_arm1_mask
    arm2_occ = in_mask_clothes * new_arm2_mask
    bigger_arm1_occ = arm1_occ
    bigger_arm2_occ = arm2_occ

    occlude = (1 - bigger_arm1_occ * (arm2_mask + arm1_mask+in_mask_clothes)) * \
              (1 - bigger_arm2_occ * (arm2_mask + arm1_mask+in_mask_clothes))
    img_hole_hand = in_img_fore * (1 - in_mask_clothes) * occlude * (1 - fake_cl)
    dis_label = encode(armlabel_map, armlabel_map.shape)
    fake_c, warped = gmm(clothes, fake_cl, in_skeleton)
    fake_c=tanh(fake_c)

    generate_map = torch.zeros([opt.batchSize, 1, 256, 192])
    generate_map = generate_map.cuda()

    ola = (in_label == 13).float()
    ora = (in_label == 11).float()

    gla = new_arm2_mask
    gra = new_arm1_mask
    torso = (armlabel_map == 1).float()

    la = ola * (1 - gla)
    ra = ora * (1 - gra)

    resultant_la = ola * (1 - la)
    resultant_ra = ora * (1 - ra)

    final_la = gla * (1 - resultant_la)
    final_ra = gra * (1 - resultant_ra)
    generate_map = generate_map + (final_la * 3)
    generate_map = generate_map + (final_ra * 2)
    generate_map = generate_map + (torso * 1)

    tom_in = torch.cat([img_hole_hand, generate_map, fake_c, skin_color, gen_noise(shape)], 1)
    fake_image = tom(tom_in.detach())
    fake_image = tanh(fake_image)

    tensor = (fake_image[0].clone() + 1) * 0.5 * 255
    tensor = tensor.cpu().clamp(0, 255)
    array = tensor.detach().numpy().astype('uint8')
    array = array.swapaxes(0, 1).swapaxes(1, 2)
    image_pil = Image.fromarray(array)
    image_pil.save("result_name/"+h_name[0])
    step += 1