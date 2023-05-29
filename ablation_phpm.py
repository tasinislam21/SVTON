import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import tqdm
from torch.autograd import Variable
import torchgeometry as tgm
from models import networks, dataset
import statistics
from torchmetrics import StructuralSimilarityIndexMeasure
device = "cuda"

class Args:
    batchSize = 1
    dataroot = 'viton_dataset'
    datapairs = 'test_pairs.txt'
    phase = 'test'
    beta1 = 0.5
opt = Args


train_dataset = dataset.BaseDataset(opt)
dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize)

with torch.no_grad():
    G1 = networks.PHPM_OLD(7, 4).to(device)
    G1.load_state_dict(torch.load('checkpoint/phpm_standard.pth'))
    G1.eval()

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

ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
def compare(imageA, imageB):
    s = ssim(imageA, imageB)
    return s

sigmoid = nn.Sigmoid()
SSIM_array = []
for data in tqdm.tqdm(dataloader): #training
    h_name = data['name']
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
    ground_truth = extractChannel(ground_truth_14)  # changes 14 channel to 4

    arm_label = generate_discrete_label(arm_label.detach(), 4, False)
    ground_truth = generate_discrete_label(ground_truth.detach(), 4, False)

    ssim_value = compare(arm_label, ground_truth)
    SSIM_array.append(ssim_value.item())
print(statistics.mean(SSIM_array))