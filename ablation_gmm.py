import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from models import networks, dataset
from PIL import Image
import numpy as np

device = "cuda"


mean_candidate = [0.74112587, 0.69617281, 0.68865463]
std_candidate = [0.2941623, 0.30806473, 0.30613222]

mean_clothing = [0.73949153, 0.70635068, 0.71736564]
std_clothing = [0.34867646, 0.36374153, 0.35065262]

inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean_candidate, std_candidate)],
    std=[1/s for s in std_candidate]
)

class Args:
    batchSize = 1
    dataroot = 'viton_dataset'
    datapairs = 'custom_test.txt'
    phase = 'test'
    beta1 = 0.5
opt_train = Args

train_dataset = dataset.BaseDataset(opt_train)
train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1)

with torch.no_grad():
    phpm = networks.PHPM(7, 4)
    phpm.cuda()
    phpm.load_state_dict(torch.load('checkpoint/PHPM.pth'))
    phpm.eval()

with torch.no_grad():
    gmm = networks.GMM(7,3).to(device)
    gmm.load_state_dict(torch.load('checkpoint/GMM.pth'))
    gmm.eval()

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


sigmoid = torch.nn.Sigmoid()
step = 0
for data in train_dataloader:
    step += 1
    h_name = data['name']
    mask_clothes = (data['label'] == 4).float().cuda()
    in_image = data['image'].cuda()
    in_edge = data['edge'].cuda()
    in_color = data['color'].cuda()
    in_skeleton = data['skeleton'].cuda()
    in_blurry = data['blurry'].cuda()
    pre_clothes_mask = (in_edge > 0.5).float().cuda()
    clothes = in_color*pre_clothes_mask

    phpm_in = torch.cat([in_blurry, clothes, in_skeleton], dim=1)
    arm_label = phpm(phpm_in)
    arm_label = sigmoid(arm_label)

    armlabel_map = generate_discrete_label(arm_label.detach(), 4, False)
    fake_cl = (armlabel_map == 1).float()

    fake_c, affine = gmm(clothes, fake_cl, in_skeleton)
    fake_c *= fake_cl

    fake_image = inv_normalize(fake_c)
    fake_image = fake_image[0].clone() * 255
    fake_image = fake_image.cpu().clamp(0, 255)
    fake_image = fake_image.detach().numpy().astype('uint8')
    fake_image = Image.fromarray(fake_image.transpose((1, 2, 0)))
    fake_image.save("temp/" + str(step)+".jpg")