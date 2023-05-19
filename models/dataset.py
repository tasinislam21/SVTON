import torchvision.transforms as transforms
import os
import os.path as osp
import torch.utils.data as data
from PIL import Image
import numpy as np

mean_clothing = [0.73949153, 0.70635068, 0.71736564]
std_clothing = [0.34867646, 0.36374153, 0.35065262]

mean_candidate = [0.74112587, 0.69617281, 0.68865463]
std_candidate = [0.2941623, 0.30806473, 0.30613222]

mean_skeleton = [0.05440789, 0.07170792, 0.04121648]
std_skeleton = [0.20046051, 0.23692659, 0.16482468]

def get_transform(normalize=True, mean=None, std=None):
    transform_list = []
    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(transform_list)


class BaseDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        super(BaseDataset, self).__init__()

        human_names = []
        cloth_names = []
        with open(os.path.join(opt.dataroot, opt.datapairs), 'r') as f:
            for line in f.readlines():
                h_name, c_name = line.strip().split()
                human_names.append(h_name)
                cloth_names.append(c_name)
        self.human_names = human_names
        self.cloth_names = cloth_names
        self.transform_mask = get_transform(normalize=False)
        self.transform_clothes = get_transform(mean=mean_clothing, std=std_clothing)
        self.transform_candidate = get_transform(mean=mean_candidate, std=std_candidate)
        self.transform_skeleton = get_transform(mean=mean_skeleton, std=std_skeleton)

    def __getitem__(self, index):
        c_name = self.cloth_names[index]
        h_name = self.human_names[index]
        A_path = osp.join(self.opt.dataroot, self.opt.phase, self.opt.phase + '_label', h_name.replace(".jpg", ".png"))
        label = Image.open(A_path).convert('L')

        B_path = osp.join(self.opt.dataroot, self.opt.phase, self.opt.phase + '_img', h_name)
        image = Image.open(B_path).convert('RGB')

        E_path = osp.join(self.opt.dataroot, self.opt.phase, self.opt.phase + '_edge', c_name)
        edge = Image.open(E_path).convert('L')

        C_path = osp.join(self.opt.dataroot, self.opt.phase, self.opt.phase + '_color', c_name)
        color = Image.open(C_path).convert('RGB')

        S_path = osp.join(self.opt.dataroot, self.opt.phase, self.opt.phase + '_posergb', h_name)
        skeleton = Image.open(S_path).convert('RGB')

        M_path = osp.join(self.opt.dataroot, self.opt.phase, self.opt.phase + '_imgmask',
                          h_name.replace('.jpg', '.png'))
        mask = Image.open(M_path).convert('L')
        mask_array = np.array(mask)
        parse_shape = (mask_array > 0).astype(np.float32)
        parse_shape_ori = Image.fromarray((parse_shape * 255).astype(np.uint8))
        parse_shape = parse_shape_ori.resize(
            (192 // 16, 256 // 16), Image.BILINEAR)
        mask = parse_shape.resize(
            (192, 256), Image.BILINEAR)

        label_tensor = self.transform_mask(label) * 255
        image_tensor = self.transform_candidate(image)
        edge_tensor = self.transform_mask(edge)
        color_tensor = self.transform_clothes(color)
        skeleton_tensor = self.transform_skeleton(skeleton)
        mask_tensor = self.transform_mask(mask)
        normal_tensor = self.transform_mask(parse_shape_ori)
        return {'label': label_tensor, 'image': image_tensor,
                'edge': edge_tensor, 'color': color_tensor,
                'mask': mask_tensor, 'name': c_name,
                'colormask': mask_tensor, 'skeleton': skeleton_tensor,
                'blurry': mask_tensor, 'normal': normal_tensor}

    def __len__(self):
        return len(self.human_names)