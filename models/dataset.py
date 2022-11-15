import torchvision.transforms as transforms
import os
import os.path as osp
import torch.utils.data as data
from PIL import Image
import numpy as np

def get_transform(normalize=True):
    transform_list = []
    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
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

        transform_A = get_transform(normalize=False)
        label_tensor = transform_A(label) * 255
        transform_B = get_transform()
        image_tensor = transform_B(image)
        edge_tensor = transform_A(edge)
        color_tensor = transform_B(color)
        skeleton_tensor = transform_B(skeleton)
        mask_tensor = transform_A(mask)
        normal_tensor = transform_A(parse_shape_ori)
        return {'label': label_tensor, 'image': image_tensor,
                'edge': edge_tensor, 'color': color_tensor,
                'mask': mask_tensor, 'name': c_name,
                'colormask': mask_tensor, 'skeleton': skeleton_tensor,
                'blurry': mask_tensor, 'normal': normal_tensor}

    def __len__(self):
        return len(self.human_names)