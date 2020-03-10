import os.path as osp
import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..'))
TRAIN_PATH = osp.join(ROOT_PATH, 'data/Mini-ImageNet/train')
VAL_PATH = osp.join(ROOT_PATH, 'data/Mini-ImageNet/val')
TEST_PATH = osp.join(ROOT_PATH, 'data/Mini-ImageNet/test')

class MiniImageNet2(Dataset):
    def __init__(self, setname, args):
        if setname=='train':
            THE_PATH = TRAIN_PATH
            label_list = os.listdir(TRAIN_PATH)
        elif setname=='test':
            THE_PATH = TEST_PATH
            label_list = os.listdir(THE_PATH)
        elif setname=='val':
            THE_PATH = VAL_PATH
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Wrong setname.')            
        data = []
        label = []

        folders = [osp.join(THE_PATH, label) for label in label_list if os.path.isdir(osp.join(THE_PATH, label))]


        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        if args.pre_augmentation:
            image_size = 80
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.RandomResizedCrop(88),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:
            image_size = 80
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

