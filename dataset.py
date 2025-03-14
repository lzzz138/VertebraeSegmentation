
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import albumentations as A
import numpy as np
from PIL import Image
import os

label_mapping = {
    0 : 0, #背景
    128 : 1, #锥体
    255 : 2 #椎弓
}
inverse_mapping = {v:k for k,v in label_mapping.items()}

trans=A.Compose([
    A.VerticalFlip(p=0.5), #垂直旋转
    A.HorizontalFlip(p=0.5),#水平旋转
    A.RandomBrightnessContrast(p=0.2),# 随机明亮对比度
    #A.Resize(height=256,width=256,interpolation=cv2.INTER_NEAREST)
])

root = 'spineCT/spineCT'

class SpineDataset(Dataset):
    def __init__(self, root, status, transform=None):
        super().__init__()
        self.root = root
        self.data_paths = []
        self.label_paths = []
        self.transform = transform
        self.status = status
        self.as_tensor = T.ToTensor()

        ct_paths = os.path.join(root, status)
        self.ct_names = os.listdir(ct_paths)
        for ct_name in self.ct_names:
            img_names = os.listdir(os.path.join(ct_paths, ct_name, 'Image'))
            label_names = os.listdir(os.path.join(ct_paths, ct_name, 'GT'))

            for img_name in img_names:
                self.data_paths.append(os.path.join(ct_paths, ct_name, 'Image',img_name))

            for label_name in label_names:
                self.label_paths.append(os.path.join(ct_paths, ct_name, 'GT',label_name))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        label_path = self.label_paths[idx]
        #img = Image.open(data_path).convert('RGB')
        #label = Image.open(label_path).convert('L')

        img = cv2.imread(data_path)
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        if self.transform is not None:
            augments = self.transform(image=img, mask=label)
            img = augments['image']
            label = augments['mask']

        img = self.as_tensor(img)

        # 转换为Numpy数组
        #label = np.array(label)
        # 映射标签值到连续索引
        label = np.vectorize(label_mapping.get)(label)
        label = torch.from_numpy(label).long()  # (H, W)

        return img, label


if __name__ == '__main__':
    train_dataset = SpineDataset(root=root, status='train', transform=trans)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    data_iter = iter(train_dataloader)
    images, labels = next(data_iter)






