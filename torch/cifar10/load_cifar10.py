# transforms是用来数据增强
from torchvision import transforms
# DataLoader 是读取 tensor数据  Dataset 是组装数据集
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import glob

label_name = ["airplane", "automobile", "bird",
              "cat", "deer", "dog",
              "frog", "horse", "ship", "truck"]

label_dict = {}

for idx, name in enumerate(label_name):
    label_dict[name] = idx


def default_loader(path):
    return Image.open(path).convert("RGB")


# train_transform = transforms.Compose([
#     transforms.RandomCrop(28),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ])

# compose 用来拼接多个数据增强的方法
train_transform = transforms.Compose([
    # randomCrop 之后进行 Resize，crop会导致图片尺寸减小
    # transforms.RandomResizedCrop((28,28)),
    # 随机的水平翻转 注意图片时是否对翻转敏感 概率 0.5
    transforms.RandomHorizontalFlip(),
    # 随机的垂直翻转 概率 0.5
    transforms.RandomVerticalFlip(),
    # 角度在 -90 - 90 之间旋转
    # transforms.RandomRotation(90),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
    # 随机转换成灰度
    # transforms.RandomGrayscale(0.2),
    # transforms.RandomCrop(28),

    # 把PIL数据转换成 网络输入的数据
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.CenterCrop((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


# train_transform = transforms.Compose([
#     transforms.RandomCrop(28),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     # transforms.RandomRotation(90),
#     # 颜色增强 包含 亮度，对比度等
#     transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
#     transforms.RandomGrayscale(0.2),
#     transforms.ToTensor()
# ])
#
# test_transform = transforms.Compose([
#     transforms.Resize((28, 28)),
#     transforms.ToTensor()
# ])

class MyDataset(Dataset):
    def __init__(self, im_list,
                 transform=None,
                 loader=default_loader):
        super(MyDataset, self).__init__()
        imgs = []

        for im_item in im_list:
            # "/home/kuan/dataset/CIFAR10/TRAIN/" \
            # "airplane/aeroplane_s_000021.png"
            im_label_name = im_item.split(os.sep)[-2]
            imgs.append([im_item, label_dict[im_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        im_path, im_label = self.imgs[index]
        im_data = self.loader(im_path)
        if self.transform is not None:
            im_data = self.transform(im_data)

        return im_data, im_label

    def __len__(self):
        return len(self.imgs)


im_train_list = glob.glob("D:\\projects\\databases\\cifar-10-python\\TRAIN\\*\\*.png")
im_test_list = glob.glob("D:\\projects\\databases\\cifar-10-python\\TEST\\*\\*.png")

train_dataset = MyDataset(im_train_list,
                          transform=train_transform)
test_dataset = MyDataset(im_test_list,
                         transform=test_transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=4)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=128,
                         shuffle=False,
                         num_workers=4)

print("num_of_train", len(train_dataset))
print("num_of_test", len(test_dataset))
