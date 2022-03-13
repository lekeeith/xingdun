import torch


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            # 2d 卷积  在 h/w 维度上卷积
            # in_channel  out_channel
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        # 线性层
        self.fc = torch.nn.Linear(14 * 14 * 32, 10)

    def forward(self, x):
        out = self.conv(x)
        # tensor 顺序是 n c h w  -》 number channel height width
        # number 是参数分类的类型数量
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out
