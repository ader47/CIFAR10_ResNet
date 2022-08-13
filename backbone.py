import torch
import torch.nn as nn



# nn.Module是nn中十分重要的类，包含网络各层的定义及forward方法
class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,inchannel,outchannel,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.Conv1=nn.Conv2d(in_channels=inchannel,out_channels=outchannel,
                             kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(outchannel)
        self.relu=nn.ReLU()
        # out channels 和卷积核的个数有关
        self.Conv2=nn.Conv2d(in_channels=outchannel,out_channels=outchannel,
                             kernel_size=3,stride=1,padding=1,bias=False)
        # 为什么相同的batchnormalization要定义两次呢？不能重复使用吗？
        self.bn2=nn.BatchNorm2d(outchannel)
        self.downsample=downsample

    def forward(self,x):
        identity=x
        if self.downsample is not None:
            identity=self.downsample(identity)
        out=self.Conv1(x)
        #先batch 再relu
        out=self.bn1(out)
        out=self.relu(out)
        out=self.Conv2(out)
        out=self.bn2(out)
        # if self.downsample is not None:
        #     print('out')
        #     print(out.shape)
        #     print('identity')
        #     print(identity.shape)
        out+=identity
        out=self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,inchannel,outchannel,stride=1,downsample=None):
        super(Bottleneck, self).__init__()
        self.Conv1=nn.Conv2d(in_channels=inchannel,out_channels=outchannel,
                             stride=1,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(outchannel)
        self.relu=nn.ReLU(inplace=True)
        self.Conv2=nn.Conv2d(in_channels=outchannel,out_channels=outchannel,
                             stride=2,kernel_size=3,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(outchannel)
        self.Conv3=nn.Conv2d(in_channels=outchannel,out_channels=outchannel*self.expansion,
                             stride=1,kernel_size=1,bias=False)
        # 这里是outchannel*4，最后输出是outchannel的四倍
        self.bn3=nn.BatchNorm2d(outchannel*self.expansion)
        self.downsample=downsample

    def forward(self,x):
        identity=x
        if self.downsample is not None:
            identity=self.downsample(identity)
        out=self.Conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.Conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.Conv3(out)
        out=self.bn3(out)
        out+=identity
        out=self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,blocktype,blocks_num,num_classes=1000,include_top=True):
        super(ResNet, self).__init__()
        self.include_top=include_top
        self.inchannel=64

        self.Conv1=nn.Conv2d(in_channels=3,out_channels=self.inchannel,
                             kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.inchannel)
        self.relu=nn.ReLU(inplace=True)

        #todo
        # maxpool
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=self._make_layer(blocktype,64,blocks_num[0])
        self.layer2 = self._make_layer(blocktype, 128, blocks_num[1],stride=2)
        self.layer3 = self._make_layer(blocktype, 256, blocks_num[2],stride=2)
        self.layer4 = self._make_layer(blocktype, 512, blocks_num[3],stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * blocktype.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self,blocktype,channel,block_num,stride=1):
        downsample=None
        if stride!=1 or self.inchannel!=channel*blocktype.expansion:
            downsample=nn.Sequential(
                #todo
                # 输入数量梳理
                nn.Conv2d(self.inchannel,channel*blocktype.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(channel*blocktype.expansion)
            )
        layers=[]
        layers.append(
            blocktype(self.inchannel,channel,downsample=downsample,stride=stride)
        )
        #这是指一个残差块的输入输出，self。inchannel是指一个残差块的输入，channel
        self.inchannel=channel*blocktype.expansion
        for i in range (1,block_num):
            layers.append(blocktype(self.inchannel,channel))
        return nn.Sequential(*layers)
    def forward(self,x):
        x=x.to(torch.float32)
        out=self.Conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)
        out=self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.include_top:
            out=self.avgpool(out)
            #todo
            # 这里是为什么要展开呢？
            out = torch.flatten(out, 1)
            out=self.fc(out)
        return out