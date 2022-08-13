import _pickle as cPickle
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import backbone

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class CIFAR10(Dataset):
    def __init__(self,datapath,is_train=True):
        super(CIFAR10, self).__init__()
        self.datapath=datapath
        dicts=self.unpickle(self.datapath)
        self.labels=dicts[b'labels']
        self.labels_uniq=set(self.labels)
        self.lb_data_dict=dict()
        lb_array=np.array(self.labels)
        for i in self.labels_uniq:
            index=np.where(lb_array==i)
            self.lb_data_dict.update({i:index})
        self.data=dicts[b'data']
        self.classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        dicts.clear()
        # 预处理
        if is_train:
            self.trans=transforms.Compose([
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),

                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.trans=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    #成员函数必须有self，且在第一个？
    def unpickle(self,file):
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo, encoding='bytes')
        return dict

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        #todo
        # 随机数？数据增强和图像翻转
        img = self.data[index]
        img=img.reshape(-1,1024)
        r=img[0].reshape(32,32)
        g=img[1].reshape(32,32)
        b=img[2].reshape(32,32)
        image=np.zeros((32,32,3))
        image[:,:,0]=r
        image[:,:,1]=g
        image[:,:,2]=b
        image=Image.fromarray(np.uint8(image))
        #todo
        # trans 的输入是什么   PIL   ，可以直接输入吗？
        image=self.trans(image)
        image=np.array(image)
        return image,self.labels[index]

if __name__ == '__main__':
    #batch1=unpickle('cifar-10-batches-py/data_batch_1')
    # 在每个key前都要加上b
    CIFAR=CIFAR10('cifar-10-batches-py/data_batch_1',True)
    img,label=CIFAR[10]
    img=torch.tensor(img)
    name=CIFAR.classes[label]
    print(name)
    print(type(img))
    # model=backbone.ResNet(backbone.BasicBlock,[3,4,6,3],10,True)
    # out=model(img)
    # print(out)
    img=np.array(img)
    img=np.moveaxis(img,0,-1)

