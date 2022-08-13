# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import backbone
import torch
import dataset.dataset
import dataset.batch_sampler
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
train_batch=['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
test_batch='test_batch'

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LR = 0.01

if __name__ == '__main__':
    data=dataset.dataset.CIFAR10('dataset/cifar-10-batches-py/data_batch_1',True)
    sampler=dataset.batch_sampler.BatchSampler(data,8,16)
    dl=DataLoader(data,batch_sampler=sampler,num_workers=4)
    diter=iter(dl)

    model = backbone.ResNet(backbone.BasicBlock, [3, 4, 6, 3], 10, True)
    model=model.to(device)

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=5e-4)
    count=0
    batch_id=0
    sum_loss=0
    print(device)
    model.load_state_dict(torch.load('model.pkl'))
    is_train=False

    #Train
    while True and is_train:
        try:
            imgs,labels=next(diter)
        except StopIteration:
            diter=iter(dl)
            imgs,labels=next(diter)
        print(count)
        model.train()
        imgs=imgs.cuda()
        labels=labels.cuda()

        optimizer.zero_grad()
        outputs=model(imgs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

        if count%390==0 and count!=0:
            if count%3900==0 and count!=0:
                data = dataset.dataset.CIFAR10('dataset/cifar-10-batches-py/' + test_batch, False)
                sampler = dataset.batch_sampler.BatchSampler(data, 8, 16)
                dl = DataLoader(data, batch_sampler=sampler, num_workers=4)
                diter = iter(dl)
                acc = 0
                total=0
                while True:
                    try:
                        imgs, labels = next(diter)
                    except StopIteration:
                        break
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    model.eval()
                    torch.no_grad()
                    outputs = model(imgs)
                    # todo
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    acc = 100 * ((labels == predicted).sum()) / total
                print(acc)

            print(sum_loss/(count+1))
            batch_id+=1
            if batch_id==5: batch_id=0
            data = dataset.dataset.CIFAR10('dataset/cifar-10-batches-py/'+train_batch[batch_id], True)
            sampler = dataset.batch_sampler.BatchSampler(data, 8, 16)
            dl = DataLoader(data, batch_sampler=sampler, num_workers=4)
            diter = iter(dl)
            print(train_batch[batch_id])
        count+=1
        if count == 58500 : break

    #Test
    if not is_train:
        data = dataset.dataset.CIFAR10('dataset/cifar-10-batches-py/' + test_batch, False)
        sampler = dataset.batch_sampler.BatchSampler(data, 8, 16)
        dl = DataLoader(data, batch_sampler=sampler, num_workers=4)
        diter = iter(dl)
        acc = 0
        total=0
        right=0
        while True:
            try:
                imgs, labels = next(diter)
            except StopIteration:
                diter = iter(dl)
                imgs, labels = next(diter)
            print('labels:',labels)
            imgs = imgs.to(device)
            labels = labels.to(device)
            model.eval()
            torch.no_grad()
            outputs = model(imgs)
            # todo
            _, predicted = torch.max(outputs.data, 1)
            print('predicted:',predicted)
            total += labels.size(0)
            right+=(labels == predicted).sum()
            print(100*right/total)



    #torch.save(model.state_dict(), 'model.pkl')