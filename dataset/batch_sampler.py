from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np
import random
import backbone

class BatchSampler(Sampler):
    def __init__(self,dataset,n_class,n_num,*args, **kwargs):
        super(BatchSampler, self).__init__(dataset,*args,**kwargs)
        self.n_class=n_class
        self.n_num=n_num
        self.batch_size=n_class*n_num
        self.dataset=dataset
        self.labels=np.array(dataset.labels)
        self.labels_uniq=np.array(list(dataset.labels_uniq))

        self.len=len(dataset) // self.batch_size
        self.lb_data_dict=dataset.lb_data_dict
        #todo
        self.iter_num = len(self.labels_uniq) // self.n_class
    def __iter__(self):
        curr_p=0
        np.random.shuffle(self.labels_uniq)

        for k,v in self.lb_data_dict.items():
            random.shuffle(self.lb_data_dict[k])

        for i in range(self.iter_num):
            label_batch=self.labels_uniq[curr_p:curr_p+self.n_class]
            curr_p+=self.n_class
            idx=[]
            for lb in label_batch:
                if len(self.lb_data_dict[lb][0])>self.n_num:
                    idx_smp = np.random.choice(self.lb_data_dict[lb][0],
                                               self.n_num, replace=False)
                else:
                    idx_smp = np.random.choice(self.lb_data_dict[lb][0],
                            self.n_num, replace = True)
                idx.extend(idx_smp.tolist())
            yield idx
    def __len__(self):
        return self.iter_num

if __name__ == '__main__':
    import dataset
    data=dataset.CIFAR10('cifar-10-batches-py/data_batch_1',True)
    sampler=BatchSampler(data,10,20)
    dl = DataLoader(data,batch_sampler=sampler,num_workers=4)
    diter=iter(dl)
    imgs,lbs =next(diter)
    model=backbone.ResNet(backbone.BasicBlock,[3,4,6,3],10,True)
    out=model(imgs)
    print(out)
    print(imgs.shape)
    print('asd')
    print(lbs)

