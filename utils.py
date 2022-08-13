import _pickle as cPickle

def unpickle(file):
    with open(file,'rb') as fo:
        dict=cPickle.load(fo,encoding='bytes')
    return dict