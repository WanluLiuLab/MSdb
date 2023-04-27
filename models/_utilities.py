import os
import argparse
import sys
import numpy as np
import random
import pandas as pd
from typing import Iterable
import torch
import torch.nn.functional as F
from collections import Counter
from _logger import Colors

class ClassBalancer:
    def __init__(self, classes: Iterable[int]):
        self.classes = classes

    def to_binary_balanced(self, n_per_batch: int = 32, show_progress: bool=False):
        positive_class_number = Counter(self.classes)[1]
        positive_class_index = np.array(list(map(lambda x: x[0], filter(lambda z: z[1] == 1, enumerate(self.classes)))))
        negative_class_index = np.array(list(map(lambda x: x[0], filter(lambda z: z[1] == 0, enumerate(self.classes)))))
        half_n_per_batch = int(n_per_batch / 2)
        j = random.randint(0, len(negative_class_index))
        if int(len(negative_class_index) / positive_class_number) < 1:
            idx = np.hstack([positive_class_index, negative_class_index])
            np.random.shuffle(idx)
            for i in range(0, len(idx), n_per_batch):
                yield idx[i:i+n_per_batch]
        else:
            if positive_class_number > n_per_batch:
                prange = range(0, positive_class_number, half_n_per_batch)
                for i in prange:
                    for _ in range( int(len(negative_class_index) / positive_class_number)):
                        yield np.hstack([positive_class_index[i:i+half_n_per_batch], negative_class_index[j:j+half_n_per_batch]])
                        j += half_n_per_batch
                        if j >= len(negative_class_index):
                            j  = 0
    
    def to_multiple_balanced(self, n_per_batch: int = 32, show_progress: bool=False):
        pass

def get_k_elements(arr: Iterable, k:int):
    return list(map(lambda x: x[k], arr))

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return gpu(i)
    return cpu()

def one_hot_(labels, return_dict = False):
    n_labels_ = np.unique(labels)
    n_labels = dict(zip(n_labels_, range(len(n_labels_))))
    if return_dict:
        return {"one_hot": F.one_hot( torch.tensor(list(map(lambda x: n_labels[x], labels)))), "labels": n_labels}
    return F.one_hot( torch.tensor(list(map(lambda x: n_labels[x], labels))))

def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)
    
def exists(x):
    return x != None

def make_argparser():
    parser = argparse.ArgumentParser(description='Main entrance of scTE')
    parser.add_argument("--no_cuda", action='store_true', default=False, help='whether not to use CUDA in training')
    parser.add_argument("--no_minibatch", action='store_true', default=False, help='whether use minibatch in training')
    parser.add_argument("--highvar", action='store_true', default=False, help='whether use highly variable genes in training')
    parser.add_argument("--dataset", type=str, required = True, 
        choices = ["GSE109555_STRTseq",
                   "GSE109555_STRTseq_DeUMI",
                   "GSE136447_Smartseq2",
                   "GSE140021_10x",
                   "GSE36552_TruSeq",
                   "GSE74767_SC3seq"]
        )

    parser.add_argument("--dataset", type=str, default = './')
    parser.add_argument("--preprocessing", choices=["scale","cpm","magic"], default="scale")
    parser.add_argument('--precisionModel', type=str, default='Float',
                        help='Single Precision/Double precision: Float/Double (default:Float)')
    parser.add_argument('--model', type=str, default="AE", choices=["AE","VAE","GCN","GAT"])
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--l1a', type=float, default=0)
    parser.add_argument('--l2a', type=float, default=0)
    return parser 

def sliceSimutaneuously(a, index):
    return pd.DataFrame(a).iloc[index,index].to_numpy()

def mask_split(tensor, indices):
    sorter = torch.argsort(indices)
    _, counts = torch.unique(indices, return_counts=True)
    return torch.split(tensor[sorter], counts.tolist())


def print_version():
    print(Colors.YELLOW)
    print('Python VERSION:{}\n'.format(Colors.NC), sys.version)
    print(Colors.YELLOW)
    print('pyTorch VERSION:{}\n'.format(Colors.NC), torch.version)
    print(Colors.YELLOW)
    print('CUDA VERSION{}\n'.format(Colors.NC))
    from subprocess import call
    # call(["nvcc", "--version"]) does not work
    print(Colors.YELLOW)
    print('CUDNN VERSION:{}\n'.format(Colors.NC), torch.backends.cudnn.version())
    print(Colors.YELLOW)
    print('Number CUDA Devices:{}\n'.format(Colors.NC), torch.cuda.device_count())
    try:
        print('Devices             ')
        call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    except FileNotFoundError:
        # There is no nvidia-smi in this machine
        pass
    if torch.cuda.is_available():
        print('Active CUDA Device: GPU', torch.cuda.current_device())
        print ('Available devices     ', torch.cuda.device_count())
        print ('Current cuda device   ', torch.cuda.current_device())
    else:
        # cuda not available
        pass

def read_tsv(path, header:bool = True, skip_first_line: bool = False, return_pandas: bool = True):
    result = []
    if os.path.exists(path):
        f = open(path)
        if skip_first_line:
            line = f.readline()
        header_length = None
        if header:
            header = f.readline().strip().split('\t')
            header_length = len(header)

        while 1:
            line = f.readline()
            if not line:
                break
            line = line.strip().split('\t')
            if not header_length:
                header_length = len(line)
            result.append(line[:header_length])
        f.close()
        if return_pandas:
            if header:
                return pd.DataFrame(result, columns = header)
            else:
                return pd.DataFrame(result)
        else:
            return result
    else:
        it = iter(path.split('\n'))
        if skip_first_line:
            line = next(it)
        header_length = None
        if header:
            header = next(it).strip().split('\t')
            header_length = len(header)

        while 1:
            try:
                line = next(it)
                if not line:
                    break
                line = line.strip().split('\t')
                if not header_length:
                    header_length = len(line)
                result.append(line[:header_length])
            except:
                break 
        if return_pandas:
            if header:
                return pd.DataFrame(list(filter(lambda x: len(x) == 125, result)), columns = header)
            else:
                return pd.DataFrame(list(filter(lambda x: len(x) == 125, result)))
        else:
            return result
