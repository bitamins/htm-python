"""
    source: https://www.youtube.com/watch?v=XMB0ri4qgwc&list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&index=1

"""

import math
import numpy as np
from scipy.special import comb
from functools import reduce


def sdr_capacity(n,w):
    """
        Returns information capacity
    """
    capacity = comb(n,w,exact=True)
    return capacity

def sdr_overlap_set(n,w,wx,b):
    """ 
        Overlap set
        n - shape
        wx - subsample on bits
        w - total on bits
        b - overlap
    """
    count = sdr_capacity(wx,b) * sdr_capacity((n-wx),(w-b))
    return count


def sdr_false_positive(n,w,wx,theta):
    """ 
        Overlap set
        n - shape
        wx - subsample on bits
        w - total on bits
        b - overlap
    """
    count = np.sum([sdr_overlap_set(n,w,wx,b) for b in range(theta,wx+1)]) / sdr_capacity(n,w)
    return count

def sdr_false_positive_approximate(n,w,theta):
    return sdr_overlap_set(n,w,theta) / sdr_capacity(n,w)

def sdr_match(left,right,theta,score=True):
    """
        fuzzy matching via theta
        threshold of the overlap score
        lower theta to decrease sensitivity and increase robustness
    """
    overlap = sdr_overlap(left,right)
    if overlap >= theta:
        return True
    return False

def sdr_overlap(left,right,score=True):
    """
        Returns overlap score, logical and
    """
    if score:
        return np.intersect1d(left,right).shape[0]
    return np.intersect1d(left,right)

def sdr_overlap_many(sdrs,score=True):
    if score:
        return reduce(np.intersect1d,sdrs).shape[0]
    return reduce(np.intersect1d,sdrs)

def sdr_union(left,right,score=True):
    """
        Returns union score, logical or
    """
    if score:
        return np.union1d(left,right).shape[0]
    return np.union1d(left,right)

def sdr_union_many(sdrs,score=True):
    if score:
        return reduce(np.union1d,sdrs).shape[0]
    return reduce(np.union1d,sdrs)


def sdr_subsample(sdr,ratio):
    """
        n - full input bit array size
        sdr - on bit inicie array
        ratio - number of on bits to select
    """
    count = round(ratio * sdr.shape[0])
    return np.random.choice(sdr,count,replace=False)

def SDR_create(n,w):
    sdr = np.random.choice(a=n,size=w,replace=False)
    return sdr

def SDR_STACK(n,w,sdr=None):
    if not sdr:
        sdr = SDR_create(n,w)
    return sdr

def SDR_STACK_add(stack,sdr):
    return np.vstack((stack,sdr))

def SDR_STACK_add_n(stack,count,n,w):
    """
        adds count random sdrs
    """
    for x in range(count):
        stack = SDR_STACK_add(stack,SDR_create(n,w))
    return stack

def SDR_STACK_remove(stack,sdr_index):
    return np.delete(stack,sdr_index,axis=0)

def SDR_STACK_compare(stack,sdr_index):
    """
        stack: stack of sdr's
        sdr_index: index of stack to compare to

        returns - sdr_stack indecies sorted by comparison score.
    """
    stack_scores = np.apply_along_axis(lambda sdr: sdr_overlap(left=sdr,right=stack[sdr_index]),axis=1,arr=stack)
    return np.argsort(stack_scores,axis=0,kind='quicksort')[::-1],np.sort(stack_scores,axis=0,kind='quicksort')[::-1],stack[np.argsort(stack_scores,axis=0,kind='quicksort')[::-1]]

def SDR_STACK_false_positive(stack,noise,n):
    """
        upper bound of false positive of sdr stack
    """
    return np.sum(np.apply_along_axis(lambda sdr: sdr_false_positive(n,sdr.shape[0],sdr.shape[0],noise),axis=1,arr=stack))

def SDR_STACK_false_match(stack,noise,n,w,m):
    """
        upper bound of false positive of sdr stack
    """
    wx = round(n*(1.0-sdr_pzero(n,w,m)))
    return np.sum(np.apply_along_axis(lambda sdr: sdr_false_positive(n,w,wx,noise),axis=1,arr=stack))

def SDR_STACK_false_positive_probability(n,w,m):
    """
        upper bound of false positive of sdr stack
    """
    return (sdr_pzero(n,w,m))**w

def sdr_pzero(n,w,m):
    return (1.0 - (w/n))**m


def test():
    n = 64
    w = 1
    pattern_num = 64
    false_match = 0.015625
    print(sdr_capacity(n,w))
    print(sdr_false_positive(n,w,w,w))


    n = 64
    w = 3
    pattern_num = 41664
    false_match = 2.40015E-05
    print(sdr_capacity(n,w))
    print(sdr_false_positive(n,w,w,w))

    n = 64
    w = 8
    t = 6
    false_match = 0.000379303
    print(sdr_false_positive(n,w,w,t))

    n = 64
    w = 12
    M = 10
    noise = 8
    false_match = 0.00042311185036318427
    stack = SDR_STACK(n=n,w=w)
    stack = SDR_STACK_add_n(stack,M-1,n,w)
    sfp = SDR_STACK_false_positive(stack,noise,n)
    print(sfp)

    n = 64
    w = 4
    M = 10
    noise = 4
    stack = SDR_STACK(n=n,w=w)
    stack = SDR_STACK_add_n(stack,M-1,n,w)
    sfpp = SDR_STACK_false_match(stack,noise,n,w,M)
    print(sfpp)



def run():
    n = 256
    sparsity = 0.5

    w = 40

    x = SDR_create(n=n,w=w)
    y = SDR_create(n=n,w=w)

    # print(x)
    # print(y)

    # print(sdr_overlap(x,y))
    overlap = sdr_overlap(x,y)

    b_ratio = 0.5
    b = round(b_ratio * x.shape[0])

    # print(n,overlap,b)
    # print(sdr_overlap_set(n,overlap,b))

    theta_ratio = 0.5
    theta = round(theta_ratio * n)

    # os = sdr_overlap_set(n=1024,w=8,wx=4,b=2)
    fp = sdr_false_positive(n=1024,w=20,wx=10,theta=5)
    print(fp)

    stack = SDR_STACK(n=n,w=w)

    stack = SDR_STACK_add_n(stack,103,n,w)

    # print(stack)

    # stack = SDR_STACK_remove(stack,0)
    # print(stack)

    comp,scores,sortd = SDR_STACK_compare(stack,0)

    # print(comp)
    # print(scores)
    # print(sortd)

    un = sdr_union_many(stack,score=False)
    uncount = sdr_union_many(stack)
    # print(un,un/n)
    print(un)
    print(uncount)

    theta_ratio = 0
    # theta = round(theta_ratio * un)
    print(n,uncount,w,0)
    fp = sdr_false_positive(n=n,w=uncount,wx=w,theta=0)
    print(fp)


if __name__ == '__main__':
    print('HTM')
    # run()
    test()