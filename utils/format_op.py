import numpy as np
import os

def params2id(*args):
    nargs = len(args)
    id_ = '{}'+'_{}'*(nargs-1)
    return id_.format(*args)

def listformat(list_, replace=False, src='0.', dst='.'):
    if replace: return '/'.join([str(v) for v in list_]).replace(src, dst)
    else: return "/".join([str(v) for v in list_])

def listformat_form(list_, form):
    return "/".join([form%v for v in list_])
