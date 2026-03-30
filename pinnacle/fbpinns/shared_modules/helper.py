"""FBPINNs helper utilities (vendored).

This file is used by multiple FBPINNs modules. Keep API stable.
"""

from __future__ import annotations

import copy as python_copy
import functools
import time
from typing import Any, Callable, List

import torch


def recursive_list_map(func: Callable[[Any], Any], items: List[Any]) -> List[Any]:
    """Recursively map `func` over nested lists."""
    return [
        recursive_list_map(func, x) if isinstance(x, list) else func(x)
        for x in items
    ]


# Helper classes

class cache_x:
    """Cache decorator for functions taking a torch tensor `x` as input.

    Inputs are treated as equal if they are close under `torch.norm(x - x_) < eps`.
    """

    def __init__(self, eps: float = 1e-3, maxsize: int = 100):
        self.eps = eps
        self.maxsize = maxsize
        self.cache_dict: dict[torch.Tensor, Any] = {}

    def isclose(self, x: torch.Tensor, x_: torch.Tensor) -> bool:
        return x.shape == x_.shape and torch.norm(x - x_) < self.eps

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped_func(_, x):
            v = None
            for x_, v_ in self.cache_dict.items():
                if self.isclose(x, x_):
                    v = v_
                    break
            if v is None:
                v = func(_, x)
                self.cache_dict[x] = v
            if len(self.cache_dict) > self.maxsize:
                self.cache_dict.clear()
            return v

        return wrapped_func

class DictToObj:
    "Convert a dictionary into a python object"
    def __init__(self, copy=True, **kwargs):
        "Input dictionary by values DictToObj(**dict)"
        assert type(copy)==bool
        for key in kwargs.keys():
            if copy:
                item = python_copy.deepcopy(kwargs[key])
                key = python_copy.deepcopy(key)
            else:
                item = kwargs[key]
            self[key] = item
            
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, item):
        setattr(self, key, item)
        
    def __str__(self):
        s = repr(self) + '\n'
        for k in vars(self): s+="%s: %s\n"%(k,self[k])
        return s
        
class Timer:    
    "Simple timer context manager"
    
    def __init__(self, name=None, verbose=True):
        self.name = name
        self.verbose=verbose
        
    def __enter__(self):
        self.start = time.time()
        return self# so we can access this using "with Timer as timer"
    
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        tag = " (%s)"%(self.name) if self.name is not None else ""
        if self.verbose: print("Time elapsed"+tag+": %.4f s"%(self.interval))
    
    
if __name__ == "__main__":
    
    d = {"a":[1,2,3], "b":2}
    
    a = DictToObj(**d)
    b = DictToObj(copy=False, **d)
    b.fun = "fun"
    b["yo"] = "yo"
    
    print(a,b)
    d["a"][0]=10
    print(a,b)
    
    l = [1, 2, 3, [1, 2, 3], [5, 6, [7, 8]]]
    a = recursive_list_map(lambda x: x + 1, l)
    print(a)
    
    with Timer(verbose=True) as timer:
        time.sleep(1)
    print(timer.interval)
    
    with Timer("test") as timer:
        time.sleep(1)
    print(timer.interval)
