from datetime import datetime
import logging
import math
import os
from typing import List
import numpy as np
import torch
from torch import Tensor
TIME = datetime.now().strftime('%Y%m%d%H%M')


def cal_memory(param, set_layout=None):
    '''
    calculate the memory size of a tensor
    param: tensor
    set_layout: can be 'torch.sparse_csr', 'torch.sparse_coo', 'torch.strided'
    '''
    if isinstance(param, int):
        param = Tensor([param])
    if isinstance(param, np.ndarray):
        return param.nbytes
    assert isinstance(param, torch.Tensor), 'param must be a tensor but a ' + (f'dict {param.keys()}' if isinstance(param, dict) else f'{type(param)}')
    layout = str(param.layout)

    if set_layout == 'bit':
        return math.ceil(param.numel() / 8)

    if set_layout is not None and layout != set_layout:
        print('change layout from', layout, 'to', set_layout)
        if set_layout == 'torch.sparse_csr':
            param = param.to_sparse_csr()
        elif set_layout == 'torch.sparse_coo':
            param = param.to_sparse_coo()
        elif set_layout == 'torch.strided':
            param = param.to_dense()
        else:
            raise ValueError('Unsupported layout', set_layout, 'for tensor layout', layout)

    layout = str(param.layout)
    if layout == 'torch.sparse_csr':
        row = param.crow_indices().numel() * param.crow_indices().element_size()
        col = param.col_indices().numel() * param.col_indices().element_size()
        data = param.values().numel() * param.values().element_size()
        return row + col + data
    elif layout == 'torch.sparse_coo':
        indices = param.indices().numel() * param.indices().element_size()
        data = param.values().numel() * param.values().element_size()
        return indices + data
    elif layout == 'torch.strided':
        return param.numel() * param.element_size()
    else:
        raise ValueError('Unsupported layout', layout, set_layout)
        

def calculate_data_size(param, set_sparse = None, set_layout='torch.sparse_csr'):
    '''
    set_layout: can be 'torch.sparse_csr', 'torch.sparse_coo', 'torch.strided'
    '''
    total = 0
    if set_sparse == 'all':
        sparse_param = param
        dense_param = {}
    elif set_sparse is not None:
        sparse_filter = LayerFilter(any_select_keys=set_sparse)
        sparse_param = sparse_filter(param)
        # print('sparse_param', sparse_param.keys())
        dense_filter = LayerFilter(unselect_keys=list(sparse_param.keys()))
        dense_param = dense_filter(param)
    else:
        sparse_param = {}
        dense_param = param

    for k, v in dense_param.items():
        if isinstance(v, tuple):
            for i in v:
                total += cal_memory(i)
        elif isinstance(v, dict) and 'param' in v and 'new_diff' in v:
            v = v['param']
            total += cal_memory(v)
        else:
            total += cal_memory(v)

    for k, v in sparse_param.items():
        layout = set_layout
        if isinstance(v, tuple):
            for i in v:
                total += cal_memory(i, set_layout=layout)
        elif isinstance(v, dict) and 'param' in v and 'new_diff' in v:
            if not v['new_diff']:
                layout = None
            v = v['param']
            total += cal_memory(v, set_layout=layout)
        else:
            total += cal_memory(v, set_layout=layout)

    return total

class LayerFilter:

    def __init__(self,
                 unselect_keys: List[str] = None,
                 all_select_keys: List[str] = None,
                 any_select_keys: List[str] = None) -> None:
        self.update_filter(unselect_keys, all_select_keys, any_select_keys)

    def update_filter(self,
                      unselect_keys: List[str] = None,
                      all_select_keys: List[str] = None,
                      any_select_keys: List[str] = None):
        self.unselect_keys = unselect_keys if unselect_keys is not None else []
        self.all_select_keys = all_select_keys if all_select_keys is not None else []
        self.any_select_keys = any_select_keys if any_select_keys is not None else []

    def __call__(self, param_dict, param_dict_template=None):
        if param_dict_template is not None:
            return {
                layer_key: param for layer_key, param in param_dict.items()
                if layer_key in param_dict_template
            }
        
        elif len(self.unselect_keys + self.all_select_keys +
               self.any_select_keys) == 0:
            return param_dict
        
        else:
            d = {}
            for layer_key, param in param_dict.items():
                if isinstance(layer_key, str):
                    if (len(self.unselect_keys) == 0 or all(key not in layer_key for key in self.unselect_keys)) and (
                        len(self.all_select_keys) == 0 or all(key in layer_key for key in self.all_select_keys)) and (
                        len(self.any_select_keys) == 0 or any(key in layer_key for key in self.any_select_keys)):
                        d[layer_key] = param
                elif isinstance(layer_key, int):
                    if (len(self.unselect_keys) == 0 or layer_key not in self.unselect_keys) and (
                        len(self.all_select_keys) == 0 or layer_key in (self.all_select_keys + self.any_select_keys)):
                        d[layer_key] = param
            return d
        
    def __str__(self) -> str:    
        return f"unselect_keys:{self.unselect_keys}  all_select_keys:{self.all_select_keys}  any_select_keys:{self.any_select_keys}"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, LayerFilter):
            return False
        return self.unselect_keys == value.unselect_keys and self.all_select_keys == value.all_select_keys and self.any_select_keys == value.any_select_keys

    def __hash__(self) -> int:
        return hash(str(self))

def save_model_param(model_params,
                     round_idx,
                     path_tag,
                     pre_desc=None,
                     post_desc=None,
                     is_grad=True,
                     path=None):
    # Save global weights as an experiment
    if pre_desc is None:
        pre_desc = path_tag

    if path is None:
        path = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(path,
                            'grad_lists' if is_grad else 'weight_lists',
                            path_tag)
    os.makedirs(save_dir, exist_ok=True)
    if post_desc is None:
        path = os.path.join(save_dir, f'{pre_desc}_round_{round_idx}.pt')
    else:
        path = os.path.join(save_dir,
                            f'{pre_desc}_round_{round_idx}_{post_desc}.pt')

    logging.info(f"Save {path_tag} {round_idx} model params to '{path}'.")
    torch.save(model_params, path)
    return path

# -----------------------Cosine similarity calculation--------------------
def cos_similar(x1: Tensor, x2: Tensor):
    x1 = x1.flatten()
    x2 = x2.flatten()
    # Two norms will cause precision problems
    # return torch.sum(x1 * x2) / (torch.norm(x1) * torch.norm(x2))
    w1 = torch.sum(x1 * x1)
    w2 = torch.sum(x2 * x2)
    if w1 == 0 or w2 == 0:
        return torch.Tensor([1.0])
    return torch.sum(x1 * x2) / torch.sqrt(w1 * w2)
