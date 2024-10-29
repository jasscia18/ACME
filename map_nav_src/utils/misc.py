import random
import numpy as np
import torch

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .distributed import init_distributed

from typing import Tuple, Union, Dict, Any

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).cuda()
    return mask
#
#
#
# def set_cuda(args) -> Tuple[bool, int, torch.device]:
#     """
#     Initialize CUDA for distributed computing
#     """
#     if not torch.cuda.is_available():
#         assert args.local_rank == -1, args.local_rank
#         return True, 0, torch.device("cpu")
#     default_gpu = True
#     device = torch.device("cuda")
#     n_gpu = torch.cuda.device_count()
#
#     # get device settings
#     #print('thisi  rrr:', args.local_rank)
#     # if args.local_rank != -1:
#     #     #print('thi s i==========+++++++++++++++============================')
#     #     init_distributed(args)
#     #     #print('thi s i======================================')
#     #     torch.cuda.set_device(args.local_rank)
#     #     device = torch.device("cuda", args.local_rank)
#     #     n_gpu = 1
#     #     default_gpu = dist.get_rank() == 0
#         # if default_gpu:
#         #     LOGGER.info(f"Found {dist.get_world_size()} GPUs")
#     # else:
#     #     default_gpu = True
#     #     device = torch.device("cuda")
#     #     n_gpu = torch.cuda.device_count()
#
#     return default_gpu, n_gpu, device
#
