# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch

def collate_gan(batch):
  '''
  :param batch: [imgs, boxes, labels] dtype = np.ndarray
  imgs:
    shape = (C H W)
  :return:
  '''
  #使用 PyTorch 的 torch.stack 函数将 CT 影像数据堆叠在一起，返回一个 PyTorch 张量。
  ct = [x[0] for x in batch]
  #这部分代码将 X 光影像数据堆叠在一起，返回一个 PyTorch 张量。
  xray = [x[1] for x in batch]
  
  file_path = [x[2] for x in batch]

  return torch.stack(ct), torch.stack(xray), file_path

def collate_gan_views(batch):
  '''
  Support both legacy two-view tuple and variable-view tensor [V,1,H,W].
  '''
  ct = [x[0] for x in batch]
  xr = [x[1] for x in batch]
  file_path = [x[2] for x in batch]

  # legacy: tuple/list of two xrays
  if isinstance(xr[0], (tuple, list)):
    xray1 = [x[0] for x in xr]
    xray2 = [x[1] for x in xr]
    return torch.stack(ct), [torch.stack(xray1), torch.stack(xray2)], file_path

  # new: tensor [V,1,H,W] per sample
  if torch.is_tensor(xr[0]):
    result=(torch.stack(ct), torch.stack(xr), file_path)
    if len(batch[0])==4:return (*result,torch.stack([x[3] for x in batch]))
    return result

  raise TypeError('Unsupported xray data type in collate_gan_views: {}'.format(type(xr[0])))
