# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import glob
import h5py
import numpy as np
import torch
import torch.nn.functional as F

from lib.dataset.baseDataSet import Base_DataSet
from lib.dataset.utils import get_dataset_from_txt_file


class AlignDataSet(Base_DataSet):
  def __init__(self, opt, center_num=None):
    super(AlignDataSet, self).__init__()
    self.opt = opt
    self.center_num = center_num
    self.dataset_paths = sorted(get_dataset_from_txt_file(self.opt.datasetfile))
    self.dataset_size = len(self.dataset_paths)
    self.dir_root = os.path.join(self.opt.dataroot)

    self.ct_size = int(getattr(opt, 'ct_channel', 128))
    self.xy_size = int(getattr(opt, 'fine_size', 128))
    req_views = getattr(opt, 'drr_num_views', None)
    self.req_views = int(req_views) if req_views else None

  @property
  def name(self):
    return 'AlignDataSetDRRViews'

  @property
  def num_samples(self):
    return self.dataset_size

  def get_case_h5(self, root, index_name):
    folder = os.path.join(root, index_name)
    assert os.path.isdir(folder), 'Folder does not exist: {}'.format(folder)
    h5s = sorted(glob.glob(os.path.join(folder, '*.h5')))
    assert len(h5s) > 0, 'No h5 found in {}'.format(folder)
    return h5s[0]

  def _resize_ct(self, ct):
    t = torch.from_numpy(ct).float().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    t = F.interpolate(t, size=(self.ct_size, self.xy_size, self.xy_size), mode='trilinear', align_corners=False)
    return t.squeeze(0).squeeze(0)

  def _resize_xrays(self, xr):
    # xr: [V,H,W]
    t = torch.from_numpy(xr).float().unsqueeze(1)  # [V,1,H,W]
    t = F.interpolate(t, size=(self.xy_size, self.xy_size), mode='bilinear', align_corners=False)
    return t

  def _normalize_ct(self, ct):
    ct = ct.clamp(float(self.opt.CT_MIN_MAX[0]), float(self.opt.CT_MIN_MAX[1]))
    ct = (ct - float(self.opt.CT_MIN_MAX[0])) / float(self.opt.CT_MIN_MAX[1] - self.opt.CT_MIN_MAX[0])
    ct = (ct - float(self.opt.CT_MEAN_STD[0])) / float(self.opt.CT_MEAN_STD[1])
    return ct

  def _normalize_xray(self, xr):
    # xr in [0,1]
    x1 = (xr - float(self.opt.XRAY1_MEAN_STD[0])) / float(self.opt.XRAY1_MEAN_STD[1])
    x2 = (xr - float(self.opt.XRAY2_MEAN_STD[0])) / float(self.opt.XRAY2_MEAN_STD[1])
    return x1, x2

  def _select_views(self, xr):
    # xr: [V,1,H,W]
    v = xr.shape[0]
    # If req_views is not set, use all available views.
    if self.req_views is None:
      return xr

    if v == self.req_views:
      return xr

    if v > self.req_views:
      idx = np.linspace(0, v - 1, self.req_views).round().astype(np.int64)
      return xr[idx]

    # Strict mode: never pad/repeat views. The generated dataset must contain
    # the exact number of views required by the experiment.
    raise ValueError(
      'Case has fewer views than requested: got {}, need {}. '
      'Please regenerate DRR with matching --num-views.'.format(v, self.req_views)
    )

  def pull_item(self, item):
    file_path = self.get_case_h5(self.dir_root, self.dataset_paths[item])
    with h5py.File(file_path, 'r') as f:
      ct = np.asarray(f['ct'], dtype=np.float32)
      geometry = None
      uploaded = getattr(self.opt, 'drr_uploaded_mode', False)
      if uploaded:
        n = f['xrays'].shape[0]
        requested = self.req_views or n
        if n % requested:
          raise ValueError('Uploaded view count must divide requested uniform grid')
        indices = np.arange(requested, dtype=np.int64) * (n // requested)
        angles = np.asarray(f['angles'], dtype=np.float32)
        span = float(self.opt.diffusion_backproject_angle_span)
        offset = float(self.opt.diffusion_backproject_angle_offset)
        expected_type = getattr(self.opt, 'drr_input_type', '')
        if expected_type and f.attrs.get('input_type', '') != expected_type:
          raise ValueError('Uploaded input_type does not match configuration')
        if getattr(self.opt, 'drr_uniform_angles', True):
          if len(angles) != n or not np.allclose(angles, offset + np.arange(n)*span/n, atol=1e-4):
            raise ValueError('Uploaded angles do not match configured angular span')
        elif requested != n or len(angles) != n or not np.isfinite(angles).all():
          raise ValueError('Slice/mixed inputs require all original views and valid metadata')
        xr = np.asarray(f['xrays'][indices], dtype=np.float32)
        if getattr(self.opt, 'drr_undo_display_flip', False) and f.attrs.get('drr_display_orientation') == 'flipud':
          xr = np.flip(xr, axis=1).copy()
      else:
        xr = np.asarray(f['xrays'], dtype=np.float32)
      geometry_mode=getattr(self.opt,'drr_geometry_mode','')
      if geometry_mode:
        if not uploaded:raise ValueError('Observation geometry requires uploaded input protocol')
        geometry=np.zeros((len(xr),3),dtype=np.float32)
        kind=f.attrs.get('input_type','')
        if kind=='middle_axial_ct_slice':
          if len(xr)!=1:raise ValueError('Expected one middle slice')
          geometry[:,0]=1;geometry[:,2]=2.*(ct.shape[0]//2)/max(ct.shape[0]-1,1)-1.
        elif kind in ('two_middle_axial_ct_slices','ap_lat_drr_plus_two_middle_axial_ct_slices'):
          positions=np.asarray(f['slice_indices'],dtype=np.float32).reshape(-1)
          if len(positions)!=2 or np.any(positions<0) or np.any(positions>=ct.shape[0]):raise ValueError('Invalid slice indices')
          start=2 if kind.startswith('ap_lat') else 0
          if len(xr)!=start+2:raise ValueError('Unexpected mixed observation count')
          geometry[:start,1]=[0.,90.][:start]
          geometry[start:,0]=1;geometry[start:,2]=2.*positions/max(ct.shape[0]-1,1)-1.
        elif geometry_mode=='nominal_parallel':
          if kind not in ('real_lidc_xray_single_ap','real_lidc_xray_2view'):raise ValueError('Nominal geometry is restricted to labeled X-ray groups')
          geometry[:,1]=angles[indices]
        else:raise ValueError('Unsupported geometry mode/input type')
      if getattr(self.opt, 'diffusion_require_geometry', False):
        span = float(self.opt.diffusion_backproject_angle_span)
        offset = float(self.opt.diffusion_backproject_angle_offset)
        expected = offset + np.arange(len(xr)) * span / len(xr)
        if 'angles' not in f or not np.allclose(f['angles'][:], expected, atol=1e-4):
          raise ValueError('Configured angles do not match HDF5: ' + file_path)
        if f.attrs.get('protocol', '') != 'parallel_mean_v2':
          raise ValueError('Expected revision v2 projection data: ' + file_path)
        if list(ct.shape) != [self.ct_size, self.xy_size, self.xy_size] or list(xr.shape[1:]) != [self.ct_size, self.xy_size]:
          raise ValueError('Revision geometry requires native model resolution')
        if not np.allclose([f.attrs['ct_min'], f.attrs['ct_max']], self.opt.CT_MIN_MAX):
          raise ValueError('CT normalization differs from DRR generation')
        if self.req_views is not None and self.req_views != len(xr):
          raise ValueError('Generate the requested views explicitly; no implicit subsampling')

    ct_t = self._resize_ct(ct)
    xr_t = self._resize_xrays(xr)
    xr_t = self._select_views(xr_t)

    ct_t = self._normalize_ct(ct_t)
    xr1, _ = self._normalize_xray(xr_t)

    if getattr(self.opt, 'model_class', '') in ('MultiViewCTGAN','X2CTOfficial','DPGANOfficial'):
      if len(xr1) != 2:
        raise ValueError('CTGAN requires exactly two DRR views')
      return ct_t, (xr1[0], xr1[1]), file_path
    if geometry is not None:return ct_t, xr1, file_path, torch.from_numpy(geometry)
    return ct_t, xr1, file_path
