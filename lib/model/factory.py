# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Original framework created by Kai Ma (makai0324@gmail.com).
from torch.optim import lr_scheduler

def get_model(name):
    if name != 'MultiViewCTDiffusion':
        raise ValueError('This export supports MultiViewCTDiffusion only')
    from .multiView_CTDiffusion import CTDiffusion
    return CTDiffusion

def get_scheduler(optimizer, opt):
  if opt.lr_policy == 'lambda':
    def lambda_rule(epoch):
      lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
      return lr_l

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
  elif opt.lr_policy == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
  elif opt.lr_policy == 'plateau':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
  elif opt.lr_policy == 'invariant':
    def lambda_rule(epoch):
      lr_l = 1.0
      return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
  else:
    return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
  return scheduler
