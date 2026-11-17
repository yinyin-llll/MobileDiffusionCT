"""Verify the exported model without loading any patient data or checkpoint."""
import argparse
import copy
import json
from pathlib import Path
import tempfile

import torch
from lib.config.config import cfg,cfg_from_yaml,merge_dict_and_yaml
from lib.model.multiView_CTDiffusion import CTDiffusion


def check(config,checkpoint=None):
    torch.set_num_threads(2)
    cfg_from_yaml(str(config))
    with tempfile.TemporaryDirectory(prefix='mobilect_arch_') as temporary:
        opt=merge_dict_and_yaml(dict(data='LIDC256',tag='architecture_check',gpu_ids=[],dataset='test'),copy.deepcopy(cfg))
        opt.MODEL_SAVE_PATH=temporary
        model=CTDiffusion();model.init_network(opt);net=model._get_inner_net()
        spatial=[(name,module) for name,module in net.named_modules()
                 if isinstance(module,(torch.nn.Conv2d,torch.nn.Conv3d,torch.nn.ConvTranspose2d,torch.nn.ConvTranspose3d))]
        grouped=[name for name,module in spatial if module.groups!=1]
        parameters=sum(p.numel() for p in net.parameters())
        assert parameters==17536467,(parameters,'unexpected architecture')
        assert not grouped,grouped
        assert type(net.cond_encoder).__name__=='ResidualPyramidConditionEncoder'
        assert net.cond_encoder.mode=='adaptive_pyramid'
        assert type(net.denoiser).__name__=='LatentDenoiser3D'
        assert net.condition_refiner is None
        result=dict(status='passed',parameters=parameters,condition_encoder=type(net.cond_encoder).__name__,
                    lifting=net.cond_encoder.mode,denoiser=type(net.denoiser).__name__,
                    spatial_convolutions=len(spatial),grouped_convolutions=grouped,
                    architecture=net.architecture_spec)
        if checkpoint:
            saved=torch.load(checkpoint,map_location='cpu')
            net.load_state_dict(saved['state_dict'],strict=True)
            result['checkpoint_compatibility']=dict(status='passed',matched_keys=len(saved['state_dict']),strict=True)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/standard_aplat_2v.yml');p.add_argument('--out');p.add_argument('--checkpoint');a=p.parse_args()
    result=check(Path(a.config),a.checkpoint);print(json.dumps(result,indent=2))
    if a.out:
        output=Path(a.out);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
