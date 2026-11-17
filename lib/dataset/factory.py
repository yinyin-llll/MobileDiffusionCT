# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Original framework created by Kai Ma (makai0324@gmail.com).
def get_dataset(name):
    if name != 'align_ct_drr_multiview_std':
        raise ValueError('Use align_ct_drr_multiview_std for the supplied HDF5 configs')
    from .alignDataSetDRRViews import AlignDataSet
    from .collate_fn import collate_gan_views
    return AlignDataSet, None, None, collate_gan_views
