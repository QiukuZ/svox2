import torch
import os
import shutil
import argparse
from tqdm import trange
from tqdm import tqdm

# Import Libs for svox2
# First, install svox2
# Then, python opt.py <path_to>/nerf_synthetic/<scene> -t ckpt/<some_name>
# or use launching script:   sh launch.sh <EXP_NAME> <GPU> <DATA_DIR>
import torch.cuda
import torch.optim
import torch.nn.functional as F
import svox2
import json
import imageio
from os import path
import gc
import numpy as np
import math
import cv2
from util.dataset import datasets
from util.util import Timing, get_expon_lr_func, generate_dirs_equirect, viridis_cmap
from util import config_util
# Import Libs for convonet
import pandas as pd
import sys
BASE_DIR = os.path.dirname(sys.path[0])
sys.path.append(os.path.join(BASE_DIR, "convonet"))
from src import config
from src.checkpoints import CheckpointIO

from torch.utils.tensorboard import SummaryWriter

from typing import NamedTuple, Optional, Union


def get_convonet_config(args):
    cfg_convonet = config.load_config(args.config_convonet, 'convonet/configs/default.yaml')
    return cfg_convonet

def get_svox2_config(parser):
    config_util.define_common_args(parser)
    group = parser.add_argument_group("general")
    group.add_argument('--train_dir', '-t', type=str, default='ckpt',
                        help='checkpoint and logging directory')

    group.add_argument('--reso',
                            type=str,
                            default=
                            "[[256, 256, 256], [512, 512, 512]]",
                        help='List of grid resolution (will be evaled as json);'
                                'resamples to the next one every upsamp_every iters, then ' +
                                'stays at the last one; ' +
                                'should be a list where each item is a list of 3 ints or an int')
    group.add_argument('--upsamp_every', type=int, default=
                        3 * 12800,
                        help='upsample the grid every x iters')
    group.add_argument('--init_iters', type=int, default=
                        0,
                        help='do not upsample for first x iters')
    group.add_argument('--upsample_density_add', type=float, default=
                        0.0,
                        help='add the remaining density by this amount when upsampling')
    group.add_argument('--crop_image_edges', type=int, default=
                        0,
                        help='for some dataset need crop image edges because of distortion')
    group.add_argument('--basis_type',
                        choices=['sh', '3d_texture', 'mlp'],
                        default='sh',
                        help='Basis function type')

    group.add_argument('--basis_reso', type=int, default=32,
                    help='basis grid resolution (only for learned texture)')
    group.add_argument('--sh_dim', type=int, default=9, help='SH/learned basis dimensions (at most 10)')

    group.add_argument('--mlp_posenc_size', type=int, default=4, help='Positional encoding size if using MLP basis; 0 to disable')
    group.add_argument('--mlp_width', type=int, default=32, help='MLP width if using MLP basis')

    group.add_argument('--background_nlayers', type=int, default=0,#32,
                    help='Number of background layers (0=disable BG model)')
    group.add_argument('--background_reso', type=int, default=512, help='Background resolution')
    group.add_argument('--init_by_occ', type=bool, default=False, help='init by occ')
    group.add_argument('--init_by_occ_sparse', type=bool, default=False, help='init by occ sparse')
    group.add_argument('--occ_prob_path', type=str, default="", help='occ prob grid path')
    group.add_argument('--gpu_id', type=int, default=0, help='GPU id')



    group = parser.add_argument_group("optimization")
    group.add_argument('--n_iters', type=int, default=10 * 12800, help='total number of iters to optimize for')
    group.add_argument('--batch_size', type=int, default=
                        5000,
                        #100000,
                        #  2000,
                    help='batch size')


    # TODO: make the lr higher near the end
    group.add_argument('--sigma_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
    group.add_argument('--lr_sigma', type=float, default=3e1, help='SGD/rmsprop lr for sigma')
    group.add_argument('--lr_sigma_final', type=float, default=5e-2)
    group.add_argument('--lr_sigma_decay_steps', type=int, default=250000)
    group.add_argument('--lr_sigma_delay_steps', type=int, default=15000,
                    help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_sigma_delay_mult', type=float, default=1e-2)#1e-4)#1e-4)


    group.add_argument('--sh_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="SH optimizer")
    group.add_argument('--lr_sh', type=float, default=
                        1e-2,
                    help='SGD/rmsprop lr for SH')
    group.add_argument('--lr_sh_final', type=float,
                        default=
                        5e-6
                        )
    group.add_argument('--lr_sh_decay_steps', type=int, default=250000)
    group.add_argument('--lr_sh_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_sh_delay_mult', type=float, default=1e-2)

    group.add_argument('--lr_fg_begin_step', type=int, default=0, help="Foreground begins training at given step number")

    # BG LRs
    group.add_argument('--bg_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Background optimizer")
    group.add_argument('--lr_sigma_bg', type=float, default=3e0,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_sigma_bg_final', type=float, default=3e-3,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_sigma_bg_decay_steps', type=int, default=250000)
    group.add_argument('--lr_sigma_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_sigma_bg_delay_mult', type=float, default=1e-2)

    group.add_argument('--lr_color_bg', type=float, default=1e-1,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_color_bg_final', type=float, default=5e-6,#1e-4,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_color_bg_decay_steps', type=int, default=250000)
    group.add_argument('--lr_color_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_color_bg_delay_mult', type=float, default=1e-2)
    # END BG LRs

    group.add_argument('--basis_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Learned basis optimizer")
    group.add_argument('--lr_basis', type=float, default=#2e6,
                        1e-6,
                    help='SGD/rmsprop lr for SH')
    group.add_argument('--lr_basis_final', type=float,
                        default=
                        1e-6
                        )
    group.add_argument('--lr_basis_decay_steps', type=int, default=250000)
    group.add_argument('--lr_basis_delay_steps', type=int, default=0,#15000,
                    help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_basis_begin_step', type=int, default=0)#4 * 12800)
    group.add_argument('--lr_basis_delay_mult', type=float, default=1e-2)

    group.add_argument('--rms_beta', type=float, default=0.95, help="RMSProp exponential averaging factor")

    group.add_argument('--print_every', type=int, default=20, help='print every')
    group.add_argument('--save_every', type=int, default=5,
                    help='save every x epochs')
    group.add_argument('--eval_every', type=int, default=1,
                    help='evaluate every x epochs')

    group.add_argument('--init_sigma', type=float,
                    default=0.1,
                    help='initialization sigma')
    group.add_argument('--init_sigma_bg', type=float,
                    default=0.1,
                    help='initialization sigma (for BG)')

    # Extra logging
    group.add_argument('--log_mse_image', action='store_true', default=False)
    group.add_argument('--log_depth_map', action='store_true', default=False)
    group.add_argument('--log_depth_map_use_thresh', type=float, default=None,
            help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term")


    group = parser.add_argument_group("misc experiments")
    group.add_argument('--thresh_type',
                        choices=["weight", "sigma"],
                        default="weight",
                    help='Upsample threshold type')
    group.add_argument('--weight_thresh', type=float,
                        default=0.0005 * 512,
                        #  default=0.025 * 512,
                    help='Upsample weight threshold; will be divided by resulting z-resolution')
    group.add_argument('--density_thresh', type=float,
                        default=5.0,
                    help='Upsample sigma threshold')
    group.add_argument('--background_density_thresh', type=float,
                        default=1.0+1e-9,
                    help='Background sigma threshold for sparsification')
    group.add_argument('--max_grid_elements', type=int,
                        default=44_000_000,
                    help='Max items to store after upsampling '
                            '(the number here is given for 22GB memory)')

    group.add_argument('--tune_mode', action='store_true', default=False,
                    help='hypertuning mode (do not save, for speed)')
    group.add_argument('--tune_nosave', action='store_true', default=False,
                    help='do not save any checkpoint even at the end')



    group = parser.add_argument_group("losses")
    # Foreground TV
    group.add_argument('--lambda_tv', type=float, default=1e-5)
    group.add_argument('--tv_sparsity', type=float, default=0.01)
    group.add_argument('--tv_logalpha', action='store_true', default=False,
                    help='Use log(1-exp(-delta * sigma)) as in neural volumes')

    group.add_argument('--lambda_tv_sh', type=float, default=1e-3)
    group.add_argument('--tv_sh_sparsity', type=float, default=0.01)

    group.add_argument('--lambda_tv_lumisphere', type=float, default=0.0)#1e-2)#1e-3)
    group.add_argument('--tv_lumisphere_sparsity', type=float, default=0.01)
    group.add_argument('--tv_lumisphere_dir_factor', type=float, default=0.0)

    group.add_argument('--tv_decay', type=float, default=1.0)

    group.add_argument('--lambda_l2_sh', type=float, default=0.0)#1e-4)
    group.add_argument('--tv_early_only', type=int, default=1, help="Turn off TV regularization after the first split/prune")

    group.add_argument('--tv_contiguous', type=int, default=1,
                            help="Apply TV only on contiguous link chunks, which is faster")
    # End Foreground TV

    group.add_argument('--lambda_sparsity', type=float, default=
                        0.0,
                        help="Weight for sparsity loss as in SNeRG/PlenOctrees " +
                            "(but applied on the ray)")
    group.add_argument('--lambda_beta', type=float, default=
                        0.0,
                        help="Weight for beta distribution sparsity loss as in neural volumes")


    # Background TV
    group.add_argument('--lambda_tv_background_sigma', type=float, default=1e-2)
    group.add_argument('--lambda_tv_background_color', type=float, default=1e-2)

    group.add_argument('--tv_background_sparsity', type=float, default=0.01)
    # End Background TV

    # Basis TV
    group.add_argument('--lambda_tv_basis', type=float, default=0.0,
                    help='Learned basis total variation loss')
    # End Basis TV

    group.add_argument('--weight_decay_sigma', type=float, default=1.0)
    group.add_argument('--weight_decay_sh', type=float, default=1.0)

    group.add_argument('--lr_decay', action='store_true', default=True)

    group.add_argument('--n_train', type=int, default=None, help='Number of training images. Defaults to use all avaiable.')

    group.add_argument('--nosphereinit', action='store_true', default=False,
                        help='do not start with sphere bounds (please do not use for 360)')

    args = parser.parse_args()
    config_util.maybe_merge_config_file(args)
    gpu_id = args.gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(gpu_id)
    print("Use GPU ID = ", gpu_id)

    assert args.lr_sigma_final <= args.lr_sigma, "lr_sigma must be >= lr_sigma_final"
    assert args.lr_sh_final <= args.lr_sh, "lr_sh must be >= lr_sh_final"
    assert args.lr_basis_final <= args.lr_basis, "lr_basis must be >= lr_basis_final"

    os.makedirs(args.train_dir, exist_ok=True)

    with open(path.join(args.train_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        # Changed name to prevent errors
        shutil.copyfile(__file__, path.join(args.train_dir, 'opt_frozen.py'))
    
    return args, gpu_id, device 

class Pipeline():
    """
    Pipeline
    """

    def __init__(self, args, is_cuda=False):
        self.args = args
        self.device = torch.device("cuda" if is_cuda else "cpu")
        self.convonet = {}
        self.svox2 = {}
        self.summary_writer = SummaryWriter(args.train_dir)
        self.is_cropped = False
        self.ckpt_path = path.join(args.train_dir, 'ckpt.npz')
        bbox_path = os.path.join(args.data_dir, "bbox.txt")
        if os.path.exists(bbox_path):
            self.bbox = np.loadtxt(os.path.join(args.data_dir, "bbox.txt"))
        self.grid_bbox = None
        self.cam_links_scale = 8

    def init_convonet(self, cfg_convonet):
        out_dir = cfg_convonet['training']['out_dir']
        
        # Dataset
        self.convonet['dataset'] = config.get_dataset('test', cfg_convonet, return_idx=True)
        
        # Model
        self.convonet['model'] = config.get_model(cfg_convonet, device=self.device, dataset=self.convonet['dataset'])
        self.convonet['checkpoint_io'] = CheckpointIO(out_dir, model=self.convonet['model'])
        self.convonet['checkpoint_io'].load(cfg_convonet['test']['model_file'])

        # Generator
        info = {}
        info["case"] = "crop_by_z"
        info["bbox"] = self.bbox
        self.convonet['generator'] = config.get_generator_crop(self.convonet['model'], cfg_convonet, info=info, device=self.device)

        # Generate
        self.convonet['model'].eval()

    def get_convonet_data(self, dataset=None):
        if dataset is None:
            dataset = self.convonet['dataset'][0]
        
        data = {}
        data['inputs'] = torch.tensor(dataset['inputs'],device=self.device).unsqueeze(0)
        return data 

    def get_grid_bbox(self):
        # Use points_bbox get gird_bbox
        n_crop, n_crop_axis = self.convonet['generator'].generate_vols_sliding_by_bbox(self.bbox)
        self.convonet["n_crop_axis"] = n_crop_axis
        self.convonet["n_crop"] = n_crop
        bb_box_min = self.convonet['generator'].vol_bound["query_vol"][0][0]
        bb_box_max = self.convonet['generator'].vol_bound["query_vol"][-1][-1]
        bbox = np.hstack([bb_box_min, bb_box_max])
        print("BBox size = ", bb_box_max - bb_box_min)
        np.savetxt(os.path.join(self.args.train_dir, "bbox.txt"), bbox)
        print("BBox saved in : ", os.path.join(self.args.train_dir, "bbox.txt"))
        self.is_cropped = True
        return bbox, n_crop_axis

    def get_convonet_occ(self, input=None, crop_reso=128, occ_by_net=True):
        occ_threshold = 0.01
        threshold = np.log(occ_threshold) - np.log(1. - occ_threshold)
        if input == None:
            data = self.get_convonet_data(input)
            inputs = data.get('inputs', torch.empty(1, 0)).to(self.device)
        else:
            # inputs = input # 1 * N * 3
            inputs = input[:,::3,:]

        if self.is_cropped:
            n_crop = self.convonet["n_crop"]
            n_crop_axis = self.convonet["n_crop_axis"]
            bb_box_min = self.convonet['generator'].vol_bound["query_vol"][0][0]
            bb_box_max = self.convonet['generator'].vol_bound["query_vol"][-1][-1]
            bbox = np.hstack([bb_box_min, bb_box_max])
        else:
            n_crop, n_crop_axis = self.convonet['generator'].generate_vols_sliding(inputs)
            self.convonet["n_crop_axis"] = n_crop_axis
            self.convonet["n_crop"] = n_crop
            bb_box_min = self.convonet['generator'].vol_bound["query_vol"][0][0]
            bb_box_max = self.convonet['generator'].vol_bound["query_vol"][-1][-1]
            bbox = np.hstack([bb_box_min, bb_box_max])
            print("BBox size = ", bb_box_max - bb_box_min)
            np.savetxt(os.path.join(self.args.train_dir, "bbox.txt"), bbox)
            print("BBox saved in : ", os.path.join(self.args.train_dir, "bbox.txt"))
            self.is_cropped = True

        all_grid_resolution = n_crop_axis * crop_reso
        all_grid_links = torch.ones(all_grid_resolution.tolist()).int().to(self.device) * -1
        n_crop_div = np.array([n_crop_axis[1]*n_crop_axis[2], n_crop_axis[2], 1])

        occ_cur_cap = 0
        if occ_by_net:
            print("Init occ by convonet ...")
            torch.cuda.empty_cache()
            for i in trange(n_crop):
                occ_value = self.convonet['generator'].generate_n_idx_occ(inputs, i)
                occ_mask = occ_value > threshold
                occ_cap = occ_mask.sum().item()

                # index to xiyizi
                i_xyz = np.zeros(3)
                i_tmp = i
                for ii in range(3):
                    i_xyz[ii] = i_tmp // n_crop_div[ii]
                    i_tmp = i_tmp % n_crop_div[ii]

                idx_start = i_xyz.astype(np.int32())  * crop_reso
                idx_end = idx_start + crop_reso
                all_grid_links[idx_start[0]:idx_end[0], idx_start[1]:idx_end[1], idx_start[2]:idx_end[2]][occ_mask] = torch.arange(occ_cap).int().to(self.device) + occ_cur_cap
                occ_cur_cap += occ_cap
        else:
            print("Init occ by points ...")
            scale_inputs = inputs[0] - torch.tensor(bb_box_min).to(self.device)
            scale_inputs /= (bb_box_max - bb_box_min).max()
            grid_inputs = (scale_inputs * all_grid_resolution.max()).long()

            all_grid_links[grid_inputs[:,0], grid_inputs[:,1], grid_inputs[:,2]] = 0
            occ_cap = (all_grid_links >= 0 ).sum()
            all_grid_links[all_grid_links>=0] = torch.arange(occ_cap).int().to(self.device) 

        # Get grid cap
        occ_cap = (all_grid_links >= 0).sum().item()
        print("All grid capacity = ", occ_cap)
        torch.cuda.empty_cache()
        return all_grid_links, bbox

    def svox2_load_dataset(self, bbox, split="all"):
        # Plenoxel DataSet Loading
        if split == "all" or split == "train":
            print("Loading train dataset ...")
            self.svox2['dset'] = datasets[self.args.dataset_type](self.args.data_dir,
                                                    split="train",
                                                    device=self.device,
                                                    crop_image_edges=self.args.crop_image_edges,
                                                    factor=1,
                                                    bbox=bbox,
                                                    n_images=self.args.n_train,
                                                    **config_util.build_data_options(self.args))
        self.n_image = self.svox2['dset'].n_images
        if split == "all" or split == "test":
            print("Loading test dataset ...")
            self.svox2['dset_test'] = datasets[self.args.dataset_type](self.args.data_dir, 
                                                        bbox=bbox, 
                                                        split="test", 
                                                        **config_util.build_data_options(self.args))

    def svox2_init_grid(self, reso, grid_links=None):
        init_by_links = False if grid_links is None else True
        grid = svox2.SparseGrid(reso=list(reso*self.convonet["n_crop_axis"]),
                                center=self.svox2['dset'].scene_center,
                                radius=self.convonet["n_crop_axis"] / self.convonet["n_crop_axis"].max(),
                                # use_sphere_bound=dset.use_sphere_bound and not args.nosphereinit,
                                use_sphere_bound=False,
                                basis_dim=self.args.sh_dim,
                                use_z_order=False,
                                device=self.device,
                                basis_reso=self.args.basis_reso,
                                basis_type=svox2.__dict__['BASIS_TYPE_' + self.args.basis_type.upper()],
                                mlp_posenc_size=self.args.mlp_posenc_size,
                                mlp_width=self.args.mlp_width,
                                background_nlayers=0,
                                background_reso=self.args.background_reso,
                                init_by_links=init_by_links,
                                init_occ_links=grid_links)
        grid.sh_data.data[:] = 0.0
        grid.density_data.data[:] = 100.0

        if grid.use_background:
            grid.background_data.data[..., -1] = self.args.init_sigma_bg
        self.svox2["optim_basis_mlp"] = None

        if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
            grid.reinit_learned_bases(init_type='sh')
            #  grid.reinit_learned_bases(init_type='fourier')
            #  grid.reinit_learned_bases(init_type='sg', upper_hemi=True)
            #  grid.basis_data.data.normal_(mean=0.28209479177387814, std=0.001)

        elif grid.basis_type == svox2.BASIS_TYPE_MLP:
            # MLP!
            self.svox2["optim_basis_mlp"] = torch.optim.Adam(
                            grid.basis_mlp.parameters(),
                            lr=self.args.lr_basis
                        )
        grid.requires_grad_(True)
        config_util.setup_render_opts(grid.opt, self.args)
        print('Render options', grid.opt)
        self.svox2["grid"] = grid
        self.svox2_reso = reso*self.convonet["n_crop_axis"]
        self.svox2_cam_links = torch.zeros((self.svox2_reso / self.cam_links_scale).astype(np.int).tolist()+[self.n_image]) > 0


    def svox2_update_grid(self, grid_links):
        assert self.svox2["grid"].links.shape == grid_links.shape, "grid shape error !"
        cur_cap = self.svox2["grid"].capacity.item()
        tmp_grid_mask = grid_links >=0
        tmp_grid_cap = tmp_grid_mask.sum().item()
        # push gird to all
        old_grid_mask = self.svox2["grid"].links >= 0 
        old_grid_cap = old_grid_mask.sum().item()
        if old_grid_cap > 0:
            self.svox2["grid"].density_data_all[self.svox2["grid"].links_all[old_grid_mask].long()] = self.svox2["grid"].density_data.detach().clone()
            self.svox2["grid"].sh_data_all[self.svox2["grid"].links_all[old_grid_mask].long()] = self.svox2["grid"].sh_data.detach().clone()

        # get new grid links
        new_grid_mask = torch.logical_xor(tmp_grid_mask >=0, self.svox2["grid"].links_all >=0)
        new_grid_mask = torch.logical_and(new_grid_mask, tmp_grid_mask)
        new_grid_cap = new_grid_mask.sum()
        self.svox2["grid"].links_all[new_grid_mask] = torch.arange(new_grid_cap).int().to(self.device) + cur_cap
        self.svox2["grid"].density_data_all = torch.vstack([self.svox2["grid"].density_data_all, 10. * torch.ones([new_grid_cap,1]).to(self.device)])
        self.svox2["grid"].sh_data_all = torch.vstack([self.svox2["grid"].sh_data_all, torch.zeros([new_grid_cap, self.svox2["grid"].sh_data.shape[1]]).to(self.device)])
        self.svox2["grid"].capacity = (self.svox2["grid"].links_all >=0 ).sum()
        print("Add grid cap = ", new_grid_cap.item(), "/ Cur grid cap = ", self.svox2["grid"].capacity.item())
        # get cur grid links -> make sure idx current
        self.svox2["grid"].links = torch.ones_like(grid_links) * -1
        # self.svox2["grid"].links = self.svox2["grid"].links_all
        update_links_by_cam_links = True
        if update_links_by_cam_links:
            tmp_cam_links = (torch.stack(torch.where(grid_links >= 0)).permute(1,0) / self.cam_links_scale).long().unique(dim=0)
            cam_links_mask = torch.zeros(self.svox2_cam_links.shape[:3]).to(self.device)
            cam_links_mask[tmp_cam_links[:,0], tmp_cam_links[:,1], tmp_cam_links[:,2]] = 1.0
            links_mask = F.interpolate(cam_links_mask[None,None,:,:,:], scale_factor=(8, 8, 8))[0,0] > 0.5
            tmp_grid_mask = (self.svox2["grid"].links_all >= 0) & links_mask
            print(f"Acctivate rate = {tmp_grid_mask.sum().item()}/{tmp_grid_cap}")
            tmp_grid_cap = tmp_grid_mask.sum().item()
        self.svox2["grid"].links[tmp_grid_mask] = torch.arange(tmp_grid_cap).int().to(self.device)
        self.svox2["grid"].density_data = torch.nn.Parameter(self.svox2["grid"].density_data_all[self.svox2["grid"].links_all[tmp_grid_mask].long()])
        self.svox2["grid"].sh_data = torch.nn.Parameter(self.svox2["grid"].sh_data_all[self.svox2["grid"].links_all[tmp_grid_mask].long()])
        print("Svox2 grid update !")

    def svox2_save(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = self.ckpt_path
        print('Saving grid ckpt to : ', ckpt_path)
        self.svox2["grid"].save(ckpt_path)

    def svox2_init(self, bbox, data_split="all"):
        self.svox2_load_dataset(bbox, data_split)

        self.svox2["resample_cameras"] = [
                            svox2.Camera(c2w.to(device=self.device),
                                        self.svox2['dset'].intrins.get('fx', i),
                                        self.svox2['dset'].intrins.get('fy', i),
                                        self.svox2['dset'].intrins.get('cx', i),
                                        self.svox2['dset'].intrins.get('cy', i),
                                        width=self.svox2['dset'].get_image_size(i)[1],
                                        height=self.svox2['dset'].get_image_size(i)[0],
                                        ndc_coeffs=self.svox2['dset'].ndc_coeffs) for i, c2w in enumerate(self.svox2['dset'].c2w)
                            ]
                    
        self.svox2["lr_sigma_func"] = get_expon_lr_func(self.args.lr_sigma, self.args.lr_sigma_final, self.args.lr_sigma_delay_steps,
                                  self.args.lr_sigma_delay_mult, self.args.lr_sigma_decay_steps)
        self.svox2["lr_sh_func"] = get_expon_lr_func(self.args.lr_sh, self.args.lr_sh_final, self.args.lr_sh_delay_steps,
                                    self.args.lr_sh_delay_mult, self.args.lr_sh_decay_steps)
        self.svox2["lr_basis_func"] = get_expon_lr_func(self.args.lr_basis, self.args.lr_basis_final, self.args.lr_basis_delay_steps,
                                    self.args.lr_basis_delay_mult, self.args.lr_basis_decay_steps)
        self.svox2["lr_sigma_bg_func"] = get_expon_lr_func(self.args.lr_sigma_bg, self.args.lr_sigma_bg_final, self.args.lr_sigma_bg_delay_steps,
                                    self.args.lr_sigma_bg_delay_mult, self.args.lr_sigma_bg_decay_steps)
        self.svox2["lr_color_bg_func"] = get_expon_lr_func(self.args.lr_color_bg, self.args.lr_color_bg_final, self.args.lr_color_bg_delay_steps,
                                    self.args.lr_color_bg_delay_mult, self.args.lr_color_bg_decay_steps)
        self.svox2["lr_sigma_factor"] = 0.2
        self.svox2["lr_sh_factor"] = 1.0
        self.svox2["lr_basis_factor"] = 1.0

        # epcoh info init
        self.svox2["epoch_id"] = 0
        self.svox2["gstep_id_base"] = 0
        self.grid_bbox = bbox
        print("Svox2 Init !")


    def svox2_train(self, epoch_num=1, cam_list=None):

        bb_box_min = self.convonet['generator'].vol_bound["query_vol"][0][0]
        bb_box_max = self.convonet['generator'].vol_bound["query_vol"][-1][-1]
        bbox = np.hstack([bb_box_min, bb_box_max])

        self.svox2["dset"].shuffle_rays_links(self.svox2['grid'].links, self.svox2_cam_links, bbox, cam_list, self.cam_links_scale)
        # if cam_list is None:
        #     self.svox2["dset"].shuffle_rays()
        # else:
        #     self.svox2["dset"].shuffle_rays_crop(cam_list)
        epoch_size = self.svox2["dset"].rays.origins.size(0)
        batches_per_epoch = (epoch_size-1)//self.args.batch_size+1

        print(f"Trian Step(Batch={self.args.batch_size}/Iter={batches_per_epoch})")
        
        for i in range(epoch_num):
            # Train
            pbar = tqdm(enumerate(range(0, epoch_size, self.args.batch_size)), total=batches_per_epoch)
            stats = {"mse" : 0.0, "psnr" : 0.0, "invsqr_mse" : 0.0}
            for iter_id, batch_begin in pbar:
                gstep_id = iter_id + self.svox2["gstep_id_base"]
                if self.args.lr_fg_begin_step > 0 and gstep_id == self.args.lr_fg_begin_step:
                    self.svox2["grid"].density_data.data[:] = self.args.init_sigma
                lr_sigma = self.svox2["lr_sigma_func"](gstep_id) * self.svox2["lr_sigma_factor"]
                lr_sh = self.svox2["lr_sh_func"](gstep_id) * self.svox2["lr_sh_factor"]
                lr_basis = self.svox2["lr_basis_func"](gstep_id - self.args.lr_basis_begin_step) * self.svox2["lr_basis_factor"]
                lr_sigma_bg = self.svox2["lr_sigma_bg_func"](gstep_id - self.args.lr_basis_begin_step) * self.svox2["lr_basis_factor"]
                lr_color_bg = self.svox2["lr_color_bg_func"](gstep_id - self.args.lr_basis_begin_step) * self.svox2["lr_basis_factor"]
                if not self.args.lr_decay:
                    lr_sigma = self.args.lr_sigma * self.svox2["lr_sigma_factor"]
                    lr_sh = self.args.lr_sh * self.svox2["lr_sh_factor"]
                    lr_basis = self.args.lr_basis * self.svox2["lr_basis_factor"]

                batch_end = min(batch_begin + self.args.batch_size, epoch_size)
                batch_origins = self.svox2["dset"].rays.origins[batch_begin: batch_end]
                batch_dirs = self.svox2["dset"].rays.dirs[batch_begin: batch_end]
                rgb_gt = self.svox2["dset"].rays.gt[batch_begin: batch_end]
                rays = svox2.Rays(batch_origins, batch_dirs)

                #  with Timing("volrend_fused"):
                # rgb_pred_tes = self.svox2["grid"]._volume_render_gradcheck_lerp(rays=rays)
                rgb_pred = self.svox2["grid"].volume_render_fused(rays, rgb_gt,
                        beta_loss=self.args.lambda_beta,
                        sparsity_loss=self.args.lambda_sparsity,
                        randomize=self.args.enable_random)

                #  with Timing("loss_comp"):
                mse = F.mse_loss(rgb_gt, rgb_pred)

                # Stats
                mse_num : float = mse.detach().item()
                psnr = -10.0 * math.log10(mse_num)
                stats['mse'] += mse_num
                stats['psnr'] += psnr
                stats['invsqr_mse'] += 1.0 / mse_num ** 2

                if (iter_id + 1) % self.args.print_every == 0:
                    # Print averaged stats
                    pbar.set_description(f'epoch {self.svox2["epoch_id"]} psnr={psnr:.2f}')
                    for stat_name in stats:
                        stat_val = stats[stat_name] / self.args.print_every
                        self.summary_writer.add_scalar(stat_name, stat_val, global_step=gstep_id)
                        stats[stat_name] = 0.0
                    
                    #  if self.args.lambda_tv > 0.0:
                    #      with torch.no_grad():
                    #          tv = self.svox2["grid"].tv(logalpha=self.args.tv_logalpha, ndc_coeffs=dset.ndc_coeffs)
                    #      self.summary_writer.add_scalar("loss_tv", tv, global_step=gstep_id)
                    #  if self.args.lambda_tv_sh > 0.0:
                    #      with torch.no_grad():
                    #          tv_sh = self.svox2["grid"].tv_color()
                    #      self.summary_writer.add_scalar("loss_tv_sh", tv_sh, global_step=gstep_id)
                    #  with torch.no_grad():
                    #      tv_basis = self.svox2["grid"].tv_basis() #  self.summary_writer.add_scalar("loss_tv_basis", tv_basis, global_step=gstep_id)

                    self.summary_writer.add_scalar("lr_sh", lr_sh, global_step=gstep_id)
                    self.summary_writer.add_scalar("lr_sigma", lr_sigma, global_step=gstep_id)
                    if self.svox2["grid"].basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                        self.summary_writer.add_scalar("lr_basis", lr_basis, global_step=gstep_id)
                    if self.svox2["grid"].use_background:
                        self.summary_writer.add_scalar("lr_sigma_bg", lr_sigma_bg, global_step=gstep_id)
                        self.summary_writer.add_scalar("lr_color_bg", lr_color_bg, global_step=gstep_id)

                    # if self.args.weight_decay_sh < 1.0:
                    #     self.svox2["grid"].sh_data.data *= self.args.weight_decay_sigma
                    # if self.args.weight_decay_sigma < 1.0:
                    #     self.svox2["grid"].density_data.data *= self.args.weight_decay_sh

                # Apply TV/Sparsity regularizers
                if self.args.lambda_tv > 0.0:
                    #  with Timing("tv_inpl"):
                    self.svox2["grid"].inplace_tv_grad(self.svox2["grid"].density_data.grad,
                            scaling=self.args.lambda_tv,
                            sparse_frac=self.args.tv_sparsity,
                            logalpha=self.args.tv_logalpha,
                            ndc_coeffs=self.svox2["dset"].ndc_coeffs,
                            contiguous=self.args.tv_contiguous)
                if self.args.lambda_tv_sh > 0.0:
                    #  with Timing("tv_color_inpl"):
                    self.svox2["grid"].inplace_tv_color_grad(self.svox2["grid"].sh_data.grad,
                            scaling=self.args.lambda_tv_sh,
                            sparse_frac=self.args.tv_sh_sparsity,
                            ndc_coeffs=self.svox2["dset"].ndc_coeffs,
                            contiguous=self.args.tv_contiguous)
                if self.args.lambda_tv_lumisphere > 0.0:
                    self.svox2["grid"].inplace_tv_lumisphere_grad(self.svox2["grid"].sh_data.grad,
                            scaling=self.args.lambda_tv_lumisphere,
                            dir_factor=self.args.tv_lumisphere_dir_factor,
                            sparse_frac=self.args.tv_lumisphere_sparsity,
                            ndc_coeffs=self.svox2["dset"].ndc_coeffs)
                if self.args.lambda_l2_sh > 0.0:
                    self.svox2["grid"].inplace_l2_color_grad(self.svox2["grid"].sh_data.grad,
                            scaling=self.args.lambda_l2_sh)
                if self.svox2["grid"].use_background and (self.args.lambda_tv_background_sigma > 0.0 or self.args.lambda_tv_background_color > 0.0):
                    self.svox2["grid"].inplace_tv_background_grad(self.svox2["grid"].background_data.grad,
                            scaling=self.args.lambda_tv_background_color,
                            scaling_density=self.args.lambda_tv_background_sigma,
                            sparse_frac=self.args.tv_background_sparsity,
                            contiguous=self.args.tv_contiguous)
                if self.args.lambda_tv_basis > 0.0:
                    tv_basis = self.svox2["grid"].tv_basis()
                    loss_tv_basis = tv_basis * self.args.lambda_tv_basis
                    loss_tv_basis.backward()

                # Manual SGD/rmsprop step
                if gstep_id >= self.args.lr_fg_begin_step:
                    self.svox2["grid"].optim_density_step(lr_sigma, beta=self.args.rms_beta, optim=self.args.sigma_optim)
                    self.svox2["grid"].optim_sh_step(lr_sh, beta=self.args.rms_beta, optim=self.args.sh_optim)
                if self.svox2["grid"].use_background:
                    self.svox2["grid"].optim_background_step(lr_sigma_bg, lr_color_bg, beta=self.args.rms_beta, optim=self.args.bg_optim)
                if gstep_id >= self.args.lr_basis_begin_step:
                    if self.svox2["grid"].basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                        self.svox2["grid"].optim_basis_step(lr_basis, beta=self.args.rms_beta, optim=self.args.basis_optim)
                    elif self.svox2["grid"].basis_type == svox2.BASIS_TYPE_MLP:
                        self.svox2["optim_basis_mlp"].step()
                        self.svox2["optim_basis_mlp"].zero_grad()
            self.svox2["epoch_id"] += 1
            self.svox2["gstep_id_base"] += batches_per_epoch
            gc.collect()

    def svox2_eval(self, idx_start=None, idx_end=None):
        dset_test = self.svox2['dset_test']
        old_grid_mask = self.svox2["grid"].links >= 0 
        old_grid_cap = old_grid_mask.sum().item()
        if old_grid_cap > 0:
            self.svox2["grid"].density_data_all[self.svox2["grid"].links_all[old_grid_mask].long()] = self.svox2["grid"].density_data.detach().clone()
            self.svox2["grid"].sh_data_all[self.svox2["grid"].links_all[old_grid_mask].long()] = self.svox2["grid"].sh_data.detach().clone()
        self.svox2["grid"].links = self.svox2["grid"].links_all
        self.svox2["grid"].density_data = torch.nn.Parameter(self.svox2["grid"].density_data_all)
        self.svox2["grid"].sh_data = torch.nn.Parameter(self.svox2["grid"].sh_data_all)
        print('Eval step')
        with torch.no_grad():
            stats_test = {'psnr' : 0.0, 'mse' : 0.0}

            # Standard set
            # N_IMGS_TO_EVAL = min(100 if self.svox2["epoch_id"] >= 0 else 1, dset_test.n_images)
            N_IMGS_TO_EVAL = dset_test.n_images
            N_IMGS_TO_SAVE = N_IMGS_TO_EVAL # if not args.tune_mode else 1
            img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
            img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE)
            img_ids = range(0, dset_test.n_images, img_eval_interval)
            
            if idx_start is None or idx_end is None:
                pass
            else:
                img_ids = img_ids[idx_start:idx_end]

            n_images_gen = 0
            images_test = []
            pbar = tqdm(enumerate(img_ids), total=len(img_ids))

            log_info_mses = []
            
            for i, img_id in pbar:
                c2w = dset_test.c2w[img_id].to(device=self.device)
                cam = svox2.Camera(c2w,
                                   dset_test.intrins.get('fx', img_id),
                                   dset_test.intrins.get('fy', img_id),
                                   dset_test.intrins.get('cx', img_id),
                                   dset_test.intrins.get('cy', img_id),
                                   width=dset_test.get_image_size(img_id)[1],
                                   height=dset_test.get_image_size(img_id)[0],
                                   ndc_coeffs=dset_test.ndc_coeffs)
                rgb_pred_test = self.svox2["grid"].volume_render_image(cam, use_kernel=True)
                rgb_gt_test = dset_test.gt[img_id].to(device=self.device)
                images_test.append((rgb_pred_test.cpu().clamp_max_(1.0).numpy() * 255).astype(np.uint8))
                all_mses = ((rgb_gt_test - rgb_pred_test) ** 2).cpu()
                if i % img_save_interval == 0:
                    img_pred = rgb_pred_test.cpu()
                    img_pred.clamp_max_(1.0)

                    self.summary_writer.add_image(f'test/image_{img_id:04d}',
                            img_pred, global_step=self.svox2["gstep_id_base"], dataformats='HWC')
                    if self.args.log_mse_image:
                        mse_img = all_mses / all_mses.max()
                        self.summary_writer.add_image(f'test/mse_map_{img_id:04d}',
                                mse_img, global_step=self.svox2["gstep_id_base"], dataformats='HWC')
                    if self.args.log_depth_map:
                        depth_img = self.svox2["grid"].volume_render_depth_image(cam,
                                    self.args.log_depth_map_use_thresh if
                                    self.args.log_depth_map_use_thresh else None
                                )
                        depth_img = viridis_cmap(depth_img.cpu())
                        self.summary_writer.add_image(f'test/depth_map_{img_id:04d}',
                                depth_img,
                                global_step=self.svox2["gstep_id_base"], dataformats='HWC')

                if self.args.crop_image_edges != 0:
                    crop_size = self.args.crop_image_edges
                    crop_mask = torch.zeros_like(rgb_gt_test[:,:,0])
                    crop_mask[crop_size:-crop_size, crop_size:-crop_size] = 1.
                else:
                    crop_mask = torch.ones_like(rgb_gt_test[:,:,0])
                crop_mask = crop_mask > 0.5    
                rgb_pred_test = rgb_gt_test = None
                if crop_mask.device.type == 'cuda':
                    crop_mask = crop_mask.cpu()
                mse_num : float = all_mses[crop_mask].mean().item()
                psnr = -10.0 * math.log10(mse_num)
                log_info_mses.append(mse_num)
                if math.isnan(psnr):
                    print('NAN PSNR', i, img_id, mse_num)
                    assert False
                stats_test['mse'] += mse_num
                stats_test['psnr'] += psnr
                pbar.set_description(f'psnr={psnr:.2f}')
                n_images_gen += 1
                
            # save test video
            epoch_id = self.svox2["epoch_id"]
            vid_path = os.path.join(self.args.train_dir, f"test_{epoch_id}.mov")
            imageio.mimwrite(vid_path, images_test, fps=25, quality=8)
            print(f"video saved in {vid_path}")
            if self.svox2["grid"].basis_type == svox2.BASIS_TYPE_3D_TEXTURE or \
               self.svox2["grid"].basis_type == svox2.BASIS_TYPE_MLP:
                 # Add spherical map visualization
                EQ_RESO = 256
                eq_dirs = generate_dirs_equirect(EQ_RESO * 2, EQ_RESO)
                eq_dirs = torch.from_numpy(eq_dirs).to(device=self.device).view(-1, 3)

                if self.svox2["grid"].basis_type == svox2.BASIS_TYPE_MLP:
                    sphfuncs = self.svox2["grid"]._eval_basis_mlp(eq_dirs)
                else:
                    sphfuncs = self.svox2["grid"]._eval_learned_bases(eq_dirs)
                sphfuncs = sphfuncs.view(EQ_RESO, EQ_RESO*2, -1).permute([2, 0, 1]).cpu().numpy()

                stats = [(sphfunc.min(), sphfunc.mean(), sphfunc.max())
                        for sphfunc in sphfuncs]
                sphfuncs_cmapped = [viridis_cmap(sphfunc) for sphfunc in sphfuncs]
                for im, (minv, meanv, maxv) in zip(sphfuncs_cmapped, stats):
                    cv2.putText(im, f"{minv=:.4f} {meanv=:.4f} {maxv=:.4f}", (10, 20),
                                0, 0.5, [255, 0, 0])
                sphfuncs_cmapped = np.concatenate(sphfuncs_cmapped, axis=0)
                self.summary_writer.add_image(f'test/spheric',
                        sphfuncs_cmapped, global_step=self.svox2["gstep_id_base"], dataformats='HWC')
                # END add spherical map visualization

            stats_test['mse'] /= n_images_gen
            stats_test['psnr'] /= n_images_gen
            for stat_name in stats_test:
                self.summary_writer.add_scalar('test/' + stat_name,
                        stats_test[stat_name], global_step=self.svox2["gstep_id_base"])
            self.summary_writer.add_scalar('epoch_id', float(self.svox2["epoch_id"]), global_step=self.svox2["gstep_id_base"])
            print('eval stats:', stats_test)
            np.savetxt(os.path.join(self.args.train_dir, f"eval_mse.txt"), np.array(log_info_mses))
            gc.collect()

    def get_pointcloud_from_svox2_dset(self, idx_start, idx_end):
        points_all = self.svox2["dset"].get_pointcloud_crop(idx_start, idx_end)
        
        for i, points in enumerate(points_all):
            cur_idx = i + idx_start
            x_crop = torch.logical_and(self.bbox[0] < points[:, 0], points[:, 0] < self.bbox[3])
            y_crop = torch.logical_and(self.bbox[1] < points[:, 1], points[:, 1] < self.bbox[4])
            z_crop = torch.logical_and(self.bbox[2] < points[:, 2], points[:, 2] < self.bbox[5])
            mask = torch.logical_and(torch.logical_and(x_crop, y_crop), z_crop)
            points = points[mask]
            scale_points = points - torch.tensor(self.grid_bbox[:3]).to(self.device)
            scale_points /= (self.grid_bbox[3:6] - self.grid_bbox[:3]).max()
            grid_points = (scale_points * self.svox2_reso.max() /self.cam_links_scale).int()
            grid_points_unq = grid_points.unique(dim=0).long()
            self.svox2_cam_links[grid_points_unq[:,0], grid_points_unq[:,1], grid_points_unq[:,2], cur_idx] = True
            points_all[i] = points
        points_all = torch.vstack(points_all)

        # # crop by bbox
        # x_crop = torch.logical_and(self.bbox[0] < points_all[:, 0], points_all[:, 0] < self.bbox[3])
        # y_crop = torch.logical_and(self.bbox[1] < points_all[:, 1], points_all[:, 1] < self.bbox[4])
        # z_crop = torch.logical_and(self.bbox[2] < points_all[:, 2], points_all[:, 2] < self.bbox[5])
        # mask = torch.logical_and(torch.logical_and(x_crop, y_crop), z_crop)

        # points_all = points_all[mask]
        return points_all[None,:,:]

    def links_get_cam(self, links=None):
        if links is None:
            links = self.svox2['grid'].links
        
        print("Now links cap = ", (links >= 0).sum())
        # Downsample links to cam_links resolution
        cam_links_mask = (torch.vstack(torch.where(links >=0)).permute(1,0) / self.cam_links_scale).int().unique(dim=0).long()
        all_grid_cams = torch.where(self.svox2_cam_links[cam_links_mask[:, 0], cam_links_mask[:, 1], cam_links_mask[:, 2]].sum(0) > 0)[0]
        return all_grid_cams
