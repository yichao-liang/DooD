from collections import namedtuple

from pyparsing import countedArray
import util

import numpy as np
from numpy import prod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, Laplace, Bernoulli

from models.ssp_mlp import *


ZSample = namedtuple("ZSample", "z_pres z_what z_where")
ZLogProb = namedtuple("ZLogProb", "z_pres z_what z_where")
GuideState = namedtuple('GuideState', 'h_l h_c bl_h z_pres z_where z_what')
GenState = namedtuple('GenState', 'h_l h_c z_pres z_where z_what')

class Guide(nn.Module):
    def __init__(self,
                max_strks=2, 
                img_dim=[1,28,28],
                hidden_dim=256, 
                img_feat_dim=256,
                z_where_type='3', 
                use_canvas=False,
                use_residual=None,
                feature_extractor_sharing=True,
                z_what_in_pos='z_where_rnn',
                prior_dist='Independent',
                target_in_pos="RNN",
                intermediate_likelihood=None,
                num_bl_layers=2,
                bl_mlp_hid_dim=512,
                bl_rnn_hid_dim=256,
                maxnorm=True,
                dependent_prior=False,
                spline_decoder=True,
                residual_pixel_count=False,
                sep_where_pres_net=False,
                simple_pres=False,
                no_post_rnn=False,
                residual_no_target=False,
                canvas_only_to_zwhere=False,
                detach_canvas_so_far=True,
                detach_canvas_embed=True,
                detach_rsd=True,
                detach_rsd_embed=True,
                no_pres_rnn=False,
                detach_target_at_pr_mlp=False,
                no_rnn=False,
                only_rsd_ratio_pres=False,
                feature_extractor_type='CNN',
                no_what_post_rnn=False,
                no_pres_post_rnn=False,
                ):
        super().__init__()
        
        # Parameters
        # for NVIL normalization
        self.register_buffer("v", torch.tensor(0.))
        self.register_buffer("c", torch.tensor(0.))
        self.max_strks = max_strks
        self.img_dim = img_dim
        self.img_numel = np.prod(img_dim)
        self.hidden_dim = hidden_dim
        self.z_pres_dim = 1
        self.z_where_type = z_where_type
        self.z_where_dim = util.init_z_where(self.z_where_type).dim
        self.maxnorm = maxnorm
        self.z_what_in_pos = z_what_in_pos
        self.intr_ll = intermediate_likelihood
        if self.intr_ll is not None:
            assert use_canvas, "intermediate likelihood needs" + \
                                            "use_canvas = True"
        self.dependent_prior = dependent_prior
        self.spline_decoder = spline_decoder
        self.detach_canvas_so_far = detach_canvas_so_far
        self.detach_canvas_embed = detach_canvas_embed
        self.detach_rsd = detach_rsd
        self.detach_rsd_embed = detach_rsd_embed
        self.no_pres_rnn = no_pres_rnn
        self.no_rnn = no_rnn
        self.no_what_post_rnn = no_what_post_rnn
        self.no_pres_post_rnn = no_pres_post_rnn
        if no_pres_rnn or no_rnn:
            assert sep_where_pres_net, "need to have seperate where pres net"+\
                "to get h for zpres prior and renderer_mlp_in"

        self.simple_pres = simple_pres
        # if use_residual:
        #     assert use_canvas, "residual needs canvas to be computed"
        if simple_pres:
            assert use_residual,\
                    "simple_pres requires execution guide and residual"
        self.no_post_rnn = no_post_rnn
        if no_post_rnn or no_what_post_rnn:
            assert use_residual,\
                    "no_post_rnn requires execution guide and residual"
        self.residual_no_target = residual_no_target
        if residual_no_target:
            assert use_residual,\
                    "residual_no_target requires execution guide and residual"
        # when this is set to True by the outside, self.constrain_z_pres_param
        # will be set on the last timestep of the reconstruction
        self.constrain_z_pres_param_this_ite = False
        self.constrain_z_pres_param_this_step = False

        self.detach_target_at_pr_mlp = detach_target_at_pr_mlp
        if self.no_pres_rnn or self.no_post_rnn or self.no_rnn:
            self.detach_target_at_pr_mlp = True

        # Internal renderer
        self.use_canvas = use_canvas
        self.use_residual = use_residual
        self.prior_dist = prior_dist
        self.target_in_pos = target_in_pos

        # Inference networks
        # 1. feature_cnn
        #   image -> `cnn_out_dim`-dim hidden representation
        # -> res=50, 33856 when [1, 32, 64]; 16928 when [1, 16, 32]
        # -> res=28, 4608 when
        self.feature_extractor_sharing = feature_extractor_sharing
        # if feature_extractor_type == 'reshape':
        #     self.img_feature_extractor = lambda x: torch.reshape(
        #                                             x, (x.shape[0], -1))
        #     self.feature_extractor_out_dim = 2500
        # elif feature_extractor_type == 'MLP':
        #     in_dim = prod(self.img_dim)
        #     self.feature_extractor_out_dim = 256
        #     hid_dim, num_layers = 512, 3
        #     self.img_feature_extractor = ImageMLP(
        #                                 in_dim=in_dim,
        #                                 out_dim=self.feature_extractor_out_dim,
        #                                 hid_dim=hid_dim,
        #                                 num_layers=num_layers,
        #                             )
        #     if use_residual and not detach_rsd_embed:
        #         # if detach_rsd_embed, not gradient is produced to learn this
        #         self.residual_feature_extractor = ImageMLP(
        #                                 in_dim=in_dim,
        #                                 out_dim=self.feature_extractor_out_dim,
        #                                 hid_dim=hid_dim,
        #                                 num_layers=num_layers,
        #                             )
        # else:
        # arch 1
        # self.cnn_out_dim = 16928 if self.img_dim[-1] == 50 else 4608 # 1568 -- if another maxnorm
        # arch 4
        self.cnn_out_dim = 33856
        self.feature_extractor_out_dim = img_feat_dim
        self.img_feature_extractor = util.init_cnn(
                                            n_in_channels=1,
                                            n_mid_channels=16,#32, 
                                            n_out_channels=32,#64,
                                            cnn_out_dim=self.cnn_out_dim,
                                            mlp_out_dim=
                                                self.feature_extractor_out_dim,
                                            mlp_hidden_dim=
                                                self.feature_extractor_out_dim,
                                            num_mlp_layers=1)
        self.use_sep_trans_img_extractor = False
        if not self.feature_extractor_sharing:
            self.use_sep_trans_img_extractor = True
            self.trans_img_feature_extractor = util.init_cnn(
                                            n_in_channels=1,
                                            n_mid_channels=16,#32, 
                                            n_out_channels=32,#64,
                                            cnn_out_dim=self.cnn_out_dim,
                                            mlp_out_dim=
                                                self.feature_extractor_out_dim,
                                            mlp_hidden_dim=
                                                self.feature_extractor_out_dim,
                                            num_mlp_layers=1)
        self.use_sep_rsd_extractor = False
        self.use_sep_trans_rsd_extractor = False
        if use_residual and not detach_rsd_embed:
            self.use_sep_rsd_extractor = True
            # if detach_rsd_embed, not gradient is produced to learn this
            self.residual_feature_extractor = util.init_cnn(
                                            n_in_channels=1,
                                            n_mid_channels=16,#32, 
                                            n_out_channels=32,#64,
                                            cnn_out_dim=self.cnn_out_dim,
                                            mlp_out_dim=
                                                self.feature_extractor_out_dim,
                                            mlp_hidden_dim=
                                                self.feature_extractor_out_dim,
                                            num_mlp_layers=1)
            if not self.feature_extractor_sharing:
                self.use_sep_trans_rsd_extractor = True
                self.trans_rsd_feature_extractor = util.init_cnn(
                                            n_in_channels=1,
                                            n_mid_channels=16,#32, 
                                            n_out_channels=32,#64,
                                            cnn_out_dim=self.cnn_out_dim,
                                            mlp_out_dim=
                                                self.feature_extractor_out_dim,
                                            mlp_hidden_dim=
                                                self.feature_extractor_out_dim,
                                            num_mlp_layers=1)


        self.sep_where_pres_net = sep_where_pres_net
        # 2.1 pres_where_rnn
        if sep_where_pres_net:
            self.pr_rnn_in = []
            if not self.no_pres_rnn:
                self.pr_rnn_in = ['z_pres']
                self.pr_rnn_in_dim = self.z_pres_dim
            self.wr_rnn_in = ['z_where']
            self.wr_rnn_in_dim = self.z_where_dim

            if self.z_what_in_pos == 'z_where_rnn':
                self.wr_rnn_in.append('z_what')
                self.wr_rnn_in_dim += self.z_what_dim
            # Canvas: only default in full; adding the target, residual would 
            # make the model not able to generate from prior
            if self.use_canvas:
                if not self.no_pres_rnn:
                    self.pr_rnn_in.append('canvas')
                    self.pr_rnn_in_dim += self.feature_extractor_out_dim
                self.wr_rnn_in.append('canvas')
                self.wr_rnn_in_dim += self.feature_extractor_out_dim

            if self.target_in_pos == 'RNN' and not self.residual_no_target:
                self.pr_rnn_in.append('target')
                self.wr_rnn_in.append('target')
                # assert False, "not recommanded unless in ablation"
                self.pr_rnn_in_dim += self.feature_extractor_out_dim
                self.wr_rnn_in_dim += self.feature_extractor_out_dim
            if self.target_in_pos == 'RNN' and self.use_residual:
                # assert False, "not recommanded unless in ablation"
                self.pr_rnn_in.append('residual')
                self.pr_rnn_in_dim += self.feature_extractor_out_dim
                self.wr_rnn_in.append('residual')
                self.wr_rnn_in_dim += self.feature_extractor_out_dim
                
            self.pr_wr_rnn_hid_dim = hidden_dim
            if not self.no_pres_rnn:
                self.pr_rnn = torch.nn.GRUCell(self.pr_rnn_in_dim, 
                                            self.pr_wr_rnn_hid_dim)

            self.wr_rnn = torch.nn.GRUCell(self.wr_rnn_in_dim, 
                                           self.pr_wr_rnn_hid_dim)
        else:
            self.pr_wr_rnn_in = ['z_pres', 'z_where']
            self.pr_wr_rnn_in_dim = self.z_pres_dim + self.z_where_dim

            if self.z_what_in_pos == 'z_where_rnn':
                self.pr_wr_rnn_in.append('z_what')
                self.pr_wr_rnn_in_dim += self.z_what_dim
            # Canvas: only default in full; adding the target, residual would 
            # make the model not able to generate from prior
            if self.use_canvas:
                self.pr_wr_rnn_in.append('canvas')
                self.pr_wr_rnn_in_dim += self.feature_extractor_out_dim

            if self.target_in_pos == 'RNN' and not self.residual_no_target:
                self.pr_wr_rnn_in.append('target')
                # assert False, "not recommanded unless in ablation"
                self.pr_wr_rnn_in_dim += self.feature_extractor_out_dim
            if self.target_in_pos == 'RNN' and self.use_residual:
                # assert False, "not recommanded unless in ablation"
                self.pr_wr_rnn_in.append('residual')
                self.pr_wr_rnn_in_dim += self.feature_extractor_out_dim
                
            self.pr_wr_rnn_hid_dim = hidden_dim
            self.pr_wr_rnn = torch.nn.GRUCell(self.pr_wr_rnn_in_dim, 
                                            self.pr_wr_rnn_hid_dim)

        # 2.2 pres_where_mlp
        if self.simple_pres:
            self.pr_rsd_power = torch.nn.Parameter(torch.zeros(1)+5., 
                                                        requires_grad=True)
        if self.sep_where_pres_net:
            self.pr_mlp_in = []
            self.pr_mlp_in_dim = 0
            self.wr_mlp_in = []
            self.wr_mlp_in_dim = 0
            
            self.only_rsd_ratio_pres = only_rsd_ratio_pres
            if not self.no_post_rnn and not self.no_rnn:
                if not self.no_pres_rnn and not self.no_pres_post_rnn:
                    self.pr_mlp_in = ['h']
                    self.pr_mlp_in_dim += self.pr_wr_rnn_hid_dim
                self.wr_mlp_in = ['h']
                self.wr_mlp_in_dim += self.pr_wr_rnn_hid_dim

            if self.target_in_pos == 'MLP' and not self.residual_no_target:
                if not self.only_rsd_ratio_pres and\
                   not self.residual_no_target_pres:
                    self.pr_mlp_in.append('target')
                    self.pr_mlp_in_dim += self.feature_extractor_out_dim
                self.wr_mlp_in.append('target')
                self.wr_mlp_in_dim += self.feature_extractor_out_dim

            if self.target_in_pos == 'MLP' and self.use_residual:
                if not self.only_rsd_ratio_pres:
                    self.pr_mlp_in.append('residual')
                    self.pr_mlp_in_dim += self.feature_extractor_out_dim
                self.wr_mlp_in.append('residual')
                self.wr_mlp_in_dim += self.feature_extractor_out_dim

            self.residual_pixel_count = residual_pixel_count
            if residual_pixel_count:
                self.pr_mlp_in.append('residual_pixel_count')
                self.pr_mlp_in_dim += 1
                self.wr_mlp_in.append('residual_pixel_count')
                self.wr_mlp_in_dim += 1
        else:
            self.pr_wr_mlp_in = []
            self.pr_wr_mlp_in_dim = 0
            if not self.no_post_rnn and not self.no_rnn:
                self.pr_wr_mlp_in.append('h')
                self.pr_wr_mlp_in_dim += self.pr_wr_rnn_hid_dim
            if self.target_in_pos == 'MLP' and not self.residual_no_target:
                self.pr_wr_mlp_in.append('target')
                self.pr_wr_mlp_in_dim += self.feature_extractor_out_dim

            if self.target_in_pos == 'MLP' and self.use_residual:
                self.pr_wr_mlp_in.append('residual')
                self.pr_wr_mlp_in_dim += self.feature_extractor_out_dim

            self.residual_pixel_count = residual_pixel_count
            if residual_pixel_count:
                self.pr_wr_mlp_in.append('residual_pixel_count')
                self.pr_wr_mlp_in_dim += 1
            
            

        # 3.1. what_rnn
        self.wt_rnn_in = []
        self.wt_rnn_in_dim = 0

        if self.z_what_in_pos == 'z_what_rnn':
            self.wt_rnn_in.append('z_what')
            self.wt_rnn_in_dim += self.z_what_dim
        # Target (transformed)
        if self.use_canvas and not canvas_only_to_zwhere:
            self.wt_rnn_in.append('canvas')
            self.wt_rnn_in_dim += self.feature_extractor_out_dim

        # the full model doesn't have target in at RNN
        if self.target_in_pos == "RNN" and not self.residual_no_target:
            self.wt_rnn_in.append('trans_target')
            # assert False, "not recommanded unless in ablation"
            self.wt_rnn_in_dim += self.feature_extractor_out_dim
        if self.target_in_pos == 'RNN' and self.use_residual:
            # assert False, "not recommanded unless in ablation"
            self.wt_rnn_in.append('trans_residual')
            self.wt_rnn_in_dim += self.feature_extractor_out_dim

        self.wt_rnn_hid_dim = hidden_dim
        self.wt_rnn = torch.nn.GRUCell(self.wt_rnn_in_dim, 
                                            self.wt_rnn_hid_dim)

        # 3.2 wt_mlp: instantiated in specific modules
        self.wt_mlp_in = []
        self.wt_mlp_in_dim = 0
        if (not self.no_post_rnn and 
            not self.no_rnn and 
            not self.no_what_post_rnn):
            self.wt_mlp_in = ['h']
            self.wt_mlp_in_dim += self.wt_rnn_hid_dim
        if self.target_in_pos == 'MLP' and not self.residual_no_target:
            self.wt_mlp_in.append('trans_target')
            self.wt_mlp_in_dim += self.feature_extractor_out_dim
        if self.target_in_pos == 'MLP' and self.use_residual:
            self.wt_mlp_in.append('trans_residual')
            self.wt_mlp_in_dim += self.feature_extractor_out_dim

        # 4. baseline
        self.bl_in = ['target', 'all_prev_z']
        self.bl_hid_dim = bl_rnn_hid_dim
        self.bl_in_dim = (self.feature_extractor_out_dim  + 
                          self.z_pres_dim + 
                          self.z_where_dim +
                          self.z_what_dim)
        if self.use_canvas:
            self.bl_in.append('canvas')
            self.bl_in_dim += self.feature_extractor_out_dim
        self.bl_rnn = torch.nn.GRUCell(self.bl_in_dim, self.bl_hid_dim)
        self.bl_regressor = util.init_mlp(in_dim=self.bl_hid_dim,
                                          out_dim=1,
                                          hidden_dim=bl_mlp_hid_dim,
                                          num_layers=num_bl_layers)
        self.print_model_statistics()

    def print_model_statistics(self):
        # feature extractors
        print("=== Feature extractors ===")
        tot_num_param = sum(p.numel() for p in 
                            self.img_feature_extractor.parameters())
        print(f"## Each feat. extr. has {tot_num_param} parameters")
        print(f"## [{self.use_sep_rsd_extractor}] Use seperate rsd extractor")
        print(f"## [{self.use_sep_trans_rsd_extractor}] "+
                                        "Use seperated rsd glimpse extractor")
        print(f"## [{self.use_sep_trans_img_extractor}] "+
                                        "Use seperated img glimpse extractor\n")

        print("=== Posterior networks ===")
        if self.sep_where_pres_net:
            print("# Seperated z_where, z_pres networks:")
            # pres, where rnn and mlp
            if self.no_pres_rnn:
                print("## No z_pres RNN")
            else:
                print(f"## z_pres rnn in {self.pr_rnn_in_dim}dim: "+
                        f"{self.pr_rnn_in}")
            print(f"## z_pres mlp in {self.pr_mlp_in_dim}dim: "+
                    f"{self.pr_mlp_in}\n")
            print(f"## z_where rnn in {self.wr_rnn_in_dim}dim: "+
                    f"{self.wr_rnn_in}")
            print(f"## z_where mlp in {self.wr_mlp_in_dim}dim: "+
                    f"{self.wr_mlp_in}\n")
                
        else:
            print("# Shared z_where, z_pres networks:")
            print(f"## z_pres_where rnn in {self.pr_wr_rnn_in_dim}dim: "+
                    f"{self.pr_wr_rnn_in}")
            print(f"## z_pres_where mlp in {self.pr_wr_mlp_in_dim}dim: "+
                    f"{self.pr_wr_mlp_in}\n")
        print(f"## z_where rnn in {self.wt_rnn_in_dim}dim: {self.wt_rnn_in}")
        print(f"## z_where mlp in {self.wt_mlp_in_dim}dim: {self.wt_mlp_in}\n")

        print("=== Baseline networks ===")
        print(f"## bl_rnn in {self.bl_in_dim}dim: {self.bl_in}")

    def get_pr_rsd_power(self):
        # return F.softplus(self.pr_rsd_power)
        return F.sigmoid(self.pr_rsd_power)

    def get_img_features(self, imgs, canvas, residual):
        '''
        Args:
            imgs [ptcs, bs, 1, res, res'''
        ptcs, bs = shp = imgs.shape[:2]
        img_dim = imgs.shape[2:]
        
        canvas_embed, residual_embed, rsd_ratio = [None]*3

        # embed image
        img_embed = self.img_feature_extractor(imgs.view(prod(shp), *img_dim
                                                            )).view(*shp, -1)
        if self.use_canvas:
            canvas_embed = self.img_feature_extractor(canvas.view(prod(shp), 
                                                    *img_dim)).view(*shp, -1)

            if self.detach_canvas_embed: canvas_embed = canvas_embed.detach()

        if self.use_residual:
            if self.detach_rsd_embed:
                residual_embed = self.img_feature_extractor(
                        residual.view(prod(shp), *img_dim)).view(*shp, -1
                                                                ).detach()
            else:
                residual_embed = self.residual_feature_extractor(
                        residual.view(prod(shp), *img_dim)).view(*shp, -1)
        if self.simple_pres or self.residual_pixel_count:
            rsd_ratio = residual.sum([2,3,4]) / imgs.sum([2,3,4])
            rsd_ratio = torch.nan_to_num(rsd_ratio, nan=0.)
            # print("rsd_ratio", rsd_ratio.isnan().any())

            if self.detach_rsd_embed: rsd_ratio = rsd_ratio.detach()
               
        return img_embed, canvas_embed, residual_embed, rsd_ratio

        
    def get_pr_wr_mlp_in(self, img_embed, canvas_embed, residual_embed, 
                        rsd_ratio, p_state):
        '''Get the input for `pr_wr_mlp` from the current img and p_state
        Args:
            img_embed [ptcs, bs, embed_dim]
            canvas_embed [ptcs, bs, embed_dim] if self.use_canvas or None 
            residual_embed [ptcs, bs, embed_dim] or None
            residual [ptcs, bs, 1, res, res] or None
            p_state GuideState
        Return:
            pr_wr_mlp_in [ptcs, bs, pr_wr_mlp_in_dim]
            h_l [ptcs, bs, h_dim]
            pr_wr_rnn_in [ptcs, bs, pr_wr_rnn_in_dim]
        '''
        if img_embed != None:
            shp = img_embed.shape[:2]
        else:
            shp = residual_embed.shape[:2]
        # Style RNN input

        if self.sep_where_pres_net:
            if not self.no_pres_rnn:
                pr_rnn_in = [p_state.z_pres.view(prod(shp), -1)]
            wr_rnn_in = [p_state.z_where.view(prod(shp), -1)]

            if 'z_what' in self.pr_rnn_in:
                pr_rnn_in.append(p_state.z_what.view(prod(shp), -1))
            if 'z_what' in self.wr_rnn_in:
                wr_rnn_in.append(p_state.z_what.view(prod(shp), -1))
            if 'canvas' in self.pr_rnn_in:
                pr_rnn_in.append(canvas_embed.view(prod(shp), -1)) 
            if 'canvas' in self.wr_rnn_in:
                wr_rnn_in.append(canvas_embed.view(prod(shp), -1)) 
            if 'target' in self.pr_rnn_in:
                pr_rnn_in.append(img_embed.view(prod(shp), -1))
            if 'target' in self.wr_rnn_in:
                wr_rnn_in.append(img_embed.view(prod(shp), -1))
            if 'residual' in self.pr_rnn_in:
                pr_rnn_in.append(residual_embed.view(prod(shp), -1))
            if 'residual' in self.wr_rnn_in:
                wr_rnn_in.append(residual_embed.view(prod(shp), -1))
            if not self.no_pres_rnn:
                pr_rnn_in = torch.cat(pr_rnn_in, dim=1)
                h_pr = self.pr_rnn(pr_rnn_in, p_state.h_l[0].view(prod(shp), -1))
            else:
                h_pr = p_state.h_l[0]
            
            wr_rnn_in = torch.cat(wr_rnn_in, dim=1)
            h_wr = self.wr_rnn(wr_rnn_in, p_state.h_l[1].view(prod(shp), -1))
        else:
            rnn_in = [p_state.z_pres.view(prod(shp), -1), 
                    p_state.z_where.view(prod(shp), -1)]

            if 'z_what' in self.pr_wr_rnn_in:
                rnn_in.append(p_state.z_what.view(prod(shp), -1))
            if 'canvas' in self.pr_wr_rnn_in:
                rnn_in.append(canvas_embed.view(prod(shp), -1)) 
            if 'target' in self.pr_wr_rnn_in:
                rnn_in.append(img_embed.view(prod(shp), -1))
            if 'residual' in self.pr_wr_rnn_in:
                rnn_in.append(residual_embed.view(prod(shp), -1))
            rnn_in = torch.cat(rnn_in, dim=1)
            h_l = self.pr_wr_rnn(rnn_in, p_state.h_l.view(prod(shp), -1))

        # Style MLP input
        if self.sep_where_pres_net:
            pr_mlp_in, wr_mlp_in = [], []
            if 'h' in self.pr_mlp_in: pr_mlp_in.append(h_pr)
            if 'h' in self.wr_mlp_in: wr_mlp_in.append(h_wr)

            if 'target' in self.pr_mlp_in:
                if self.detach_target_at_pr_mlp:
                    pr_mlp_in.append(img_embed.detach().view(prod(shp), -1))
                else:
                    pr_mlp_in.append(img_embed.view(prod(shp), -1))
            if 'target' in self.wr_mlp_in:
                wr_mlp_in.append(img_embed.view(prod(shp), -1))
                
            if 'residual' in self.pr_mlp_in:
                pr_mlp_in.append(residual_embed.view(prod(shp), -1))
            if 'residual' in self.wr_mlp_in:
                wr_mlp_in.append(residual_embed.view(prod(shp), -1))
            if 'residual_pixel_count' in self.pr_mlp_in:
                pr_mlp_in.append(rsd_ratio.view(prod(shp), 1))
            if 'residual_pixel_count' in self.wr_mlp_in:
                wr_mlp_in.append(rsd_ratio.view(prod(shp), 1))
            pr_mlp_in = torch.cat(pr_mlp_in, dim=1).view(*shp, -1)
            wr_mlp_in = torch.cat(wr_mlp_in, dim=1).view(*shp, -1)
            mlp_in = (pr_mlp_in, wr_mlp_in)
        else:
            mlp_in = []
            if 'h' in self.pr_wr_mlp_in:
                mlp_in.append(h_l)
            if 'target' in self.pr_wr_mlp_in:
                mlp_in.append(img_embed.view(prod(shp), -1))
            if 'residual' in self.pr_wr_mlp_in:
                mlp_in.append(residual_embed.view(prod(shp), -1))
            if 'residual_pixel_count' in self.pr_wr_mlp_in:
                mlp_in.append(rsd_ratio.view(prod(shp), 1))
            mlp_in = torch.cat(mlp_in, dim=1).view(*shp, -1)

        if self.sep_where_pres_net:
            h_l = (h_pr.view(*shp, -1), h_wr.view(*shp, -1))
        else:
            h_l = h_l.view(*shp, -1)

        return mlp_in, h_l
        
    def get_wt_mlp_in(self, trans_imgs, trans_rsd, canvas_embed, p_state):
        '''Get the input for the wt_mlp
        Args:
            trans_imgs: [ptcs, bs, *img_dim]
            trans_rsd: [ptcs, bs, *img_dim]
            canvas_embed: [ptcs, bs, em_dim]
            p_state
        Return:
            wt_mlp_in: [ptcs, bs, in_dim]
            h_c: [ptcs, bs, in_dim]
        '''
        # Sample z_what, get log_prob
        if trans_imgs != None:
            shp, img_dim = trans_imgs.shape[:2], trans_imgs.shape[2:]
        else:
            shp, img_dim = trans_rsd.shape[:2], trans_rsd.shape[2:]

        # [bs, pts_per_strk, 2, 1]
        if not self.residual_no_target:
            if self.feature_extractor_sharing:
                trans_tar_em = self.img_feature_extractor(
                                        trans_imgs.view(prod(shp), *img_dim))
            else:
                trans_tar_em = self.trans_img_feature_extractor(
                                        trans_imgs.view(prod(shp), *img_dim))

        if self.use_residual:
            if self.detach_rsd_embed:
                if self.feature_extractor_sharing:
                    trans_rsd_em = self.img_feature_extractor(
                                trans_rsd.view(prod(shp), *img_dim)).detach()
                else:
                    trans_rsd_em = self.trans_img_feature_extractor(
                                trans_rsd.view(prod(shp), *img_dim)).detach()
            else:
                if self.feature_extractor_sharing:
                    trans_rsd_em = self.residual_feature_extractor(
                                trans_rsd.view(prod(shp), *img_dim))
                else:
                    trans_rsd_em = self.trans_rsd_feature_extractor(
                                trans_rsd.view(prod(shp), *img_dim)).detach()
            
        # z_what RNN input
        wt_rnn_in = []
        if 'z_what' in self.wt_rnn_in:
            wt_rnn_in.append(p_state.z_what.view(prod(shp), -1))
        if 'canvas' in self.wt_rnn_in:
            wt_rnn_in.append(canvas_embed.view(prod(shp), -1))
        if 'trans_target' in self.wt_rnn_in:
            wt_rnn_in.append(trans_tar_em.view(prod(shp), -1))
        if 'trans_residual' in self.wt_rnn_in:
            wt_rnn_in.append(trans_rsd_em.view(prod(shp), -1))
        wt_rnn_in = torch.cat(wt_rnn_in, dim=1)
        h_c = self.wt_rnn(wt_rnn_in, p_state.h_c.view(prod(shp), -1))

        # z_what MLP input
        wt_mlp_in = []
        if 'h' in self.wt_mlp_in:
            wt_mlp_in.append(h_c)
        if 'trans_target' in self.wt_mlp_in:
            wt_mlp_in.append(trans_tar_em.view(prod(shp), -1))
        if 'trans_residual' in self.wt_mlp_in:
            wt_mlp_in.append(trans_rsd_em.view(prod(shp), -1))
        mlp_in = torch.cat(wt_mlp_in, dim=1)

        return mlp_in.view(*shp, -1), h_c.view(*shp, -1)

    def sample_pr_wr(self, p_state, z_pres_p, z_where_loc, z_where_scale, 
                     z_where_cor):
        '''
        z_pres_p [ptcs, bs, -1]
        z_where_loc [ptcs, bs, z_where_dim]
        z_where_scale [ptcs, bs, z_where_dim]
        z_where_cor [ptcs, bs, 1]
        '''
        shp = z_pres_p.shape[:2]

        z_pres_p_tmp = z_pres_p
        # If previous z_pres is 0, force z_pres to 0
        z_pres_p = z_pres_p * p_state.z_pres
        # Numerical stability
        eps = 1e-12
        z_pres_p = z_pres_p.clamp(min=eps, max=1.0-eps)

        # Sample z_pres
        assert z_pres_p.shape == torch.Size([*shp, 1])

        try:
            z_pres_post = Independent(Bernoulli(z_pres_p), 
                                        reinterpreted_batch_ndims=1)
        except:
            breakpoint()
        assert (z_pres_post.event_shape == torch.Size([1]) and
                z_pres_post.batch_shape == torch.Size([*shp]))
        # z_pres: [ptcs, bs, 1]
        z_pres = z_pres_post.sample()

        # If previous z_pres is 0, then this z_pres should also be 0.
        # However, this is sampled from a Bernoulli whose probability is at
        # least eps. In the unlucky event that the sample is 1, we force this
        # to 0 as well.
        z_pres = z_pres * p_state.z_pres

        # log prob: log q(z_pres[i] | x, z_{<i}) if z_pres[i-1]=1, else 0
        # Mask with p_state.z_pres instead of z_pres. 
        # Keep if prev == 1, curr == 0 or 1; remove if prev == 0
        z_pres_lprb = z_pres_post.log_prob(z_pres).unsqueeze(-1) * p_state.z_pres
        
        # Sample z_where, get log_prob
        assert z_where_loc.shape == torch.Size([*shp, self.z_where_dim])
        
        if self.correlated_latent:
            # [*shp, z_where_dim, z_where_dim]
            tril = torch.diag_embed(z_where_scale)
            tril[..., 1, 0] = z_where_cor.squeeze(-1)
            # cov[..., 0, 1] = z_where_cor
            z_where_post = MultivariateNormal(loc=z_where_loc, 
                                              scale_tril=tril)
        else:
            z_where_post = Independent(Normal(z_where_loc, z_where_scale),
                                                    reinterpreted_batch_ndims=1)
        assert (z_where_post.event_shape == torch.Size([self.z_where_dim]) and
                z_where_post.batch_shape == torch.Size([*shp]))        
        # z_where: [ptcs, bs, z_where_dim]
        z_where = z_where_post.rsample()

        # constrain sample
        # if self.constrain_smpl:
        #     z_where = constrain_z_where(self.z_where_type, 
        #                                 z_where.view(prod(shp), -1),
        #                                 clamp=True)
        #     z_where = z_where.view(*shp, -1)
                                                            
        z_where_lprb = z_where_post.log_prob(z_where).unsqueeze(-1) * z_pres
        # z_where_lprb = z_where_lprb.squeeze()
        return z_pres, z_where, z_pres_lprb, z_where_lprb

    def sample_wt(self, z_what_loc, z_what_std, z_what_cor, z_pres):
        '''
            z_what_loc [ptcs, bs, pts_per_strk, 2]
            z_what_std [ptcs, bs, pts_per_strk, 2]
            z_what_cor [ptcs, bs, pts_per_strk, 2]
        '''
        shp = z_what_loc.shape[:2]
        if self.correlated_latent:
            # [ptcs, bs, pts_per_strk, 2, 2]
            tril = torch.diag_embed(z_what_std)
            tril[..., 1, 0] = z_what_cor
            z_what_post = Independent(MultivariateNormal(loc=z_what_loc, 
                                scale_tril=tril), reinterpreted_batch_ndims=1)
        else:
            z_what_post = Independent(Normal(z_what_loc, z_what_std), 
                                                    reinterpreted_batch_ndims=2)
        assert (z_what_post.event_shape == torch.Size([self.pts_per_strk, 2]) 
                and z_what_post.batch_shape == torch.Size([*(shp)]))

        # [ptcs, bs, pts_per_strk, 2] 
        z_what = z_what_post.rsample()
        # constrain samples
        # if self.constrain_smpl:
        #     z_what = constrain_z_what(z_what, clamp=True)

        # log_prob(z_what): [ptcs, bs, 1]
        # z_pres: [ptcs, bs, 1]
        z_what_lprb = z_what_post.log_prob(z_what).unsqueeze(-1) * z_pres
        # z_what_lprb = z_what_lprb.squeeze()
        return z_what, z_what_lprb
