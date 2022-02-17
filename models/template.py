from collections import namedtuple
import util

import numpy as np
from numpy import prod
import torch
import torch.nn as nn
import torch.nn.functional as F


ZSample = namedtuple("ZSample", "z_pres z_what z_where")
ZLogProb = namedtuple("ZLogProb", "z_pres z_what z_where")
GuideState = namedtuple('GuideState', 'h_l h_c bl_h z_pres z_where z_what')
GenState = namedtuple('GenState', 'h_l h_c z_pres z_where z_what')

class Guide(nn.Module):
    def __init__(self,
                z_what_dim,
                max_strks=2, 
                img_dim=[1,28,28],
                hidden_dim=256, 
                z_where_type='3', 
                execution_guided=False,
                exec_guid_type=None,
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
                sep_where_pres_mlp=True,
                simple_pres=False,
                simple_arch=False,
                residual_no_target=False,
                ):
        super().__init__()
        
        # Parameters
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
            assert execution_guided, "intermediate likelihood needs" + \
                                            "execution_guided = True"
        self.dependent_prior = dependent_prior
        self.spline_decoder = spline_decoder
        self.simple_pres = simple_pres
        if simple_pres:
            assert execution_guided and exec_guid_type=='residual',\
                    "simple_pres requires execution guide and residual"
        self.simple_arch = simple_arch
        if simple_arch:
            assert execution_guided and exec_guid_type=='residual',\
                    "simple_arch requires execution guide and residual"
        self.residual_no_target = residual_no_target
        if residual_no_target:
            assert execution_guided and exec_guid_type=='residual',\
                    "residual_no_target requires execution guide and residual"


        # Internal renderer
        self.execution_guided = execution_guided
        self.exec_guid_type = exec_guid_type
        self.prior_dist = prior_dist
        self.target_in_pos = target_in_pos

        # Inference networks
        # 1. feature_cnn
        #   image -> `cnn_out_dim`-dim hidden representation
        # -> res=50, 33856 when [1, 32, 64]; 16928 when [1, 16, 32]
        # -> res=28, 4608 when
        self.feature_extractor_sharing = feature_extractor_sharing
        self.cnn_out_dim = 16928 if self.img_dim[-1] == 50 else 4608
        self.feature_extractor_out_dim = 256
        self.img_feature_extractor = util.init_cnn(
                                            n_in_channels=1,
                                            n_mid_channels=16,#32, 
                                            n_out_channels=32,#64,
                                            cnn_out_dim=self.cnn_out_dim,
                                            mlp_out_dim=
                                                self.feature_extractor_out_dim,
                                            mlp_hidden_dim=hidden_dim,
                                            num_mlp_layers=1)
        if not self.feature_extractor_sharing:
            self.glimpse_feature_extractor = util.init_cnn(
                                            n_in_channels=1,
                                            n_mid_channels=16,#32, 
                                            n_out_channels=32,#64,
                                            cnn_out_dim=self.cnn_out_dim,
                                            mlp_out_dim=
                                                self.feature_extractor_out_dim,
                                            mlp_hidden_dim=256,
                                            num_mlp_layers=1)


        # 2.1 pres_where_rnn
        self.pr_wr_rnn_in_dim = self.z_pres_dim + self.z_where_dim
        self.pr_wr_rnn_in = []

        if self.z_what_in_pos == 'z_where_rnn':
            self.pr_wr_rnn_in.append('z_what')
            self.pr_wr_rnn_in_dim += self.z_what_dim
        # Canvas: only default in full; adding the target, residual would 
        # make the model not able to generate from prior
        if self.execution_guided:
            self.pr_wr_rnn_in.append('canvas')
            self.pr_wr_rnn_in_dim += self.feature_extractor_out_dim
        if self.target_in_pos == 'RNN' and not self.residual_no_target:
            self.pr_wr_rnn_in.append('target')
            self.pr_wr_rnn_in_dim += self.feature_extractor_out_dim
        if self.target_in_pos == 'RNN' and self.execution_guided and \
           self.exec_guid_type == 'residual':
            self.pr_wr_rnn_in.append('residual')
            self.pr_wr_rnn_in_dim += self.feature_extractor_out_dim
        
        self.pr_wr_rnn_hid_dim = hidden_dim
        self.pr_wr_rnn = torch.nn.GRUCell(self.pr_wr_rnn_in_dim, 
                                          self.pr_wr_rnn_hid_dim)

        # 2.2 pres_where_mlp
        if self.simple_pres:
            self.rsd_power = torch.nn.Parameter(torch.zeros(1)-5., 
                                                        requires_grad=True)
        self.pr_wr_mlp_in = []
        self.pr_wr_mlp_in_dim = 0
        if not self.simple_arch:
            self.pr_wr_mlp_in = ['h']
            self.pr_wr_mlp_in_dim += self.pr_wr_rnn_hid_dim
        if self.target_in_pos == 'MLP' and not self.residual_no_target:
            self.pr_wr_mlp_in.append('target')
            self.pr_wr_mlp_in_dim += self.feature_extractor_out_dim

        if self.target_in_pos == 'MLP' and\
            self.execution_guided and  self.exec_guid_type == 'residual':
            self.pr_wr_mlp_in.append('residual')
            self.pr_wr_mlp_in_dim += self.feature_extractor_out_dim
            self.residual_feature_extractor = util.init_cnn(
                                                n_in_channels=1,
                                                n_mid_channels=16,#32, 
                                                n_out_channels=32,#64,
                                                cnn_out_dim=self.cnn_out_dim,
                                                mlp_out_dim=
                                                self.feature_extractor_out_dim,
                                                mlp_hidden_dim=256,
                                                num_mlp_layers=1)

        self.residual_pixel_count = residual_pixel_count
        if residual_pixel_count:
            self.pr_wr_mlp_in.append('residual_pixel_count')
            self.pr_wr_mlp_in_dim += 1
            
        self.sep_where_pres_mlp = sep_where_pres_mlp

        # 3.1. what_rnn
        self.wt_rnn_in = []
        self.wt_rnn_in_dim = 0

        if self.z_what_in_pos == 'z_what_rnn':
            self.wt_rnn_in.append('z_what')
            self.wt_rnn_in_dim += self.z_what_dim
        # Target (transformed)
        if self.execution_guided:
            self.wt_rnn_in.append('canvas')
            self.wt_rnn_in_dim += self.feature_extractor_out_dim
        if self.target_in_pos == "RNN" and not self.residual_no_target:
            self.wt_rnn_in.append('target')
            self.wt_rnn_in_dim += self.feature_extractor_out_dim
        # if self.target_in_pos == 'RNN' and self.execution_guided and \
        #    self.exec_guid_type == 'residual':
        #     self.wt_rnn_in.append('residual')
        #     self.wt_rnn_in_dim += self.feature_extractor_out_dim

        self.wt_rnn_hid_dim = hidden_dim
        self.wt_rnn = torch.nn.GRUCell(self.wt_rnn_in_dim, 
                                            self.wt_rnn_hid_dim)

        # 3.2 wt_mlp: instantiated in specific modules
        self.wt_mlp_in = []
        self.wt_mlp_in_dim = 0
        if not self.simple_arch:
            self.wt_mlp_in = ['h']
            self.wt_mlp_in_dim += self.wt_rnn_hid_dim
        if self.target_in_pos == 'MLP' and not self.residual_no_target:
            self.wt_mlp_in.append('target')
            self.wt_mlp_in_dim += self.feature_extractor_out_dim
        # if self.target_in_pos == 'MLP' and \
        #    self.execution_guided and self.exec_guid_type == 'residual':
        #     self.wt_mlp_in.append('residual')
        #     self.wt_mlp_in_dim += self.feature_extractor_out_dim

        # 4. baseline
        self.bl_hid_dim = bl_rnn_hid_dim
        self.bl_in_dim = (self.feature_extractor_out_dim  + 
                          self.z_pres_dim + 
                          self.z_where_dim +
                          self.z_what_dim)
        if self.execution_guided:
            self.bl_in_dim += self.feature_extractor_out_dim
        self.bl_rnn = torch.nn.GRUCell(self.bl_in_dim, self.bl_hid_dim)
        self.bl_regressor = util.init_mlp(in_dim=self.bl_hid_dim,
                                          out_dim=1,
                                          hidden_dim=bl_mlp_hid_dim,
                                          num_layers=num_bl_layers)

    def get_rsd_power(self):
        return F.softplus(self.rsd_power)

    def get_img_features(self, imgs, canvas, residual):
        ptcs, bs = shp = imgs.shape[:2]
        img_dim = imgs.shape[2:]
        
        # embed image
        img_embed = self.img_feature_extractor(imgs.view(prod(shp), *img_dim
                                                            )).view(*shp, -1)
        if canvas is not None:
            canvas_embed = self.img_feature_extractor(canvas.view(prod(shp), 
                                                    *img_dim)).view(*shp, -1)

            if self.exec_guid_type == 'residual':
                residual_embed = self.residual_feature_extractor(residual.view(
                                        prod(shp), *img_dim)).view(*shp, -1)
            else: residual_embed = None
        else: canvas_embed, residual_embed = None, None
        return img_embed, canvas_embed, residual_embed

        
    def get_pr_wr_mlp_in(self, img_embed, canvas_embed, residual_embed, 
                        residual, p_state):
        '''Get the input for `pr_wr_mlp` from the current img and p_state
        Args:
            img_embed [ptcs, bs, embed_dim]
            canvas_embed [ptcs, bs, embed_dim] if self.execution_guided or None 
            residual_embed [ptcs, bs, embed_dim] or None
            residual [ptcs, bs, 1, res, res] or None
            p_state GuideState
        Return:
            pr_wr_mlp_in [ptcs, bs, pr_wr_mlp_in_dim]
            h_l [ptcs, bs, h_dim]
            pr_wr_rnn_in [ptcs, bs, pr_wr_rnn_in_dim]
        '''
        ptcs, bs = shp = img_embed.shape[:2]
        # Style RNN input
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
        mlp_in = []
        if 'h' in self.pr_wr_mlp_in:
            mlp_in.append(h_l)
        if 'target' in self.pr_wr_mlp_in:
            mlp_in.append(img_embed.view(prod(shp), -1))
        if 'residual' in self.pr_wr_mlp_in:
            mlp_in.append(residual_embed.view(prod(shp), -1))
        if 'residual_pixel_count' in self.pr_wr_mlp_in:
            residual_pcount = residual.sum([2,3,4]) / prod(self.img_dim)
            mlp_in.append(residual_pcount.view(prod(shp), 1))
        mlp_in = torch.cat(mlp_in, dim=1)

        return (rnn_in.view(*shp, -1), h_l.view(*shp, -1),
                mlp_in.view(*shp, -1))
        
    def get_wt_mlp_in(self, trans_imgs, canvas_embed, residual_embed, 
                                                            p_state):
        '''Get the input for the wt_mlp
        Args:
            trans_imgs: [ptcs, bs, *img_dim]
            canvas_embed: [ptcs, bs, em_dim]
            residual_embed: [ptcs, bs, em_dim]
            p_state
        Return:
            wt_mlp_in: [ptcs, bs, in_dim]
            h_c: [ptcs, bs, in_dim]
        '''
        # Sample z_what, get log_prob
        ptcs, bs = shp = trans_imgs.shape[:2]
        img_dim = trans_imgs.shape[2:]

        # [bs, pts_per_strk, 2, 1]
        if self.feature_extractor_sharing:
            trans_embed = self.img_feature_extractor(trans_imgs.view(prod(shp), 
                                                                    *img_dim))
        else:
            trans_embed = self.glimpse_feature_extractor(trans_imgs.view(prod(
                                                                        shp, 
                                                                    *img_dim)))
            
        # z_what RNN input
        wt_rnn_in = []
        if 'z_what' in self.wt_rnn_in:
            wt_rnn_in.append(p_state.z_what.view(prod(shp), -1))
        if 'canvas' in self.wt_rnn_in:
            wt_rnn_in.append(canvas_embed.view(prod(shp), -1))
        if 'target' in self.wt_rnn_in:
            wt_rnn_in.append(trans_embed.view(prod(shp), -1))
        if 'residual' in self.wt_rnn_in:
            wt_rnn_in.append(residual_embed.view(prod(shp), -1))
        wt_rnn_in = torch.cat(wt_rnn_in, dim=1)
        h_c = self.wt_rnn(wt_rnn_in, p_state.h_c.view(prod(shp), -1))

        # z_what MLP input
        wt_mlp_in = []
        if 'h' in self.wt_mlp_in:
            wt_mlp_in.append(h_c)
        if 'target' in self.wt_mlp_in:
            wt_mlp_in.append(trans_embed.view(prod(shp), -1))
        if 'residual' in self.wt_mlp_in:
            wt_mlp_in.append(residual_embed.view(prod(shp), -1))
        mlp_in = torch.cat(wt_mlp_in, dim=1)

        return mlp_in.view(*shp, -1), h_c.view(*shp, -1)



