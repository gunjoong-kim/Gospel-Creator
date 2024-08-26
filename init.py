#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import sys
import os

class ModelParams:
    def __init__(self):
        self.output_path = None
        self.sh_degree = 0
        self.max_hashmap = 19
        
class OptimizationParams:
    def __init__(self):
        self.initial_step_iter = 10_000
        self.not_initial_step_iter = 2_000
        self.densify_from_iter = 500
        self.densify_until_iter = 5000
        
        self.densify_grad_threshold = 0.0002
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.mask_prune_iter = 1_000
        
        # c3dg version
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 10_000
        
        self.position_lr = 0.00016
        self.seg_lr = 0.0
        self.rotation_lr = 0.001
        self.opacity_lr = 0.05
        self.scale_lr = 0.001
        self.cam_m_lr = 1e-4
        self.cam_c_lr = 1e-4
        self.mask_lr = 0.001
        self.net_lr = 0.01
        
        self.net_lr_step = [3_000, 6_000, 9_000]
        
        self.lambda_mask = 0.0005
        self.lambda_dssim = 0.2
        
        self.percent_dense = 0.01