import torch
import numpy as np
from torch import nn
import tinycudann as tcnn

from utils.general_utils import build_scaling_rotation, strip_symmetric, inverse_sigmoid, get_expon_lr_func, build_rotation
from helpers import o3d_knn

class DynamicGaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, model_params):
        self.active_sh_degree = 0
        self.max_sh_degree = 0
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self._seg_color = torch.empty(0)
        self._cam_m = torch.empty(0)
        self._cam_c = torch.empty(0)
        
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        self.setup_functions()
        
        self.hash_table = tcnn.Encoding(
                 n_input_dims=3,
                 encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": model_params.max_hashmap,
                    "base_resolution": 16,
                    "per_level_scale": 1.447,
                },
        )
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 3 
            },
            )
        self.mlp_head = tcnn.Network(
                n_input_dims=(self.direction_encoding.n_output_dims+self.hash_table.n_output_dims),
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )
            
    def load_initial_pt_cld_and_init(self, seq, md):
        init_pt_cld = np.load(f"./data/{seq}/init_pt_cld.npz")["data"]
        seg = init_pt_cld[:, 6]
        max_cams = 50
        sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
        mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
        
        self._xyz = nn.Parameter(torch.tensor(init_pt_cld[:, :3]).cuda().float().contiguous().requires_grad_(True))
        self._seg_color = nn.Parameter(torch.tensor(np.stack((seg, np.zeros_like(seg), 1 - seg), -1)).cuda().float().contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(np.tile([1, 0, 0, 0], (seg.shape[0], 1))).cuda().float().contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(np.zeros((seg.shape[0], 1))).cuda().float().contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3))).cuda().float().contiguous().requires_grad_(True))
        self._cam_m = nn.Parameter(torch.tensor(np.zeros((max_cams, 3))).cuda().float().contiguous().requires_grad_(True))
        self._cam_c = nn.Parameter(torch.tensor(np.zeros((max_cams, 3))).cuda().float().contiguous().requires_grad_(True))
        self._mask = nn.Parameter(torch.tensor(np.ones((seg.shape[0], 1))).cuda().float().contiguous().requires_grad_(True))
        
        cam_centers = np.linalg.inv(md["w2c"][0])[:, :3, 3]
        scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
        self.variables = {
            'max_2D_radius': torch.zeros(self._xyz.shape[0]).cuda().float(),
            'scene_radius': scene_radius,
            'means2D_gradient_accum': torch.zeros(self._xyz.shape[0]).cuda().float(),
            'denom': torch.zeros(self._xyz.shape[0]).cuda().float(),
        }
    
    def training_setup(self, op):
        self.percent_dense = op.percent_dense
        
        other_params = []
        for params in self.hash_table.parameters():
            other_params.append(params)
        for params in self.mlp_head.parameters():
            other_params.append(params)
            
        l = [
            {'params': [self._xyz], 'lr': op.position_lr * self.variables['scene_radius'], "name": "xyz"},
            {'params': [self._opacity], 'lr': op.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': op.scale_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': op.rotation_lr, "name": "rotation"},
            {'params': [self._mask], 'lr': op.mask_lr, "name": "mask"},
            {'params': [self._seg_color], 'lr': op.seg_lr, "name": "seg_color"},
            {'params': [self._cam_m], 'lr': op.cam_m_lr, "name": "cam_m"},
            {'params': [self._cam_c], 'lr': op.cam_c_lr, "name": "cam_c"},
        ]
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.optimizer_net = torch.optim.Adam(other_params, lr=op.net_lr, eps=1e-15)
        self.scheduler_net = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
            self.optimizer_net, start_factor=0.01, total_iters=100
        ),
            torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_net,
            milestones=op.net_lr_step,
            gamma=0.33,
        ),
        ]
        )
        
        self.xyz_scheduler_args = get_expon_lr_func(lr_init = op.position_lr_init * self.variables['scene_radius'],
                                                    lr_final = op.position_lr_final * self.variables['scene_radius'],
                                                    lr_delay_mult = op.position_lr_delay_mult,
                                                    max_steps = op.position_lr_max_steps)
        
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr
            
    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x
            
    def get_rendervar(self, cam):
        dir_pp = (self._xyz - cam.campos.repeat(self._xyz.shape[0], 1))
        dir_pp = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        xyz = self.contract_to_unisphere(self._xyz.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
        rendervar = {
            'means3D': self._xyz,
            'rotations': torch.nn.functional.normalize(self._rotation),
            'opacities': torch.sigmoid(self._opacity),
            'scales': torch.exp(self._scaling),
            'means2D': torch.zeros_like(self._xyz, requires_grad=True, device="cuda") + 0,
            'shs': self.mlp_head(torch.cat([self.hash_table(xyz), self.direction_encoding(dir_pp)], dim=-1)).unsqueeze(1).float()
        }
        return rendervar
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            group_name = group.get("name", None)
            if group_name in ["cam_c", "cam_m", "mlp_head", "hash_table"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def densification_postfix(self, new_xyz, new_scaling, new_rotation, new_opacity, new_mask, new_seg_color):
        d = {
            "xyz": new_xyz,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "opacity": new_opacity,
            "mask": new_mask,
            "seg_color": new_seg_color
        }
        
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._opacity = optimizable_tensors["opacity"]
        self._mask = optimizable_tensors["mask"]
        self._seg_color = optimizable_tensors["seg_color"]
        
        self.variables['means2D_gradient_accum'] = torch.zeros(self._xyz.shape[0], device="cuda")
        self.variables['denom'] = torch.zeros(self._xyz.shape[0], device="cuda")
        self.variables['max_2D_radius'] = torch.zeros(self._xyz.shape[0], device="cuda")
        
    def prune_optimizer(self, to_prune):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            group_name = group.get("name", None)
            if group_name in ["cam_c", "cam_m", "mlp_head", "hash_table"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][to_prune]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_prune]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][to_prune].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][to_prune].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
        
    def prune_points(self, to_prune):
        to_keep = ~to_prune
        optimizable_tensors = self.prune_optimizer(to_keep)
        
        self._xyz = optimizable_tensors["xyz"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._opacity = optimizable_tensors["opacity"]
        self._mask = optimizable_tensors["mask"]
        self._seg_color = optimizable_tensors["seg_color"]
        
        self.variables['means2D_gradient_accum'] = self.variables['means2D_gradient_accum'][to_keep]
        self.variables['denom'] = self.variables['denom'][to_keep]
        self.variables['max_2D_radius'] = self.variables['max_2D_radius'][to_keep]
        
    def clone(self, grads, grad_threshold):
        to_clone = torch.logical_and(grads >= grad_threshold, (torch.max(torch.exp(self._scaling), dim=1).values <= 0.01 * self.variables['scene_radius']))
        
        new_xyz = self._xyz[to_clone]
        new_scaling = self._scaling[to_clone]
        new_rotation = self._rotation[to_clone]
        new_opacity = self._opacity[to_clone]
        new_mask = self._mask[to_clone]
        new_seg_color = self._seg_color[to_clone]
        
        self.densification_postfix(new_xyz, new_scaling, new_rotation, new_opacity, new_mask, new_seg_color)    
    
    def split(self, grads, grad_threshold, N=2):
        n_init_points = self._xyz.shape[0]
        padded_grads = torch.zeros((n_init_points), device="cuda")
        padded_grads[:grads.shape[0]] = grads
        to_split = torch.logical_and(padded_grads >= grad_threshold, (torch.max(torch.exp(self._scaling), dim=1).values > 0.01 * self.variables['scene_radius']))
        
        stds = torch.exp(self._scaling)[to_split].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(means, stds)
        rots = build_rotation(self._rotation[to_split]).repeat(N, 1, 1)
        
        new_xyz = self._xyz[to_split].repeat(N, 1) + torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        new_scaling = torch.log(torch.exp(self._scaling[to_split].repeat(N, 1)) / (0.8 * N))
        new_rotation = self._rotation[to_split].repeat(N, 1)
        new_opacity = self._opacity[to_split].repeat(N, 1)
        new_mask = self._mask[to_split].repeat(N, 1)
        new_seg_color = self._seg_color[to_split].repeat(N, 1)
        
        self.densification_postfix(new_xyz, new_scaling, new_rotation, new_opacity, new_mask, new_seg_color)
        
        to_prune = torch.cat((to_split, torch.zeros(N * to_split.sum(), dtype=torch.bool, device="cuda")))
        self.prune_points(to_prune)
        
    def densify_with_learnable_mask(self, iteration, op):
        if iteration <= op.densify_until_iter:
            self.variables['means2D_gradient_accum'][self.variables['seen']] += torch.norm(self.variables['means2D'].grad[self.variables['seen'], :2], dim=-1)
            self.variables['denom'][self.variables['seen']] += 1
            grad_thresh = 0.0002
            if (iteration >= op.densify_from_iter) and (iteration % op.densification_interval == 0):
                grads = self.variables['means2D_gradient_accum'] / self.variables['denom']
                grads[grads.isnan()] = 0.0
                
                self.clone(grads, grad_thresh)
                self.split(grads, grad_thresh)
                
                remove_threshold = 0.25 if iteration == 5000 else 0.005
                to_prune = torch.logical_or((torch.sigmoid(self._mask) < 0.01).squeeze(), (torch.sigmoid(self._opacity) < remove_threshold).squeeze())
                if iteration >= 3000:
                    big_points_ws = torch.exp(self._scaling).max(dim=1).values > 0.1 * self.variables['scene_radius']
                    to_prune = torch.logical_or(to_prune, big_points_ws)
                self.prune_points(to_prune)
                torch.cuda.empty_cache()
                
            # if iteration > 0 and iteration % 3000 == 0:
            #     self.reset_opacity()
                
        if iteration > op.densify_until_iter:
            remove_threshold = 0.25 if iteration == 5000 else 0.005
            to_prune = torch.logical_or((torch.sigmoid(self._mask) < 0.01).squeeze(), (torch.sigmoid(self._opacity) < remove_threshold).squeeze())
            if iteration >= 3000:
                big_points_ws = torch.exp(self._scaling).max(dim=1).values > 0.1 * self.variables['scene_radius']
                to_prune = torch.logical_or(to_prune, big_points_ws)
            self.prune_points(to_prune)
            torch.cuda.empty_cache()
            
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self._opacity, torch.ones_like(self._opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
            
    def initialize_post_first_timestep(self, num_knn=20):
        is_fg = self._seg_color[:, 0] > 0.5
        init_fg_pts = self._xyz[is_fg]
        init_bg_pts = self._xyz[~is_fg]
        init_bg_rot = torch.nn.functional.normalize(self._rotation[~is_fg])
        neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
        neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
        neighbor_dist = np.sqrt(neighbor_sq_dist)
        self.variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
        self.variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
        self.variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()
    
        self.variables["init_bg_pts"] = init_bg_pts.detach()
        self.variables["init_bg_rot"] = init_bg_rot.detach()
        self.variables["prev_pts"] = self._xyz.detach()
        self.variables["prev_rots"] = torch.nn.functional.normalize(self._rotation).detach()
        params_to_fix = ['opacity', 'scaling', 'cam_m', 'cam_c', 'mask']
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in params_to_fix:
                param_group['lr'] = 0.0
            
    def initialize_per_timestep(self):
        pts = self._xyz
        rot = torch.nn.functional.normalize(self._rotation)
        new_pts = pts + (pts - self.variables["prev_pts"])
        new_rots = torch.nn.functional.normalize(rot + (rot - self.variables["prev_rots"]))

        is_fg = self._seg_color[:, 0] > 0.5
        prev_inv_rot_fg = rot[is_fg]
        prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
        fg_pts = pts[is_fg]
        prev_offset = fg_pts[self.variables["neighbor_indices"]] - fg_pts[:, None]
        self.variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
        self.variables['prev_offset'] = prev_offset.detach()
        self.variables["prev_hash"] = self.hash_table.params.detach()
        self.variables["prev_mlp"] = self.mlp_head.params.detach()
        self.variables["prev_pts"] = pts.detach()
        self.variables["prev_rots"] = rot.detach()

        optimizable_tensor = self.replace_tensor_to_optimizer(new_pts, "xyz")
        self._xyz = optimizable_tensor["xyz"]
        optimizable_tensor = self.replace_tensor_to_optimizer(new_rots, "rotation")
        self._rotation = optimizable_tensor["rotation"]
        
    def capture(self):
        return {
            'xyz': self._xyz.detach().cpu().contiguous().numpy(),
            'rotation': self._rotation.detach().cpu().contiguous().numpy(),
            'scaling': self._scaling.detach().cpu().contiguous().numpy(),
            'rotation': self._rotation.detach().cpu().contiguous().numpy(),
            'seg_color': self._seg_color.detach().cpu().contiguous().numpy(),
            'opacity': self._opacity.detach().cpu().contiguous().numpy(),
            'cam_m': self._cam_m.detach().cpu().contiguous().numpy(),
            'cam_c': self._cam_c.detach().cpu().contiguous().numpy(),
            'hash_table': self.hash_table.params.detach().cpu().half().numpy(),
            'mlp_head': self.mlp_head.params.detach().cpu().half().numpy(),
        }
                
                
                