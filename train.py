import torch
import numpy as np
import os
import json
import copy
from PIL import Image
from random import randint
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from init import ModelParams, OptimizationParams
from dynamic_gaussian_model import DynamicGaussianModel
from helpers import setup_camera, l1_loss_v1
from external import calc_ssim, calc_psnr
# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_FOUND = True
# except ImportError:
#     TENSORBOARD_FOUND = False

def get_dataset(t, md, seq):
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"./data/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        seg = np.array(copy.deepcopy(Image.open(f"./data/{seq}/seg/{fn.replace('.jpg', '.png')}"))).astype(np.float32)
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c})
    return dataset

def get_test_dataset(t, md, seq):
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"./data/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        dataset.append({'cam': cam, 'im': im, 'id': c})
    return dataset

def get_batch_random(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data

def get_batch_with_num(todo_dataset, dataset, num):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(num)
    return curr_data

def initialize_per_timestep(dynamic_gaussians):
    pts = dynamic_gaussians._xyz
    rot = torch.nn.functional.normalize(dynamic_gaussians._rotation)
    new_pts = pts + (pts - dynamic_gaussians.variables["prev_pts"])
    new_rots = torch.nn.functional.normalize(rot + (rot - dynamic_gaussians.variables["prev_rots"]))
    
    is_fg = dynamic_gaussians._seg_color[:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[dynamic_gaussians.variables["neighbor_indices"]] - fg_pts[:, None]
    dynamic_gaussians.variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    dynamic_gaussians.variables['prev_offset'] = prev_offset.detach()
    dynamic_gaussians.variables["prev_hash"] = dynamic_gaussians.hash_table.detach()
    dynamic_gaussians.variables["prev_mlp"] = dynamic_gaussians.mlp_head.detach()
    dynamic_gaussians.variables["prev_pts"] = pts.detach()
    dynamic_gaussians.variables["prev_rot"] = rot.detach()
    
    new_params = {"xyz": new_pts, "rotation": new_rots}
    
def get_loss(dynamic_gaussians, curr_data, is_initial_timestep, op):
    losses = {}
    
    rendervar = dynamic_gaussians.get_rendervar(curr_data['cam'])
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings = curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im = torch.exp(dynamic_gaussians._cam_m[curr_id])[:, None, None] * im + dynamic_gaussians._cam_c[curr_id][:, None, None]
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im'])) + 0.0005 * torch.mean((torch.sigmoid(dynamic_gaussians._mask)))
    dynamic_gaussians.variables['means2D'] = rendervar['means2D']
    
    segrendervar = dynamic_gaussians.get_rendervar(curr_data['cam'])
    segrendervar['shs'] = None
    segrendervar['colors_precomp'] = dynamic_gaussians._seg_color
    seg, _, _, = Renderer(raster_settings = curr_data['cam'])(**segrendervar)
    losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))
    
    loss = 1.0 * losses['im'] + 3.0 * losses['seg']
    seen = radius > 0
    dynamic_gaussians.variables['max_2D_radius'][seen] = torch.max(radius[seen], dynamic_gaussians.variables['max_2D_radius'][seen])
    dynamic_gaussians.variables['seen'] = seen
    
    return loss
    
def report_progress(dynamic_gaussians, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**dynamic_gaussians.get_rendervar(data['cam']))
        curr_id = data['id']
        im = torch.exp(dynamic_gaussians._cam_m[curr_id])[:, None, None] * im + dynamic_gaussians._cam_c[curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        num_of_gaussians = dynamic_gaussians._xyz.shape[0]
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}, num_gaussians: {num_of_gaussians}"})
        progress_bar.update(every_i)
        
def evaluate_psnr_and_save_image(dynamic_gaussians, md_test, md_train, t, seq, exp):
    os.makedirs(f"./output/{exp}/{seq}", exist_ok=True)
    dataset = get_test_dataset(0, md_test, seq)
    todo_dataset = []
    total_test_psnr = 0
    for c in range(len(md_test['fn'][0])):
        cur_data = get_batch_with_num(todo_dataset, dataset, c)
        rendervar = dynamic_gaussians.get_rendervar(cur_data['cam'])
        im, _, _, = Renderer(raster_settings=cur_data['cam'])(**rendervar)
        curr_id = cur_data['id']
        im = torch.exp(dynamic_gaussians._cam_m[curr_id])[:, None, None] * im + dynamic_gaussians._cam_c[curr_id][:, None, None]
        current_psnr = calc_psnr(im, cur_data['im']).mean()
        
        # save image
        im = (im - im.min()) / (im.max() - im.min())
        im = im.detach().squeeze().cpu().numpy() * 255  # Assuming rendered_image is a single image with shape (C, H, W)
        im = im.astype(np.uint8).transpose(1, 2, 0)
        Image.fromarray(im).save(f"./output/{exp}/{seq}/{t}_{curr_id}.png")
        print(f"Test PSNR at camera {curr_id}: {current_psnr}")
        total_test_psnr += current_psnr
    
    dataset = get_dataset(0, md_train, seq)
    todo_dataset = []
    total_train_psnr = 0
    train_image_list = [5, 10, 15, 20]
    for c in range(len(md_train['fn'][0])):
        cur_data = get_batch_with_num(todo_dataset, dataset, c)
        im, _, _, = Renderer(raster_settings=cur_data['cam'])(**rendervar)
        curr_id = cur_data['id']
        im = torch.exp(dynamic_gaussians._cam_m[curr_id])[:, None, None] * im + dynamic_gaussians._cam_c[curr_id][:, None, None]
        current_psnr = calc_psnr(im, cur_data['im']).mean()
        total_train_psnr += current_psnr
        if c in train_image_list:
            # save image
            im = (im - im.min()) / (im.max() - im.min())
            im = im.detach().squeeze().cpu().numpy() * 255
            im = im.astype(np.uint8).transpose(1, 2, 0)
            Image.fromarray(im).save(f"./output/{exp}/{seq}/{t}_train_{curr_id}.png")
            print(f"Train PSNR at camera {curr_id}: {current_psnr}")

    print(f"\n")
    print(f"-------------------first timestep evaluation-------------------")
    print(f"Average Test PSNR: {total_test_psnr / len(md_test['fn'][0])}")
    print(f"Average Train PSNR: {total_train_psnr / len(md_train['fn'][0])}")
    print(f"Number of Gaussians : {dynamic_gaussians._xyz.shape[0]}")
    print(f"---------------------------------------------------------------")
    print(f"\n")
    
def params2cpu(dynamic_gaussians, is_initial_timestep):
    if is_initial_timestep:
        res = {
            'xyz': dynamic_gaussians._xyz.detach().cpu().contiguous().numpy(),
            'rotation': dynamic_gaussians._rotation.detach().cpu().contiguous().numpy(),
            'scaling': dynamic_gaussians._scaling.detach().cpu().contiguous().numpy(),
            'rotation': dynamic_gaussians._rotation.detach().cpu().contiguous().numpy(),
            'seg_color': dynamic_gaussians._seg_color.detach().cpu().contiguous().numpy(),
            'opacity': dynamic_gaussians._opacity.detach().cpu().contiguous().numpy(),
            'cam_m': dynamic_gaussians._cam_m.detach().cpu().contiguous().numpy(),
            'cam_c': dynamic_gaussians._cam_c.detach().cpu().contiguous().numpy(),
            'hash_table': dynamic_gaussians.hash_table.params.detach().cpu().half().numpy(),
            'mlp_head': dynamic_gaussians.mlp_head.params.detach().cpu().half().numpy(),
        }
    else:
        res = {
            'xyz': dynamic_gaussians._xyz.detach().cpu().contiguous().numpy(),
            'rotation': dynamic_gaussians._rotation.detach().cpu().contiguous().numpy(),
            'scaling': dynamic_gaussians._scaling.detach().cpu().contiguous().numpy(),
            'rotation': dynamic_gaussians._rotation.detach().cpu().contiguous().numpy(),
            'seg_color': dynamic_gaussians._seg_color.detach().cpu().contiguous().numpy(),
            'opacity': dynamic_gaussians._opacity.detach().cpu().contiguous().numpy(),
            'cam_m': dynamic_gaussians._cam_m.detach().cpu().contiguous().numpy(),
            'cam_c': dynamic_gaussians._cam_c.detach().cpu().contiguous().numpy(),
            'hash_table': dynamic_gaussians.hash_table.params.detach().cpu().half().numpy(),
            'mlp_head': dynamic_gaussians.mlp_head.params.detach().cpu().half().numpy(),
        }
    return res

def save_params(output_params, seq, exp):
    to_save = {}
    for k in output_params[0].keys():
        if k in output_params[0].keys():
            to_save[k] = np.stack([params[k] for params in output_params])
        else:
            to_save[k] = output_params[0][k]
    os.makedirs(f"./output/{exp}/{seq}", exist_ok=True)
    np.save(f"./output/{exp}/{seq}/params.npy", to_save)

def train(seq, exp):
    output_params = []
    mp = ModelParams()
    op = OptimizationParams()
    if os.path.exists(f"./output/{exp}/{seq}"):
        print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
        return
    
    md_train = json.load(open(f"./data/{seq}/train_meta.json", "r"))

    dynamic_gaussians = DynamicGaussianModel(mp)
    dynamic_gaussians.load_initial_pt_cld_and_init(seq, md_train)
    dynamic_gaussians.training_setup(op)
    
    num_timesteps = len(md_train['fn'])
    num_timesteps = 1
    for t in range(num_timesteps):
        dataset = get_dataset(t, md_train, seq)
        todo_dataset = []
        is_initial_timestep = (t == 0)
        # if not is_initial_timestep:
        #     dynamic_gaussians = initialize_per_timestep(dynamic_gaussians)
        num_iter_per_timestep = op.initial_step_iter if is_initial_timestep else op.not_initial_step_iter
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        for i in range(num_iter_per_timestep):
            #dynamic_gaussians.update_learning_rate(i)
            curr_data = get_batch_random(todo_dataset, dataset)
            loss = get_loss(dynamic_gaussians, curr_data, is_initial_timestep, op)
            loss.backward()
            with torch.no_grad():
                report_progress(dynamic_gaussians, dataset[0], i, progress_bar)
                if is_initial_timestep:
                    dynamic_gaussians.densify_with_learnable_mask(i, op)
                
                dynamic_gaussians.optimizer.step()
                dynamic_gaussians.optimizer.zero_grad(set_to_none = True)
                dynamic_gaussians.optimizer_net.step()
                dynamic_gaussians.optimizer_net.zero_grad(set_to_none = True)
                dynamic_gaussians.scheduler_net.step()
        progress_bar.close()
        output_params.append(params2cpu(dynamic_gaussians, is_initial_timestep))
        if is_initial_timestep:
            md_test = json.load(open(f"./data/{seq}/test_meta.json", "r"))
            evaluate_psnr_and_save_image(dynamic_gaussians, md_test, md_train, t, seq, exp)
    save_params(output_params, seq, exp)
    
    
    
    
if __name__=="__main__":
    exp_name = "mlp-final-size-3"
    
    for sequence in ["basketball"]:
        train(sequence, exp_name)
        torch.cuda.empty_cache()