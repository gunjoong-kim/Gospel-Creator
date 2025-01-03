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
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult
from external import calc_ssim, calc_psnr, build_rotation
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
    
def get_loss(dynamic_gaussians, curr_data, is_initial_timestep, op):
    losses = {}
    
    rendervar = dynamic_gaussians.get_rendervar(curr_data['cam'], is_initial_timestep)
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings = curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im = torch.exp(dynamic_gaussians._cam_m[curr_id])[:, None, None] * im + dynamic_gaussians._cam_c[curr_id][:, None, None]
    
    if is_initial_timestep:
        losses['im'] = (1 - op.lambda_dssim) * l1_loss_v1(im, curr_data['im']) + op.lambda_dssim * (1.0 - calc_ssim(im, curr_data['im'])) + op.lambda_mask * torch.mean((torch.sigmoid(dynamic_gaussians._mask)))
    else:
        losses['im'] = (1 - op.lambda_dssim) * l1_loss_v1(im, curr_data['im']) + op.lambda_dssim * (1.0 - calc_ssim(im, curr_data['im']))
    
    dynamic_gaussians.variables['means2D'] = rendervar['means2D']
    segrendervar = dynamic_gaussians.get_rendervar(curr_data['cam'], is_initial_timestep)
    segrendervar['shs'] = None
    segrendervar['colors_precomp'] = dynamic_gaussians._seg_color
    seg, _, _, = Renderer(raster_settings = curr_data['cam'])(**segrendervar)
    losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))
    
    if not is_initial_timestep:
        is_fg = (dynamic_gaussians._seg_color[:, 0] > 0.5).detach()
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]

        rel_rot = quat_mult(fg_rot, dynamic_gaussians.variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[dynamic_gaussians.variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
        
        losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, dynamic_gaussians.variables["prev_offset"],
                                              dynamic_gaussians.variables["neighbor_weight"])

        losses['rot'] = weighted_l2_loss_v2(rel_rot[dynamic_gaussians.variables["neighbor_indices"]], rel_rot[:, None],
                                            dynamic_gaussians.variables["neighbor_weight"])

        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, dynamic_gaussians.variables["neighbor_dist"], dynamic_gaussians.variables["neighbor_weight"])

        losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()

        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]
        losses['bg'] = l1_loss_v2(bg_pts, dynamic_gaussians.variables["init_bg_pts"]) + l1_loss_v2(bg_rot, dynamic_gaussians.variables["init_bg_rot"])

        losses['soft_col_cons'] = l1_loss_v2(dynamic_gaussians.hash_table.params.clone().detach(), dynamic_gaussians.variables['prev_hash']) + l1_loss_v2(dynamic_gaussians.mlp_head.params.clone().detach(), dynamic_gaussians.variables['prev_mlp'])
    
    loss_weights = {'im': 1.0, 'seg': 3.0, 'rigid': 4.0, 'rot': 4.0, 'iso': 2.0, 'floor': 2.0, 'bg': 20.0,
                    'soft_col_cons': 0.01}
    
    loss = sum([loss_weights[k] * v for k, v in losses.items()])
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
    dataset = get_test_dataset(t, md_test, seq)
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
    
    dataset = get_dataset(t, md_train, seq)
    todo_dataset = []
    total_train_psnr = 0
    for c in range(len(md_train['fn'][0])):
        cur_data = get_batch_with_num(todo_dataset, dataset, c)
        im, _, _, = Renderer(raster_settings=cur_data['cam'])(**rendervar)
        curr_id = cur_data['id']
        im = torch.exp(dynamic_gaussians._cam_m[curr_id])[:, None, None] * im + dynamic_gaussians._cam_c[curr_id][:, None, None]
        current_psnr = calc_psnr(im, cur_data['im']).mean()
        total_train_psnr += current_psnr

    print(f"\n")
    print(f"-------------------timestep : {t} evaluation-------------------")
    print(f"Average Test PSNR: {total_test_psnr / len(md_test['fn'][0])}")
    print(f"Average Train PSNR: {total_train_psnr / len(md_train['fn'][0])}")
    print(f"---------------------------------------------------------------")
    print(f"\n")
    
    # open log file and write test psnr
    with open(f"./output/{exp}/{seq}/log.txt", "a") as f:
        f.write(f"timestep {t} test psnr: {total_test_psnr / len(md_test['fn'][0])}\n")

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
    
    # metadata loading
    md_train = json.load(open(f"./data/{seq}/train_meta.json", "r"))
    md_test = json.load(open(f"./data/{seq}/test_meta.json", "r"))

    # model initialization
    dynamic_gaussians = DynamicGaussianModel(mp)
    dynamic_gaussians.load_initial_pt_cld_and_init(seq, md_train)
    dynamic_gaussians.training_setup(op)
    
    num_timesteps = len(md_train['fn'])
    num_timesteps = 30
    for t in range(num_timesteps):
        dataset = get_dataset(t, md_train, seq)
        todo_dataset = []
        is_initial_timestep = (t == 0)
        if not is_initial_timestep:
            dynamic_gaussians.initialize_per_timestep()
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
        if is_initial_timestep:
            dynamic_gaussians.initialize_post_first_timestep()
        
        progress_bar.close()
        output_params.append(dynamic_gaussians.capture())
        evaluate_psnr_and_save_image(dynamic_gaussians, md_test, md_train, t, seq, exp)
    save_params(output_params, seq, exp)

def train_with_dynamic(seq, exp):
    output_params = []
    mp = ModelParams()
    op = OptimizationParams()

    if os.path.exists(f"./output/{exp}/{seq}"):
        print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
        return
    
    # metadata loading
    md_train = json.load(open(f"./data/{seq}/train_meta.json", "r"))
    md_test = json.load(open(f"./data/{seq}/test_meta.json", "r"))

    # model initialization
    dynamic_gaussians = DynamicGaussianModel(mp)
    dynamic_gaussians.load_initial_pt_cld_and_init(seq, md_train)
    dynamic_gaussians.training_setup(op)
    
    num_timesteps = len(md_train['fn'])
    num_timesteps = 33  # 예시로 5로 제한

    # grad_xyz_list: 최종적으로 "각 타임스텝별 평균 gradient"를 담는 리스트
    #  - t=0은 제외하고(t=1부터) 저장한다.
    grad_xyz_list = []
    
    for t in range(num_timesteps):
        dataset = get_dataset(t, md_train, seq)
        todo_dataset = []
        is_initial_timestep = (t == 0)
        if not is_initial_timestep:
            dynamic_gaussians.initialize_per_timestep()

        # 한 타임스텝 내에서 러닝(누적) 합/평균 계산을 위한 변수
        iter_grad_xyz_sum = None
        iter_count = 0

        num_iter_per_timestep = op.initial_step_iter if is_initial_timestep else op.not_initial_step_iter
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        
        for i in range(num_iter_per_timestep):
            curr_data = get_batch_random(todo_dataset, dataset)
            loss = get_loss(dynamic_gaussians, curr_data, is_initial_timestep, op)
            loss.backward()
            
            # ---------- iteration마다 \_xyz의 gradient 누적 ----------
            if dynamic_gaussians._xyz.grad is not None and not is_initial_timestep and i < 30:
                grad_xyz = dynamic_gaussians._xyz.grad.detach().cpu().numpy()  # (N, 3) shape

                # (1) 아직 iter_grad_xyz_sum이 없다면 (첫 iteration), 동일 shape의 0 array 생성
                if iter_grad_xyz_sum is None:
                    iter_grad_xyz_sum = np.zeros_like(grad_xyz)

                # (2) 누적
                #     prune/clone/split 등으로 인해 shape이 바뀌면 에러가 날 수 있음.
                #     이 경우에는 매 iteration 사이에서 point의 개수가 달라지기 때문.
                #     만약 그런 경우라면, "각 iteration 직후 shape이 달라지지 않는 구간"에서만 평균 내야 합니다.
                iter_grad_xyz_sum += grad_xyz
                iter_count += 1
            # ---------------------------------------------------------
            
            with torch.no_grad():
                report_progress(dynamic_gaussians, dataset[0], i, progress_bar)
                if is_initial_timestep:
                    dynamic_gaussians.densify_with_learnable_mask(i, op)
                
                dynamic_gaussians.optimizer.step()
                dynamic_gaussians.optimizer.zero_grad(set_to_none=True)
                dynamic_gaussians.optimizer_net.step()
                dynamic_gaussians.optimizer_net.zero_grad(set_to_none=True)
                dynamic_gaussians.scheduler_net.step()
        
        if is_initial_timestep:
            dynamic_gaussians.initialize_post_first_timestep()

        progress_bar.close()
        
        # ----- 한 타임스텝이 끝난 뒤 (t=0은 건너뛰고), 평균 gradient 계산 -----
        if t > 0:  # t=0(첫 타임스텝)은 스킵
            if iter_grad_xyz_sum is not None and iter_count > 0:
                # 평균
                grad_xyz_avg = iter_grad_xyz_sum / iter_count  # shape: (N,3)
                grad_xyz_list.append(grad_xyz_avg)
            else:
                # 혹은 None을 append하거나, 원하는 대로 처리
                grad_xyz_list.append(None)
        # -------------------------------------------------------------------

        # 시각적 평가 및 파라미터 저장
        output_params.append(dynamic_gaussians.capture())
        evaluate_psnr_and_save_image(dynamic_gaussians, md_test, md_train, t, seq, exp)
    
    # 모든 timestep이 끝난 후 파라미터 저장
    save_params(output_params, seq, exp)
    
    # grad_xyz_list를 numpy 형태로 변환하여 저장
    #  - t=1~4에 해당하는 (N, 3) 배열들이 들어 있음 (처음 타임스텝은 스킵)
    #  - prune/split 때문에 shape이 매 타임스텝마다 다르다면 np.stack 불가
    #    -> allow_pickle=True 설정, 또는 별도 파일 등으로 저장
    np.save(f"./output/{exp}/{seq}/grad_xyz_avg_per_timestep.npy", grad_xyz_list, allow_pickle=True)
    # ----------------------------------------------------------------------

    
if __name__=="__main__":
    exp_name = "graidient-test-33"
    
    for sequence in ["basketball"]:
        train_with_dynamic(sequence, exp_name)
        torch.cuda.empty_cache()