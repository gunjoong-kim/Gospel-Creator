import torch
import numpy as np
import open3d as o3d
import time
import tinycudann as tcnn
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, quat_mult
from external import build_rotation
from colormap import colormap
from copy import deepcopy

# view settings
w, h = 640, 360
near, far = 0.001, 100.0
view_scale = 2.0
fps = 33
def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()

def load_gaussian_video(filepath):
    """load gaussian video from file

    Args:
        filepath (str): path to file

    Returns:
        list: list of gaussian frames
    """
    params = np.load(filepath, allow_pickle=True).item()
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    
    size = int(params['xyz'][0].shape[0])
    total_timestep = int(params['xyz'].shape[0])
        
    gaussian_video = [] # gaussian_frame = {means3D, rotation, opacities, scales, means2D, mlp, hash_table, compensate_table, dir_encoding}
    for t in range(total_timestep):
        # load vd color data
        hash_table = tcnn.Encoding(
            n_input_dims = 3,
            encoding_config = {
                "otype" : "HashGrid",
                "n_levels" : 16,
                "n_features_per_level" : 2,
                "log2_hashmap_size" : 19,
                "base_resolution" : 16,
                "per_level_scale" : 1.447,
            }
            # configurable parameters
        )
        dir_encoding = tcnn.Encoding(
            n_input_dims = 3,
            encoding_config = {
                "otype" : "SphericalHarmonics",
                "degree" : 3
            }
        )
        mlp = tcnn.Network(
            n_input_dims = (dir_encoding.n_output_dims + hash_table.n_output_dims), # maybe 41
            n_output_dims = 3,
            network_config = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            }
            # configurable parameters
        )
        compensate_table = tcnn.Encoding(
            n_input_dims = 3,
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 11,
                "base_resolution": 16,
                "per_level_scale": 1.447,
            }
            # configurable parameters
        )
        hash_table.params = torch.nn.Parameter(params['hash_table'][t].cuda().half())
        mlp.params = torch.nn.Parameter(params['mlp_head'][t].cuda().half())
        compensate_table.params = torch.nn.Parameter(params['compensate'][t].cuda().half())
        
        # load geometrical data
        gaussian_frame = {
            'means3D' : params['xyz'][t],
            'rotations' : torch.nn.functional.normalize(params['rotation'][t]),
            'opacities' : torch.sigmoid(params['opacity'][t]),
            'scales' : torch.exp(params['scaling'][t]),
            'means2D' : torch.zeros_like(params['xyz'][0], device="cuda"),
            'mlp' : mlp,
            'hash_table' : hash_table,
            'compensate_table' : compensate_table,
            'dir_encoding' : dir_encoding
        }
        gaussian_video.append(gaussian_frame)
        
    return gaussian_video

def init_camera(y_angle=0., center_dist=2.4, cam_height=1.3, f_ratio=0.82):
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k

def rgbd2pcd(im, depth, w2c, k, show_depth=False, project_to_cam_w_scale=None):
    d_near = 1.5
    d_far = 6
    invk = torch.inverse(torch.tensor(k).cuda().float())
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    radial_depth = depth[0].reshape(-1)
    def_rays = (invk @ def_pix.T).T
    def_radial_rays = def_rays / torch.linalg.norm(def_rays, ord=2, dim=-1)[:, None]
    pts_cam = def_radial_rays * radial_depth[:, None]
    z_depth = pts_cam[:, 2]
    if project_to_cam_w_scale is not None:
        pts_cam = project_to_cam_w_scale * pts_cam / z_depth[:, None]
    pts4 = torch.concat((pts_cam, pix_ones), 1)
    pts = (c2w @ pts4.T).T[:, :3]
    if show_depth:
        cols = ((z_depth - d_near) / (d_far - d_near))[:, None].repeat(1, 3)
    else:
        cols = torch.permute(im, (1, 2, 0)).reshape(-1, 3)
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols

def contract_to_unisphere(
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

def render(w2c, k, gaussian_frame):
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far)
        dir_pp = (gaussian_frame["means3D"] - cam.campos.repeat(gaussian_frame["means3D"].shape[0], 1))
        dir_pp = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        xyz = contract_to_unisphere(gaussian_frame["means3D"].clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
        shs = (gaussian_frame["mlp"])(torch.cat([gaussian_frame["hash_table"](xyz) + gaussian_frame["compensate_table"](xyz), gaussian_frame["dir_encoding"](dir_pp)], dim=-1)).unsqueeze(1).float()
        renderer_parameter = {
            'means3D': gaussian_frame['means3D'],
            'rotations': gaussian_frame['rotations'],
            'opacities': gaussian_frame['opacities'],
            'scales': gaussian_frame['scales'],
            'means2D': gaussian_frame['means2D'],
            'shs': shs
        }
        im, _, depth, = Renderer(raster_settings=cam)(**renderer_parameter)
        return im, depth

def visualize(video_path):
    # prepare for visuzlization
    gaussian_video = load_gaussian_video(video_path)
    w2c, k = init_camera()
    im, depth = render(w2c, k, gaussian_video[0])
    init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, show_depth=False)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols

    view_k = k * view_scale
    view_k[2, 2] = 1

    # visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width = int(w * view_scale),
        height = int(h * view_scale), 
        visible = True
        )
    vis.add_geometry(pcd)
    view_control = vis.get_view_control()

    cparams = o3d.camera.PinholeCameraParameters()
    cparams.extrinsic = w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(h * view_scale)
    cparams.intrinsic.width = int(w * view_scale)
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = view_scale
    render_options.light_on = False

    start_time = time.time()
    num_timestep = len(gaussian_video)

    while True:
        passed_time = time.time() - start_time
        passed_frames = passed_time * fps

        t = int(passed_frames) % num_timestep

        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / view_scale
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        im, depth = render(w2c, k, gaussian_video[t])
        pts, cols = rgbd2pcd(im, depth, w2c, k, show_depth=False)

        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()
    del view_control
    del vis
    del render_options
    
if __name__ == "__main__":
    video_path = "./output/gradient-test-33/basketball/params.npy"
    visualize(video_path)