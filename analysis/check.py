import tinycudann as tcnn
import torch
import numpy as np

from simple_mlp import SimpleMLP

hash_table = tcnn.Encoding(
    n_input_dims = 3,
    encoding_config = {
        "otype": "HashGrid",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 19,
        "base_resolution": 16,
        "per_level_scale": 1.447,
    }
)

direction_encoding = tcnn.Encoding(
    n_input_dims=3,
    encoding_config={
        "otype": "SphericalHarmonics",
        "degree": 3 
    },
)

mlp_head = SimpleMLP(
    n_input_dims=(direction_encoding.n_output_dims + hash_table.n_output_dims),
    n_output_dims=3,
    n_hidden_layers=2,
    n_neurons=64,
    activation='ReLU',
    output_activation='None'
).cuda()

to_save = np.load("./output/MLPDecoder-Test-Data/basketball/params.npy", allow_pickle=True).item()

params_t0 = {}
for k in to_save.keys():
    if isinstance(to_save[k], np.ndarray):
        # Check if the array is stacked over timesteps
        if to_save[k].ndim > 1 or k == 'mlp_head':
            # First dimension is timestep
            params_t0[k] = to_save[k][0]
        else:
            # Not per-timestep data
            params_t0[k] = to_save[k]
    # else:
    #     # For entries that are lists (e.g., mlp_head)
    #     params_t0[k] = to_save[k][0]
        
# print params_t0's key and each shape
for k in params_t0.keys():
    if k == 'mlp_head':
        for kk in params_t0[k].keys():
            print(k, kk, params_t0[k][kk].shape)
    else:
        print(k, params_t0[k].shape)
    

xyz = torch.nn.Parameter(torch.tensor(params_t0['xyz']).cuda().float())
hash_table.params = torch.nn.Parameter(torch.tensor(params_t0['hash_table']).cuda().float())
mlp_head_state_dict = {}

for k, v in params_t0['mlp_head'].items():
    mlp_head_state_dict[k] = torch.from_numpy(v).cuda().float()
    
mlp_head.load_state_dict(mlp_head_state_dict)



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
        
        
aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda')
aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
print(aabb_min, aabb_max)
        
cam_pos = torch.tensor([1.95640779, -1.0, -0.415293604]).cuda().float()
dir_pp = (xyz - cam_pos.repeat(xyz.shape[0], 1))
dir_pp = dir_pp / dir_pp.norm(dim=1, keepdim=True)
xyz = contract_to_unisphere(xyz.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
a = hash_table(xyz).float()
b = direction_encoding(dir_pp).float()
shs = mlp_head(torch.cat([a, b], dim=1)).unsqueeze(1).float()

print(a.shape)
print(b.shape)
print(shs.shape)

for i in range(0, 1):
    print(dir_pp[i])
    print(xyz[i])
    print(a[i].cpu().detach().numpy())
    print(b[i].cpu().detach().numpy())
    print(shs[i].cpu().detach().numpy())
    print("\n")

# # log to concat (a, b) / shs to txt file line by line
# with open("./output/MLPDecoder-Test-Data/basketball/a+b.txt", "w") as f:
#     for i in range(a.shape[0]):
#         f.write(str(a[i].cpu().detach().numpy()) + " " + str(b[i].cpu().detach().numpy()) + "\n")

# with open("./output/MLPDecoder-Test-Data/basketball/shs.txt", "w") as f:
#     for i in range(shs.shape[0]):
#         f.write(str(shs[i].cpu().detach().numpy()) + "\n")