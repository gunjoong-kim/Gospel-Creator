import numpy as np
import os

# Specify the output directory
output_dir = './bin_output/MLPDecoder-Test-Data'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the data
data = np.load('./output/MLPDecoder-Test-Data/basketball/params.npy', allow_pickle=True).item()

data_items = ['xyz', 'rotation', 'scaling', 'opacity', 'hash_table', 'mlp_head', 'compensate']

# For each data item
for data_item in data_items:
    # For each timestep t
    for t in range(33):
        # Extract data at timestep t
        data_t = data[data_item][t]
        
        # If data_item is 'mlp_head', flatten and concatenate
        if data_item == 'mlp_head':
            # data_t is a dictionary
            # Flatten each array and concatenate
            arrays = []
            for key in data_t:
                arrays.append(data_t[key].flatten())
            data_t = np.concatenate(arrays)
            print(data_t.shape)
        
        # Construct the filename with the output directory
        filename = os.path.join(output_dir, f'{data_item}_{t}')
        # Write data_t to binary file
        with open(filename, 'wb') as f:
            f.write(data_t.tobytes())