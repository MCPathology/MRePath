'''import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
import numpy as np
from scipy.signal import convolve2d
import scienceplots

def plot_weight_distribution(matrix,save,color):
    flattened_values = matrix.flatten()
    # print(matrix.size)
    bin_edges = np.arange(-0.4, 0.3, 0.01)
    plt.figure(figsize=(10, 3))
    plt.hist(flattened_values, bins=bin_edges, color=color, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Matrix Values')
    plt.xlabel('Value')
    plt.xlim(0,0.3)
    plt.ylabel('Frequency')
    plt.ylim(0, 60)
    plt.grid(True)
    plt.savefig(save)

def moving_average(data, window_size):
    
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    
def rescale_to_range(array, new_min, new_max):
    old_min = np.min(array)
    old_max = np.max(array)
    
    scaled_array = (array - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    
    return scaled_array
    
def map_to_range(data, min_val, max_val):
        print(min_val)
        print(max_val)
        data_mapped = np.where(data < 0, -data, data)
        return data_mapped
        
def smooth_edges(data, kernel_size=3):
    
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd number.")
    
    
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    smoothed_data = convolve2d(data, kernel, mode='same', boundary='wrap')
    
    result = data.copy()
    
    edge_indices = [(0, -1), (1, -1), (-1, 0), (-1, 1)]
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if i in [0, data.shape[0] - 1] or j in [0, data.shape[1] - 1]:
                result[i, j] = smoothed_data[i, j]
                
    return result
 
model = torch.load('heatmap.pt')
#print(state_dict)
weights = model['classifier.weight'].cpu().detach().numpy()@model['mm.0.weight'].cpu().detach().numpy()
# print(weights.size)
w_g = weights[:,:256]
w_p = weights[:,256:]
w_g_abs = w_g
w_p_abs = w_p
plt.style.use('science')

         

window_size = 50


w_g_move = np.array([moving_average(row, window_size) for row in w_g_abs])
w_p_move = np.array([moving_average(row, window_size) for row in w_p_abs])
   
smooth_max = max(w_g_move.max(), w_p_move.max())

g_r=np.abs(rescale_to_range(w_g_move,-0.4,0.3))
p_r=np.abs(rescale_to_range(w_p_move,-0.4,0.3))
r = np.concatenate((g_r,p_r),axis=1)
print('{}+{}:'.format(np.mean(r),np.std(r)))
print(np.sum(g_r>np.mean(r)+np.std(r)))
print(np.sum(p_r>np.mean(r)+np.std(r)))

#plot_weight_distribution(np.abs(rescale_to_range(w_p_move,-0.4,0.3)),'d_p.jpg','red')      

#print(w_g_move)

w_g_smooth = map_to_range(w_g_move, w_g_move.min(), w_g_move.max())
w_p_smooth = map_to_range(w_p_move,  w_p_move.min(), w_p_move.max())
#print(w_g_smooth)
global_min = min(w_g_smooth.min(), w_p_smooth.min())
global_max = smooth_max
            
sigma = 5
w_g_gaussian = smooth_edges(w_g_smooth)
w_p_gaussian = smooth_edges(w_p_smooth)
global_max = max(w_g_gaussian.max(), w_p_gaussian.max())

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(w_g_gaussian, aspect='auto', cmap='jet',extent=[0,256,0,4],vmin=global_min, vmax=global_max)
plt.colorbar()
plt.title('Heatmap of |w_g|')
plt.xlabel('Features')
plt.ylabel('Samples')

plt.subplot(1, 2, 2)
plt.imshow(w_p_gaussian, aspect='auto', cmap='jet',extent=[0,256,0,4],vmin=global_min, vmax=global_max)
plt.colorbar()
plt.title('Heatmap of |w_p|')
plt.xlabel('Features')
plt.ylabel('Samples')

            


plt.tight_layout()
plt.savefig('heatmap.jpg',dpi=600)'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import scienceplots

plt.style.use('science')
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def rescale_to_range(array, new_min, new_max):
    old_min = np.min(array)
    old_max = np.max(array)
    scaled_array = (array - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return scaled_array

def map_to_range(data, min_val, max_val):
    data_mapped = np.where(data < 0, -data, data)
    return data_mapped

def smooth_edges(data, kernel_size=3):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd number.")
    
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    smoothed_data = convolve2d(data, kernel, mode='same', boundary='wrap')
    
    result = data.copy()
    
    edge_indices = [(0, -1), (1, -1), (-1, 0), (-1, 1)]
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if i in [0, data.shape[0] - 1] or j in [0, data.shape[1] - 1]:
                result[i, j] = smoothed_data[i, j]
                
    return result

def smooth_with_gaussian(data, sigma=1):
    return gaussian_filter(data, sigma=sigma)

model = torch.load('heatmap.pt')
weights = model['classifier.weight'].cpu().detach().numpy() @ model['mm.0.weight'].cpu().detach().numpy()
plt.style.use('science')
w_g = weights[:, :256]
w_p = weights[:, 256:]
w_g_abs = w_g
w_p_abs = w_p

window_size = 50

w_g_move = np.array([moving_average(row, window_size) for row in w_g_abs])
w_p_move = np.array([moving_average(row, window_size) for row in w_p_abs])

g_r = np.abs(rescale_to_range(w_g_move,-0.4,0.3))
p_r = np.abs(rescale_to_range(w_p_move,-0.4,0.3))

w_g_smooth = map_to_range(w_g_move, w_g_move.min(), w_g_move.max())
w_p_smooth = map_to_range(w_p_move, w_p_move.min(), w_p_move.max())

sigma = 1.5
w_g_gaussian = smooth_with_gaussian(w_g_smooth, sigma=sigma)
w_p_gaussian = smooth_with_gaussian(w_p_smooth, sigma=sigma)
global_max=max(w_g_gaussian.max(), w_p_gaussian.max())

# Create a grid for the 3D plot
x_g, y_g = np.meshgrid(np.arange(w_g_gaussian.shape[1]), np.arange(w_g_gaussian.shape[0]))
x_p, y_p = np.meshgrid(np.arange(w_p_gaussian.shape[1]), np.arange(w_p_gaussian.shape[0]))

fig = plt.figure(figsize=(10, 10))

ax1 = fig.add_subplot(211, projection='3d')
ax1.plot_surface(x_g, y_g, w_g_gaussian, cmap='jet', rstride=1, cstride=1,
                 linewidth=0.1,vmin=0, vmax=global_max)
# Remove title and increase font size

ax1.set_zlim(0, global_max)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_zticklabels([])

# Plot for |w_p|
ax2 = fig.add_subplot(212, projection='3d')
ax2.plot_surface(x_p, y_p, w_p_gaussian, cmap='jet', rstride=1, cstride=1,
                 linewidth=0.1,vmin=0,vmax=global_max)
# Remove title and increase font size

ax2.set_zlim(0, global_max)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])

plt.savefig('3d_heatmap.jpg', dpi=600) # Increase DPI for higher resolution