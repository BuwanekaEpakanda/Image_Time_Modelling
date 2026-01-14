import os
import sys

import time
import numpy as np
from skimage import color
from tqdm import tqdm

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
#import torchvision.models as models
from torch.optim.lr_scheduler import LambdaLR

from utils import normalize, measure, psnr

from fractions import Fraction

#torch.cuda.set_device(0)


#save_dir = "./Results"
#input_img = "XYT_Sample_Data/img_155_20250629.png"
IMG_DIR = "XYT_Sample_Data_Part_02"#"XYT_Sample_Data"
RESULTS_DIR = IMG_DIR + "_INR_Results"

TRAIN_GRAPH_DIR = os.path.join(RESULTS_DIR, "Training_Graphs")
TRAIN_WEIGHTS_DIR = os.path.join(RESULTS_DIR, "Training_Weights")
TRAIN_INFERECE_DIR = os.path.join(RESULTS_DIR, "Inference_Results")


os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TRAIN_GRAPH_DIR, exist_ok=True)
os.makedirs(TRAIN_WEIGHTS_DIR, exist_ok=True)
os.makedirs(TRAIN_INFERECE_DIR, exist_ok=True)

save_path = os.path.join(RESULTS_DIR, "RGB")



os.makedirs(save_path, exist_ok=True)

# Model configs
niters = 10000
learning_rate = 0.01  # Learning rate
decay_rate = 0.01 # Decay rate

act_params = 2
in_features = 3
out_features = 3
hidden_layers = 3
hidden_features = 256
maxpoints = 256*256
from torch.utils.data import Dataset

# Dataset for time-series images
class TimeSeriesImageDataset(Dataset):
    def __init__(self, images):
        """
        images: numpy array of shape (T, H, W, 3) with values in [0, 1]
        """
        T, H, W, C = images.shape

        # Normalize spatial coordinates to [-1, 1]
        x = np.linspace(-1, 1, W)
        y = np.linspace(-1, 1, H)
        t = np.linspace(-1, 1, T)

        # Create spatial grid (H*W, 2)
        Y, X = np.meshgrid(y, x, indexing='ij')
        xy = np.stack([X.ravel(), Y.ravel()], axis=-1)  # shape: (H*W, 2)

        # Repeat spatial grid for each time step (T times)
        xy_repeated = np.tile(xy, (T, 1))  # shape: (H*W*T, 2)

        # Time column (time varies slowest)
        t_column = np.repeat(t, H * W)[:, None]  # shape: (H*W*T, 1)

        # Final coordinates
        coords = np.concatenate([xy_repeated, t_column], axis=-1)  # shape: (H*W*T, 3)

        # Flatten image RGB values to match coordinate order
        #rgb = images.transpose(1, 2, 0, 3).reshape(-1, C)  # shape: (H*W*T, 3)

        rgb = images.reshape(-1, 3)
        # Explanation: transpose to (H, W, T, 3) so pixels align with coords

        # Convert to torch tensors
        self.coords = torch.from_numpy(coords).float()
        self.rgb = torch.from_numpy(rgb).float()

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx], self.rgb[idx]

# Function to load images from a directory
def load_images_from_dir(image_dir):
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png'))])
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = np.array(img) / 255.0
        images.append(img)
    return np.stack(images)  # (T, H, W, 3)
images = load_images_from_dir(IMG_DIR)
dataset = TimeSeriesImageDataset(images)
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)


T, H, W, _ = images.shape
# print( T, H, W, _)
# x, y = torch.linspace(-1, 1, W), torch.linspace(-1, 1, H)
# X, Y = torch.meshgrid(x, y, indexing='xy')
coords = dataset.coords#torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]

gt = dataset.rgb#torch.tensor(im).cuda().reshape(H * W, 3)[None, ...]

img = gt
task = 'image'

mse_array = torch.zeros(niters).cuda()
time_array = torch.zeros_like(mse_array)
consts, psnr_vals, step_array = [], [], []

best_mse = torch.tensor(float('inf'))
best_img = None

rec = torch.zeros_like(gt)
class RaisedCosineImpulseResponseLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 is_first=False,
                 beta0=0.25,
                 T0=0.1,
                 c0=0,
                 eps=1e-8,
                 out_real=False,
                 trainable=False):
        super().__init__()
        self.beta0 = beta0
        self.T0 = T0
        self.c0 = c0
        self.eps= eps
        self.is_first = is_first
        self.out_real = out_real

        self.in_features = in_features
        self.out_features = out_features

        if self.is_first: dtype = torch.float
        else: dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.c0 = nn.Parameter(self.c0*torch.ones(1), trainable)
        self.beta0 = nn.Parameter(self.beta0*torch.ones(1), False)
        self.T0 = nn.Parameter(self.T0*torch.ones(1), trainable)
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

        # nn.init.uniform_(self.linear.weight, -1/self.in_features, 1/self.in_features)  # Initialize weights (new method)

    def forward(self, input):
        lin = self.linear(input).cuda()
        if not self.is_first:
            lin = lin / torch.abs(lin)

        f1 = (1 / self.T0) * torch.sinc(lin / self.T0) * torch.cos(torch.pi * self.beta0 * lin / self.T0)
        f2 = 1 - (2 * self.beta0 * lin / self.T0) ** 2 + self.eps  # self.eps is set to 1e-8, to avoid dviding by 0.
        theta = 2 * torch.pi * self.c0 * lin * 1j
        rc = (f1 / f2)

        out = rc * torch.exp(theta)
        # NOTE: Fixed normalization issue here
        if not self.is_first:
            out = out / torch.abs(out)
        if self.out_real:
            return out.real
        else:
            return out

class INR(nn.Module):
    def __init__(self, in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 outermost_linear=True,
                 first_omega_0=30,
                 hidden_omega_0=30.,
                 beta0=0.25,
                 T0=0.1,
                 c0=0,
                 scale=None,
                 pos_encode=False,
                 sidelength=512,
                 fn_samples=None,
                 use_nyquist=True):
        super().__init__()

        self.nonlin = RaisedCosineImpulseResponseLayer
        dtype = torch.cfloat

        self.net = []
        self.net.append(self.nonlin(in_features,
                                    hidden_features,
                                    beta0=beta0,
                                    T0=T0,
                                    c0=c0,
                                    is_first=True,
                                    trainable=True))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features,
                                        beta0=beta0,
                                        T0=T0,
                                        c0=c0,
                                        trainable=True))

        final_linear = nn.Sequential(nn.Linear(hidden_features, out_features, dtype=dtype),
                                     nn.Sigmoid())
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output.real
model = INR(
    in_features=in_features,
    out_features=out_features,
    hidden_features=hidden_features,
    hidden_layers=hidden_layers,
    beta0=0.05,
    T0=5.5,
    c0=1.8)

model.cuda()

# H = 256
# W = 256

optim = torch.optim.Adam(lr=learning_rate * min(1, maxpoints / (H * W)),
                         params=model.parameters())

scheduler = LambdaLR(optim, lambda x: decay_rate ** min(x / niters, 1))
tot_iters = 0
sum(p.numel() for p in model.parameters())

# save_path = os.path.join(save_dir, input_img.split('/')[-1])


coords = coords.unsqueeze(0)
rec = rec.unsqueeze(0)
gt = gt.unsqueeze(0)
# print(coords.shape)
# print(rec.shape)
# print(gt.shape)
rec = rec.cuda()
gt = gt.cuda()
coords = coords.cuda()
img = img.cuda()
k = 0
# print(coords[0,256*k:256*(k+256),:])

# plt.imshow( (gt[0,256*k:256*(k+256),:].reshape(256,-1,3)).detach().cpu().numpy())
# plt.show()


def MSE_loss(x, y): return ((x - y) ** 2).mean()

init_time = time.time()
tbar = tqdm(range(niters))

# -------------------------------- Training Loop ---------------------------------
for epoch in (tbar):
    indices = torch.randperm(H * W * T)
    for b_idx in range(0, H*W, maxpoints):

        b_indices = indices[b_idx:min(H*W, b_idx+maxpoints)]
        b_coords = coords[:, b_indices, ...].cuda()
        b_indices = b_indices.cuda()

        pixelvalues = model(b_coords)
        with torch.no_grad(): rec[:, b_indices, :] = pixelvalues
        #print(pixelvalues.device , img.device)
        loss = MSE_loss(pixelvalues[0], img[b_indices, :])

        optim.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

    time_array[epoch] = time.time() - init_time

    with torch.no_grad():
        mse_array[epoch] = ((gt - rec) ** 2).mean().item()

        #im_gt = gt.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
        #im_rec = rec.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
        #images_gt = gt.reshape(T, H, W, 3).permute(0, 3, 1, 2)  # (T, 3, H, W)
        #images_rec = rec.reshape(T, H, W, 3).permute(0, 3, 1, 2)
        #consts.append(model.prior.consts.cpu().detach().numpy())

        gt_frames = gt.reshape(T, H, W, 3)
        rec_frames = rec.reshape(T, H, W, 3)

        t = 0  # or any valid time index

        im_gt  = gt_frames[t].squeeze(0)   # (1, 3, H, W)
        im_rec = rec_frames[t].squeeze(0)  # (1, 3, H, W)

        psnrval = -10 * torch.log10(mse_array[epoch])
        psnr_vals.append(psnrval.cpu().numpy())

        tbar.set_description(f"Loss: {loss.item():.6f} "
                     f"| PSNR: {psnrval:.2f}")
        tbar.refresh()

    scheduler.step()
    imrec = im_rec.detach().cpu().numpy()

    if (mse_array[epoch] < best_mse) or (epoch == 0):
        best_mse = mse_array[epoch]
        best_img = imrec
        best_rec = rec_frames.detach().cpu().numpy()
        best_epoch = epoch

    tot_iters+=1

MSE_ARRAY = mse_array.detach().cpu().numpy()
PSNR_ARRAY = psnr_vals

# Create figure and subplots
plt.figure(figsize=(12, 5))

# --- MSE subplot ---
plt.subplot(1, 2, 1)
plt.plot(MSE_ARRAY, marker='o', linestyle='-', color='blue')
plt.title("MSE Plot")
plt.xlabel("Epoch")
plt.ylabel("MSE Value")
plt.grid(True)

# --- PSNR subplot ---
plt.subplot(1, 2, 2)
plt.plot(PSNR_ARRAY, marker='s', linestyle='-', color='green')
plt.title("PSNR Plot")
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.grid(True)

# Save plot
save_file = os.path.join(TRAIN_GRAPH_DIR, f'iters_{tot_iters}_psnr_{max(psnr_vals):.2f}_mse_psnr_plot.png')
plt.tight_layout()
plt.savefig(save_file)
plt.close()
# print(f"Plot saved to: {save_file}")

# Show plot
#plt.show()

weight_path = os.path.join(TRAIN_WEIGHTS_DIR, f"Single_Leaf_RGB_iters_{tot_iters}_psnr_{max(psnr_vals):.2f}.pth")
torch.save({'state_dict': model.state_dict(),
            'best_epoch': best_epoch,
            'gt': gt,
            'rec': best_rec,
            #'consts_array': np.array(consts),
            #'time_array': time_array.detach().cpu().numpy(),
            'mse_array': mse_array.detach().cpu().numpy(),
            'psnr_vals': np.array(psnr_vals)
            }, weight_path)



# Inference and save results
model.eval()
H = 256
W = 256
T = 100

x = np.linspace(-1, 1, W)
y = np.linspace(-1, 1, H)
t = np.linspace(-1, 1, T)

# Create spatial grid (H*W, 2)
Y, X = np.meshgrid(y, x, indexing='ij')
xy = np.stack([X.ravel(), Y.ravel()], axis=-1)   # shape: (H*W, 2)

# Repeat spatial grid for each time step (T times)
xy_repeated = np.tile(xy, (T, 1))                # shape: (H*W*T, 2)

# Create time column (time varies slowest)
t_column = np.repeat(t, H * W)[:, None]          # shape: (H*W*T, 1)

# Final coordinate tensor
test_coords = np.concatenate([xy_repeated, t_column], axis=-1)

# Convert to torch
test_coords = torch.from_numpy(test_coords).float()
#print(test_coords.shape, test_coords.device)

test_coords = test_coords.cuda()

TRAIN_INFERECE_DIR = TRAIN_INFERECE_DIR + "//" +f"{weight_path.split('/')[-1][0:-4]}"
os.makedirs(TRAIN_INFERECE_DIR, exist_ok=True)

for i in range(100):
    index = i
    #print( test_coords[256*256 * (index):256*256 * (index)+5,:] )
    test_pixel_info = model(test_coords[256*256 * (index):256*256 * (index+1),:])
    #print(test_pixel_info.shape)

    test_pixel_info = test_pixel_info.detach().cpu().numpy()
    test_pixel_info = test_pixel_info.reshape((H,W,3))
    test_pixel_info = np.clip(test_pixel_info, 0, 1)
    #print(test_pixel_info.shape)

    plt.imsave( TRAIN_INFERECE_DIR + "//" + f"{weight_path.split('/')[-1][0:-4]}_index_{index}_t{round(t[index],3)}.png", test_pixel_info)