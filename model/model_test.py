
import sys
import numpy as np
import torch
import torch.nn as nn
import os 
import h5py
import yaml
from torch.utils.data import DataLoader, random_split
from einops import rearrange
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F
import statistics
import random
import pandas as pd
from torch.utils.data import Dataset
import math
from scipy.stats import gaussian_kde
import io
from torch.utils.data import Dataset, DataLoader, Sampler
import random


def feature_normalizing(input_array):
    """input should be in shape [N, ...]
    """
    # max_vals = input_tensor.abs().max(dim=0, keepdim=True).values
    max_vals = np.max(np.abs(input_array), axis=0, keepdims=True)
    return input_array / (max_vals+ 1e-8)

def feature_denormlizing(normalized_tensor, max_vals):
    return normalized_tensor * max_vals

def normalize_csi_l2(csi_array):
    """Normalize each CSI vector to unit norm.
    Input shape: [N, 240]
    """
    norm = norm = np.linalg.norm(csi_array, axis=1, keepdims=True)  # [N, 1]
    return csi_array / (norm + 1e-8)

def reshape_and_interpolate_csi(csi_data: np.ndarray, rx_positions: np.ndarray,tx_orientation: np.ndarray, step_size_cm: float = 1.0):
    """
    Reshapes CSI data and generates interpolated Rx positions.

    Parameters:
    - csi_data: np.ndarray of shape (N, 500, 32, 33)
    - rx_positions: np.ndarray of shape (N, 3)
    - step_size_cm: step size in centimeters between interpolated points (default = 1.0 cm)

    Returns:
    - csi_final: np.ndarray of shape (N*100, 5, 32, 33)
    - rx_final: np.ndarray of shape (N*100, 3)
    """
    N = csi_data.shape[0]
    csi_reshaped = csi_data.reshape(N, 100, 5, 64, 33)
    csi_final = csi_reshaped.reshape(-1, 5, 64, 33)
    rx_final = []
    tx_or_final=[]

    for i in range(N):
        current_pos = rx_positions[i]
        if i < N - 1:
            next_pos = rx_positions[i + 1]
            direction = next_pos - current_pos
            norm = np.linalg.norm(direction)
            if norm == 0:
                direction_unit = np.zeros_like(direction)
            else:
                direction_unit = direction / norm
        else:
            direction_unit = np.zeros_like(current_pos)
        interpolated = current_pos + np.arange(100).reshape(-1, 1) * (step_size_cm / 100.0) * direction_unit
        rx_final.append(interpolated)
        tx_orientation_repeated = np.tile(tx_orientation[i], (100, 1))
        tx_or_final.append(tx_orientation_repeated)
    rx_final = np.vstack(rx_final)
    tx_or_final = np.vstack(tx_or_final)
    return csi_final, rx_final, tx_or_final

  
def save_train_eval_plot(epoch_train_loss, epoch_val_loss, mean_val_error_m_list,train_error_m_list, save_path='train_eval_plot.png'):
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_train_loss, label='Training Loss')
    plt.plot(epoch_val_loss, label='Validation Loss')
    plt.plot(train_error_m_list,label='Train Distance Error Metric in cm')
    plt.plot(mean_val_error_m_list, label='Val Distance Error Metric in cm')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Error')
    plt.title('Training and Validation Loss over Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_pdf_cdf_plots(errors, bins=100, pdf_path="/home/parmardee/deepak/Mtech_project/new_task_1/dynamic_scene_model2/model2/model2_results/pdf_trial5_cov.png", cdf_path="/home/parmardee/deepak/Mtech_project/new_task_1/dynamic_scene_model2/model2/model2_results/cdf_trial5_cov.png"):

    # --- PDF ---
    kde = gaussian_kde(errors)
    x_vals = np.linspace(0, errors.max(), 500)
    pdf_vals = kde(x_vals)

    plt.figure()
    plt.plot(x_vals, pdf_vals, color='blue')
    plt.title('PDF of Localization Error')
    plt.xlabel('Error (cm)')
    plt.ylabel('Density')
    plt.xlim(right=6)             
    plt.grid(True)  
    plt.tight_layout()
    plt.savefig(pdf_path, format='png', bbox_inches='tight')
    plt.close()

    # --- CDF ---
    sorted_errors = np.sort(errors)
    cdf_vals = np.arange(1, len(errors) + 1) / len(errors)

    plt.figure()
    plt.plot(sorted_errors, cdf_vals, color='green')
    plt.title('CDF of Localization Error')
    plt.xlabel('Error (meters)')
    plt.ylabel('Cumulative Probability')
    plt.xlim(right=6)             
    plt.grid(True)  
    plt.tight_layout()
    plt.savefig(cdf_path, format='png', bbox_inches='tight')
    plt.close()



config_file_path="/home/parmardee/deepak/Mtech_project/new_task_1/dynamic_scene_model2/model2/model2_testset.yml"
with open(config_file_path) as f:
    kwargs = yaml.safe_load(f)


base_path=kwargs['path']['dataset_path']
raw_data_path= os.path.join(base_path, 'raw_data')
file_names = kwargs['path']['file_names']
# model_save=kwargs['path']['model_save']
# model_save_path=os.path.join(model_save, 'best_model.pth')
processed_folder = os.path.join(base_path, 'processed_data')
os.makedirs(processed_folder, exist_ok=True)
result_path = kwargs['path']['results_path']

terminal_stdout = sys.stdout
sys.stdout = open('/home/parmardee/deepak/Mtech_project/new_task_1/dynamic_scene_model2/dataset/processed_test_data/test_data_result/model_testdata.txt', 'w')

print("Config file loaded process start", file=terminal_stdout)


index = []
sample_id = 0
for fff, fname in enumerate(file_names):

    save_path = os.path.join(processed_folder, f'sample_0_input.npy')
    if os.path.exists(save_path):
        print("File exists. Skipping...")
    else:
        full_path = os.path.join(raw_data_path, f'{fname}')
        with h5py.File(full_path, 'r') as f:
            csi_data = f['csidata'][:]
            tx_or = f['tx_orientation'][:]
            rx_pos = f['rx_positions'][:]
            csi_final, rx_final, tx_or_final = reshape_and_interpolate_csi(csi_data, rx_pos, tx_or)
            
            for i in range(len(csi_final)):
                sample_input = {
                    'csidata': csi_final[i],
                    'tx_orientation': tx_or_final[i]
                }
                label = rx_final[i]

                input_path = os.path.join(processed_folder, f'sample_{sample_id}_input.npy')
                label_path =  os.path.join(processed_folder, f'sample_{sample_id}_label.npy')

                np.save(input_path, sample_input)
                np.save(label_path, label)

                index.append((input_path, label_path))
                sample_id += 1

class ContiguousRandomBatchSampler(Sampler):
    def __init__(self, dataset_size, batch_size, drop_last=False):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.num_batches = dataset_size // batch_size
        if not drop_last and dataset_size % batch_size != 0:
            self.num_batches += 1

    def __iter__(self):
        # Create list of starting indices for each batch
        starts = list(range(0, self.dataset_size, self.batch_size))
        # Shuffle the batch order
        random.shuffle(starts)

        for start in starts:
            end = min(start + self.batch_size, self.dataset_size)
            if end - start == self.batch_size or not self.drop_last:
                yield list(range(start, end))

    def __len__(self):
        return self.num_batches
    
class PairedNPYDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.input_files = sorted([f for f in os.listdir(folder_path) if f.endswith('_input.npy')])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file = self.input_files[idx]
        label_file = input_file.replace('_input.npy', '_label.npy')

        input_path = os.path.join(self.folder_path, input_file)
        label_path = os.path.join(self.folder_path, label_file)

        x = np.load(input_path, allow_pickle=True).item()
        csidata = x['csidata']
        csidata= normalize_csi_l2(csidata)
        tx_orientation = x['tx_orientation']
        tx_orientation=feature_normalizing(tx_orientation)
        label = np.load(label_path)

        return csidata,tx_orientation,label


torch.manual_seed(42)

dataset = PairedNPYDataset(processed_folder)
total_size = len(dataset)
train_size = int(0.7 * total_size)
test_size = total_size - train_size

# Split into train, val, test datasets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(
    train_dataset,
    batch_sampler=ContiguousRandomBatchSampler(len(train_dataset), 32),
    num_workers=8
)
test_loader = DataLoader(
    test_dataset,
    batch_sampler=ContiguousRandomBatchSampler(len(test_dataset), 32),
    num_workers=8
)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
print(f"Test data: {len(dataset)}",flush=True)

#-----------forier encoding
class FourierEncoding(nn.Module):
    def __init__(self, input_dims, num_frequencies):
        super(FourierEncoding, self).__init__()
        self.input_dims = input_dims
        self.num_frequencies = num_frequencies

        # Create frequency bands: [1, 2, ..., 2^(L-1)] * pi
        self.freq_bands = 2.0 ** torch.arange(num_frequencies) * math.pi  # shape: (L,)

    def forward(self, x):
        """ x: Tensor of shape (..., input_dims)
        Returns: Tensor of shape (..., input_dims * 2 * num_frequencies)
        """
        # x shape: (batch_size, input_dims)
        x = x.unsqueeze(-1)  # (batch_size, input_dims, 1)
        angles = x * self.freq_bands.to(x.device)  # (batch_size, input_dims, L)

        # Apply sin and cos to each frequency
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)

        # Concatenate along the frequency dimension
        encoding = torch.cat([sin_enc, cos_enc], dim=-1)  # (batch_size, input_dims, 2L)
        return encoding.view(*x.shape[:-2], -1)  # Flatten to (batch_size, input_dims * 2L)

class DelayDopplerTransform(nn.Module):
    def __init__(self, num_symbols=5, num_subcarriers=33):
        super(DelayDopplerTransform, self).__init__()
        self.num_symbols = num_symbols
        self.num_subcarriers = num_subcarriers

        # IFFT for delay (W subcarriers)
        F_delay = np.fft.ifft(np.eye(num_subcarriers)) * np.sqrt(num_subcarriers)
        self.W_delay_real = nn.Parameter(torch.tensor(F_delay.real, dtype=torch.float32), requires_grad=False)
        self.W_delay_imag = nn.Parameter(torch.tensor(F_delay.imag, dtype=torch.float32), requires_grad=False)

        # FFT for Doppler (T symbols)
        F_dopp = np.fft.fft(np.eye(num_symbols)) / np.sqrt(num_symbols)
        self.W_dopp_real = nn.Parameter(torch.tensor(F_dopp.real, dtype=torch.float32), requires_grad=False)
        self.W_dopp_imag = nn.Parameter(torch.tensor(F_dopp.imag, dtype=torch.float32), requires_grad=False)

    def forward(self, x_complex):
        """
        Input: x_complex [N, T, TX_Y, TX_X, W] (complex tensor)
        Output: [N, V, TX_Y, TX_X, D] (Delay-Doppler map)
        """
        x = x_complex.view(x_complex.size(0), x_complex.size(1), 8, 8, 33) # shape: [T, TY, TX, W]

        # Delay transform (IFFT along last dim)
        W_delay = torch.complex(self.W_delay_real, self.W_delay_imag).to(x.device)
        x = torch.matmul(x, W_delay)  # shape: [T, TY, TX, D]

        # Doppler transform (FFT along first dim)
        x = x.permute(0, 2, 3, 4, 1)  # [TY, TX, D, T]
        W_dopp = torch.complex(self.W_dopp_real, self.W_dopp_imag).to(x.device)
        x = torch.matmul(x, W_dopp)  # [TY, TX, D, V]
        x = x.permute(0, 4, 1, 2, 3)  # [V, TY, TX, D]

        return x

class SymbolSubcarrierAutocorrelation(nn.Module):
    def __init__(self, num_symbols=5, num_subcarriers=33):
        super(SymbolSubcarrierAutocorrelation, self).__init__()
        self.num_symbols = num_symbols
        self.num_subcarriers = num_subcarriers

        def get_dft_matrix(N):
            omega = np.exp(-2j * np.pi / N)
            j, k = np.meshgrid(np.arange(N), np.arange(N))
            F = np.power(omega, j * k) / np.sqrt(N)
            return F

        # DFT for symbols (dim=1) and subcarriers (dim=4)
        F_sym = get_dft_matrix(num_symbols)
        F_sub = get_dft_matrix(num_subcarriers)

        self.F_sym_real = nn.Parameter(torch.tensor(F_sym.real, dtype=torch.float32), requires_grad=False)
        self.F_sym_imag = nn.Parameter(torch.tensor(F_sym.imag, dtype=torch.float32), requires_grad=False)
        self.F_sub_real = nn.Parameter(torch.tensor(F_sub.real, dtype=torch.float32), requires_grad=False)
        self.F_sub_imag = nn.Parameter(torch.tensor(F_sub.imag, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        # x: [N, S=5, TXH=8, TXW=4, W=33]

        N, S, TXH, TXW, W = x.shape
        x = x.view(N, S, TXH * TXW, W)  # [N, 5, 32, 33]

        # FFT matrices
        F_sym = torch.complex(self.F_sym_real, self.F_sym_imag).to(x.device)    # [5, 5]
        F_sub = torch.complex(self.F_sub_real, self.F_sub_imag).to(x.device)    # [33, 33]

        x = x.permute(0, 2, 1, 3)  # [N, 32, 5, 33]
        X = torch.matmul(F_sym, x)                # FFT along symbols: [N, 32, 5, 33]
        X = torch.matmul(X, F_sub)                # FFT along subcarriers: [N, 32, 5, 33]

        X_mag_sq = torch.abs(X) ** 2
        X = torch.matmul(F_sym.conj().T, X_mag_sq.to(torch.cfloat))   # IFFT along symbols
        X = torch.matmul(X, F_sub.conj().T)                           # IFFT along subcarriers

        X = X.permute(0, 2, 1, 3).contiguous()  # [N, 5, 32, 33]
        X = X.view(N, S, TXH, TXW, W)
        return X  


class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim=64*33*2, hidden_dim=256):
        super().__init__()
        # input_dim = int(input_dim) if not isinstance(input_dim, int) else input_dim
        # hidden_dim = int(hidden_dim) if not isinstance(hidden_dim, int) else hidden_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):
        """
        x: [B, T=5, 64*33*2]
        Goal: For each t, flatten spatial, feed to LSTM, take output at t=5
        """
        lstm_out, _ = self.lstm(x)  # lstm_out: [B, T, hidden_dim]
        last_output = lstm_out[:, -1, :]
        return last_output

class GatedWeightedFusion(nn.Module):
    def __init__(self, in_dims, out_dim=None):
        super(GatedWeightedFusion, self).__init__()
        self.num_inputs = len(in_dims)
        self.out_dim = out_dim if out_dim is not None else in_dims[0]

        self.projections = nn.ModuleList([
            nn.Linear(in_dim, self.out_dim) for in_dim in in_dims
        ])
        self.gate_layer = nn.Sequential(
            nn.Linear(sum(in_dims), self.num_inputs),
            nn.Softmax(dim=1) 
        )

    def forward(self, *inputs):
        projected_inputs = [proj(x) for proj, x in zip(self.projections, inputs)]
        x_cat = torch.cat(inputs, dim=1) 
        gate_weights = self.gate_layer(x_cat)  
        fused = sum(w.unsqueeze(1) * x for w, x in zip(gate_weights.T, projected_inputs))
        return fused



class CNN_LSTM_Net(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.no_subcarrier = kwargs['3gpp_specs']['no_subcarrier'] 
        self.no_symbols = kwargs['3gpp_specs']['no_symbols']
        self.no_anteena =  kwargs['3gpp_specs']['no_anteena'] 
        self.no_frequencies =  kwargs['encoding']['num_frequencies'] 
        
        self.FT_embedding = FourierEncoding(3, self.no_frequencies)
        self.delay_transform = DelayDopplerTransform(self.no_symbols, self.no_subcarrier)
        self.auto_corr = SymbolSubcarrierAutocorrelation(self.no_symbols, self.no_subcarrier)
        
        self.conv1 = nn.Conv2d(in_channels=2*self.no_subcarrier*self.no_symbols, out_channels=160, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(num_features=160)
        self.conv2 = nn.Conv2d(in_channels=160, out_channels=40, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(num_features=40)
        self.conv3 = nn.Conv2d(in_channels=40, out_channels=10, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(num_features=10)
        self.conv4 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=20)
        self.conv5 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=30)
        self.conv6 = nn.Conv2d(in_channels=30, out_channels=36, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=36)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        self.LSTMFeatureExtractor=LSTMFeatureExtractor(input_dim=64*33*2, hidden_dim=256)

        self.feature_fusion_layer=GatedWeightedFusion(in_dims=[36*64, 256], out_dim=100)

        self.fusion_layer = GatedWeightedFusion(in_dims=[100, 60], out_dim=100)

        self.fc =  nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 9)
        ) 

    def forward(self, x1, x2):   
    
        H_delay = self.delay_transform(x1)
        x_feat1 = self.auto_corr(H_delay)  # shape: [B, features]
        #  Reshape for CNN (assume square spatial shape: [B, 330,8,8]])
        x_feat1_real = x_feat1.real          # Shape: [B, 165, 8, 8]  
        x_feat1_imag = x_feat1.imag
        x_feat1_real_imag = torch.cat([x_feat1_real, x_feat1_imag], dim=1)
        x_feat1 = x_feat1_real_imag.view(x_feat1_real_imag.shape[0], 2*self.no_subcarrier*self.no_symbols, 8 , 8)  # [B, 330,8,4]

        x1_fisrt = self.relu(self.bn1(self.conv1(x_feat1)))
        x1_fisrt = self.dropout(self.relu(self.bn2(self.conv2(x1_fisrt))))
        x1_fisrt = self.relu(self.bn3(self.conv3(x1_fisrt)))
        x1_fisrt = self.dropout(self.relu(self.bn4(self.conv4(x1_fisrt))))
        x1_fisrt = self.relu(self.bn5(self.conv5(x1_fisrt)))
        x1_fisrt = self.dropout(self.relu(self.bn6(self.conv6(x1_fisrt))))
        x1_fisrt = x1_fisrt.view(x1_fisrt.size(0), -1)
        
        # LSTM
        x1_real = x1.real     
        x1_imag = x1.imag 
        x1_combined = torch.cat([x1_real, x1_imag], dim=-1)
        x1_flat = x1_combined.reshape(x1_combined.size(0), 5, -1)
        x1_second=self.LSTMFeatureExtractor(x1_flat)
        
        x1_fused=self.feature_fusion_layer(x1_fisrt, x1_second)
        
        x2=self.FT_embedding(x2)

        x_final = self.fusion_layer(x1_fused, x2)
        x = self.fc(x_final)
        return x


def custom_loss(mu, Sigma, target, 
                         alpha = 1, beta = 0.2, 
                         weight_1=0.1, weight_2=1):
    """
    Combines uncertainty-aware Mahalanobis loss and L2 + cosine loss.
    Args:
        mu: Predicted coordinates (B, 3)
        Sigma: Covariance matrices (B, 3, 3)
        target: Ground-truth coordinates (B, 3)
        alpha: Weight for Mahalanobis distance
        beta: Weight for log-determinant
        weight_1: Weight for uncertainty-aware loss
        weight_2: Weight for L2+Cosine loss
    Returns:
        Scalar combined loss
    """
    B = mu.shape[0]
    # ---------- Loss 1: Mahalanobis + log-determinant ----------
    diff = (target - mu).unsqueeze(-1)  # (B, 3, 1)
    inv_Sigma = torch.linalg.inv(Sigma + 1e-6 * torch.eye(3, device=Sigma.device))
    mahalanobis = torch.matmul(diff.transpose(1, 2), torch.matmul(inv_Sigma, diff)).squeeze()
    log_det = torch.logdet(Sigma + 1e-3 * torch.eye(3, device=Sigma.device))
    log_det = torch.clamp(log_det, min=-5, max=5)
    loss1 = alpha * mahalanobis + beta * log_det
    loss1 = loss1.mean()
    # ---------- Loss 2: L2 + Cosine Similarity ----------
    cos = nn.CosineSimilarity(dim=1)
    l2 = F.mse_loss(mu, target)  # (B,) → mean
    cosine = cos(mu, target).mean()
    cosine_loss = 1 - cosine
    loss2 = 0.5 * l2 + 0.5 * cosine_loss

    # ---------- Final Combined Loss ----------
    final_loss = weight_1 * loss1 + weight_2 * loss2
    return final_loss

def count_95_ci_coverage(mu, cov, true_pos):
    """
    mu:       [B, 3] → predicted coordinates
    cov:      [B, 3, 3] → predicted covariance matrices
    true_pos: [B, 3] → ground truth coordinates
    
    Returns:
        count_in_ci: number of samples where true_pos ∈ 95% CI
        percentage: (count / total) * 100
    """
    std = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2))  # [B, 3]
    factor = 1.96
    lower = mu - factor * std
    upper = mu + factor * std
    # Check if each coordinate is within the CI
    in_ci = (true_pos >= lower) & (true_pos <= upper)  # [B, 3]
    # For each point, all 3 coordinates must be inside the CI
    fully_inside = in_ci.all(dim=1)  # [B]
    count_in_ci = fully_inside.sum().item()
    total = mu.shape[0]
    percentage = 100.0 * (count_in_ci / total)
    return count_in_ci, percentage

def euclidean_distance(pred, target):
    return torch.sqrt(torch.sum((pred - target) ** 2, dim=1))

def compute_r90(predictions, ground_truth):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().detach().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().detach().numpy()
    
    errors = np.linalg.norm(predictions - ground_truth, axis=1)  # [N]
    r90 = np.percentile(errors, 90)
    return r90

def compute_confidence_from_variance(cov_matrices, epsilon=1e-6):
    """
    Computes confidence scores from diagonal covariance matrices.
    Args:
        cov_matrices (torch.Tensor): Tensor of shape (B, 3, 3), where each matrix is diagonal.
        epsilon (float): Small value to avoid division by zero.
    Returns:
        torch.Tensor: Tensor of shape (B,) with confidence scores.
    """
    variances = torch.diagonal(cov_matrices, dim1=-2, dim2=-1) 
    var_sum = variances.sum() 
    confidence = 1.0 / (var_sum + epsilon)
    
    return confidence

lr = float(kwargs['model']['lr'])
lr_factor = float(kwargs['model']['lr_factor'])
lr_patience = float(kwargs['model']['lr_patience'])
lr_eps = float(kwargs['model']['lr_eps'] )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CNN_LSTM_Net(**kwargs).to(device)

model.load_state_dict(torch.load("/home/parmardee/deepak/Mtech_project/new_task_1/dynamic_scene_model2/model2/best_model_trial4_cov_m.pth"))

for param in model.parameters():
    param.requires_grad = False

cnn_layers = [
    # model.conv1, model.bn1,
    # model.conv2, model.bn2,
    # model.conv3, model.bn3,
    # model.conv4, model.bn4,
    model.conv5, model.bn5,
    model.conv6, model.bn6,
]
for layer in cnn_layers:
    for param in layer.parameters():
        param.requires_grad = True

for param in model.LSTMFeatureExtractor.parameters():
    param.requires_grad = True

# for param in model.delay_transform.parameters():
#     param.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-5  # Try 1e-4 to 1e-6 and tune
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, eps=1e-6, verbose=True
)
print(f"Training start",flush=True)
print(f"Training start",file=terminal_stdout)
epochs=20
for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}",flush=True)
    model.train()
    train_loss = 0
    train_error=0
    for x1, x2, y in train_loader:
        x1, x2, y = x1.to(device), x2.to(device).float(), y.to(device).float()
        optimizer.zero_grad()

        output = model(x1, x2)
        pred_coords = output[:, :3] 
        cov_params  = output[:, 3:]        
        L = torch.zeros((32, 3, 3), device=device)
        L[:, 0, 0] = torch.exp(cov_params[:, 0])       # l00
        L[:, 1, 0] = cov_params[:, 1]                  # l10
        L[:, 1, 1] = torch.exp(cov_params[:, 2])       # l11
        L[:, 2, 0] = cov_params[:, 3]                  # l20
        L[:, 2, 1] = cov_params[:, 4]                  # l21
        L[:, 2, 2] = torch.exp(cov_params[:, 5])       # l22
        sigma = torch.matmul(L, L.transpose(1, 2))  # Covariance matrix       
        loss = custom_loss(pred_coords, sigma , y)
        error=torch.norm(pred_coords - y, dim=1)
        error=error.mean()
        loss = loss.mean() 
        loss.backward()
        optimizer.step()
        scheduler.step(error)
        train_loss += loss.item()
    print(f"Train Loss = {train_loss/len(train_loader):.4f}",flush=True)


all_preds = []
all_labels = []
all_sigma = []
model.eval()
with torch.no_grad():
    for x1, x2, y in test_loader:
        x1, x2, y = x1.to(device), x2.to(device).float(), y.to(device).float()
        output = model(x1, x2)
        pred_coords = output[:, :3] 
        cov_params  = output[:, 3:]  
        L = torch.zeros((32, 3, 3), device=device)
        L[:, 0, 0] = torch.exp(cov_params[:, 0])       # l00
        L[:, 1, 0] = cov_params[:, 1]                  # l10
        L[:, 1, 1] = torch.exp(cov_params[:, 2])       # l11
        L[:, 2, 0] = cov_params[:, 3]                  # l20
        L[:, 2, 1] = cov_params[:, 4]                  # l21
        L[:, 2, 2] = torch.exp(cov_params[:, 5])       # l22
        sigma = torch.matmul(L, L.transpose(1, 2))     # Covariance matrix
        all_sigma.append(sigma)

        all_preds.append(pred_coords)
        all_labels.append(y)

all_sigma = torch.cat(all_sigma, dim=0)
all_preds = torch.cat(all_preds, dim=0)
all_labels = torch.cat(all_labels, dim=0)  
ci_count, ci_percent = count_95_ci_coverage(all_preds, all_sigma, all_labels) 
errors_m = torch.norm(all_preds - all_labels, dim=1)
save_pdf_cdf_plots(errors_m.cpu())
mean_error_m = errors_m.mean().item()
avg_sigma=all_sigma.mean(dim=0)
conf = compute_confidence_from_variance(avg_sigma)

r90 = compute_r90(all_preds, all_labels)

print("---------Test results-------",flush=True)
print(f"Mean Test error: {mean_error_m:.4f} m",flush=True)
print(f"R90 matric: {r90:.4f} m",flush=True)
print(f"Avg standered deviation in [x,y,z] in test in m = {torch.sqrt(torch.diag((avg_sigma)))}")
print(f"Average confidance score: {conf:.4f}",flush=True)
print(f"{ci_count} out of {len(all_preds)} points are inside the 95% CI in total = ({ci_percent:.2f}%)")
print(f"Process complete results stored  ", file=terminal_stdout)
