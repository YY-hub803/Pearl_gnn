import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import add_self_loops


def get_training_frequency():
    # 定义合法的频率选项
    valid_freqs = {'4h', '1D'}
    default_freq = '4h'
    prompt = f"请输入训练频率 ({'/'.join(valid_freqs)}) [默认: {default_freq}]: "
    while True:
        user_input = input(prompt).strip()
        # 处理空输入（使用默认值）
        if not user_input:
            return default_freq
        # 校验输入有效性
        if user_input in valid_freqs:
            return user_input
        print(f"错误：'{user_input}' 不是合法的频率。请输入 '4h' 或 '1D'。")


def load_timeseries(dict_data, chem_site, chem_length):
    """Load data from time-series inputs"""
    data_list = []
    for path in dict_data.values():
        loaded_data = pd.read_csv(path, delimiter=",").to_numpy()
        reshaped_data = np.reshape(np.ravel(loaded_data.T), (chem_site, chem_length, 1))
        data_list.append(reshaped_data)
    return np.concatenate(data_list, axis=2)

def load_attribute(dict_data):
    """Load data from constant attributes"""
    data_list = [np.loadtxt(path, delimiter=",", skiprows=1) for path in dict_data.values()]
    return np.concatenate(data_list, axis=1)

def to_scalar(value):
    if isinstance(value, (list, np.ndarray)):
        return value[0]
    return value


def preprocess_dynamic_data(data, train_end,val_end, log_indices=None):
    """
    处理动态数据 (X, Y)
    切分 Train/Val
    标准化 (只Fit Train)
    """
    data_processed = data.copy()

    if log_indices is not None:
        for idx in log_indices:
            # 加上 epsilon 防止 log(0)
            data_processed[:, :, idx] = np.log1p(data_processed[:, :, idx])

    # 2. Split
    train_data = data_processed[:, :train_end, :]
    val_data = data_processed[:, train_end:train_end + val_end , :]
    test_data = data_processed[:, train_end + val_end:, :]

    # 计算 Mean/Std: 形状为 [1, 1, F]
    mean = np.nanmean(data_processed, axis=(0, 1), keepdims=True)
    std = np.nanstd(data_processed, axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0  # 避免除零

    train_norm = (train_data - mean) / std
    val_norm = (val_data - mean) / std
    test_norm = (test_data - mean) / std
    return train_norm, val_norm,test_norm, mean, std

def preprocess_static_data(data, num_time_steps, log_indices=None):

    data_processed = data.copy()

    if log_indices is not None:
        for idx in log_indices:
            data_processed[:, idx] = np.log1p(data_processed[:, idx])

    # 2. Fit Standard Scaler (Global: across sites)
    # [1, F]
    mean = np.nanmean(data_processed, axis=0, keepdims=True)
    std = np.nanstd(data_processed, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    # 3. Transform
    data_norm = (data_processed - mean) / std
    data_norm = np.nan_to_num(data_norm, nan=0.0)

    # 4. Expand & Repeat [N, F] -> [N, T, F]
    c_tensor = torch.from_numpy(data_norm).float()  # [N, F]
    c_expanded = c_tensor.unsqueeze(1)  # [N, 1, F]
    c_long = c_expanded.repeat(1, num_time_steps, 1)  # [N, T, F]

    return c_long.numpy(),mean,std

def Time_emb(full_date_range):

    date_processing = pd.DataFrame(index=full_date_range)

    day_of_year = date_processing.index.dayofyear
    days_in_year = np.where(date_processing.index.is_leap_year, 366, 365)

    date_processing['sin_doy'] = np.sin(2 * np.pi * day_of_year / days_in_year)
    date_processing['cos_doy'] = np.cos(2 * np.pi * day_of_year / days_in_year)


    day_of_week = date_processing.index.dayofweek
    date_processing['sin_dow'] = np.sin(2 * np.pi * day_of_week / 7)
    date_processing['cos_dow'] = np.cos(2 * np.pi * day_of_week / 7)

    return date_processing



def edge_extract(path,num_sites):
    edges_info = pd.read_csv(path)

    edges_index = torch.tensor(edges_info.iloc[:,0:2].to_numpy(),dtype=torch.long).T
    # 添加self_loop
    # 自环边的权重1
    edge_idx,_ = add_self_loops(edges_index,num_nodes=num_sites)

    return edge_idx


def get_windows(X,Y, window_size, pred_len):
    T = X.shape[1]
    xs, ys = [], []
    # 按站点分组，严格在站点内部滑动窗口
    for t in range(T-window_size-pred_len+1):
        # 确保时间排序
        x = X[:,t:t+window_size,:]
        y = Y[:,t+window_size : t+window_size+pred_len,:]
        xs.append(x)
        ys.append(y)

    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

def build_list(Lag_matrix, max_lag):
    lag_matrix = torch.tensor(Lag_matrix)
    A_list = []
    for k in range(max_lag + 1):
        # 转置：Target行，Source列
        A_k = (lag_matrix == k).float().t()
        if k == 0:
            A_k.fill_diagonal_(1.0)
        A_list.append(A_k)
    # [max_lag + 1, N, N]
    return torch.stack(A_list)

class SpatioTemporalDataset(Dataset):
    """
    步骤 2: 动态数据集。在获取每个 Batch 时，实时进行滑窗截取、Mask 生成和 NaN 填充。
    """
    def __init__(self, X, Y):
        # 将原始完整序列转为 Tensor 以加速运算，形状保持为 [N, T, F]
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x,y


def prepare_dataloader(X, Y, batch_size,shuffle=True):
    """
    步骤 3: 封装生成 DataLoader
    """
    dataset = SpatioTemporalDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return loader
