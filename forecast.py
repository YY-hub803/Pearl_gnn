import os
import utils
import json
import numpy as np
import pandas as pd
import model
import torch
import glob

@torch.no_grad()
def rolling_forecast(
    model,
    x_init,                  # [B, N, T, nx]
    steps,                   # 需要向未来滚动多少步
    target_in_idx,           # list[int]，预测值回填到输入x的哪些特征列（长度=ny）
    device,
    A_list=None,             # 如果模型需要图输入，就传
    exog_future=None,        # [B, N, steps, nx]，可选：未来已知外生变量
    clamp_min=None,
    clamp_max=None,
):
    """
    返回:
    preds: [B, N, steps, ny]
    """
    model.eval()
    model.to(device)
    exog_future = torch.Tensor(exog_future).unsqueeze(0).to(device)
    x_tensor = torch.Tensor(x_init).to(device)
    x_win = x_tensor.unsqueeze(0).clone()                # 当前滑动窗口 [B,N,T,nx]
    pred_list = []

    for t in range(steps):
        # 1) 当前窗口预测下一时刻
        if A_list is None:
            y = model(x_win)              # [B,N,pred_len,ny]
        else:
            y = model(x_win, A_list)      # [B,N,pred_len,ny]
        y1 = y[:, :, 0, :]                # 只取下一步 [B,N,ny]

        # 可选：物理范围裁剪（如浓度非负）
        if clamp_min is not None or clamp_max is not None:
            y1 = torch.clamp(y1, min=clamp_min, max=clamp_max)

        pred_list.append(y1.unsqueeze(2))  # [B,N,1,ny]
        # 2) 组装“下一时刻输入特征” new_x: [B,N,nx]
        # 默认先用上一时刻特征复制一份
        new_x = x_win[:, :, -1, :].clone()   # [B,N,nx]
        # 如果有未来外生变量，优先用它覆盖（推荐）
        if exog_future is not None:
            new_x = exog_future[:, :, t, :].clone()
        # 把模型预测回填到指定特征列
        # target_in_idx 长度必须等于 ny
        left = new_x[:, :, 0:target_in_idx]
        right = new_x[:, :, target_in_idx:]
        new_x = torch.cat((left, y1,right), dim=2)
        # 3) 滑窗更新：丢最老一帧，拼新一帧
        x_win = torch.cat([x_win[:, :, 1:, :], new_x.unsqueeze(2)], dim=2)
    preds = torch.cat(pred_list, dim=2)   # [B,N,steps,ny]
    return preds

def scalar(array,mean,std):

    return  (array-mean)/std
def inverse_scalar(array,mean,std):
    return array*std+mean

# set GPU
if torch.cuda.is_available():
    GPUid = 0
    torch.cuda.set_device(GPUid)


hyper_params = {
    "epoch_run": 400,
    "epoch_save": 10,
    "hidden_size": 32,
    'history_len': 32,
    'pred_len':1,
    "batch_size":32,
    "num_layers" : 2,
    "drop_rate": 0.3,
    "warmup_epochs":10,
    "base_lr":1e-3,
    "BACKEND":"LSTMModel", # select model    LSTMModel
    "lossFun":'MAE'
}

MODEL_FACTORY = {
    "LSTMModel": model.LSTMModel,
}

dir_model = "%s_B%d_H%d_L%d_P%d_dr%.2f_lr%.4f" % (
    hyper_params['BACKEND'],
    hyper_params['batch_size'],
    hyper_params['hidden_size'],
    hyper_params['history_len'],
    hyper_params['pred_len'],
    hyper_params['drop_rate'],
    hyper_params['base_lr'],
)


dir_proj = "forecast"
work_path = os.getcwd()
output_dir = f"Forecast_output"
os.makedirs(output_dir, exist_ok=True)
dir_input = os.path.join(work_path, dir_proj)
dir_output = os.path.join(output_dir,dir_model)
os.makedirs(dir_output, exist_ok=True)
model_file = os.path.join('OutPut_4h',dir_model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BACKEND= hyper_params["BACKEND"]

num_sites = 24
start_date = '2025-07-12'
end_date = '2025-07-31'
full_date_range = pd.date_range(start=start_date, end=end_date, freq='4h')
date_length = len(full_date_range)


print("------------------------ exog_future path ------------------------------")
dir_x = {
    "x_pet": os.path.join(dir_input, 'input_xforce_pet.csv'),
    "x_temp": os.path.join(dir_input, 'input_xforce_temp.csv'),
    "x_vp": os.path.join(dir_input, 'input_xforce_vp.csv'),
    "x_pre": os.path.join(dir_input, 'input_xforce_prcp.csv'),
}
dir_c = {
    "c_all": os.path.join(dir_input, 'input_c_all.csv'),
}
dir_init = {
    'x_init': os.path.join(dir_input, 'x_init.npy'),
    'y_init': os.path.join(dir_input, 'y_init.npy'),
    'date_emb': os.path.join(dir_input, 'date_emb.npy'),
    'train_stats': os.path.join(dir_input, 'train_stats.json'),
}

print("------------------------ load data ------------------------------")
print('  Loading X (Forcing)...')
x = utils.load_timeseries(dir_x, num_sites, date_length)
print('  Loading C (Static Attributes)...')
c = utils.load_attribute(dir_c)
c[np.where(np.isnan(c))] = 0

x_init = np.load(dir_init['x_init'])
y_init = np.load(dir_init['y_init'])
date_init = np.load(dir_init['date_emb'])
with open(dir_init['train_stats'], "r") as f:
    train_stats = json.load(f)

sites_ID= pd.read_csv(os.path.join(dir_input,"points_info.csv"))
print("------------------------ processing data ------------------------------")
exog_mean = [np.array(train_stats['x_mean']).flatten()[[0,1,2,5]].tolist()]
exog_std = [np.array(train_stats['x_std']).flatten()[[0,1,2,5]].tolist()]
x_trans = scalar(x,exog_mean,exog_std)
c_trans = scalar(c,train_stats['c_mean'],train_stats['c_std'])
x_init_trans = scalar(x_init,train_stats['x_mean'],train_stats['x_std'])
y_init_trans = scalar(y_init,train_stats['y_mean'],train_stats['y_std'])
c_init = c_trans[:,np.newaxis,:].repeat(hyper_params['history_len'],axis=1)
c = c_trans[:,np.newaxis,:].repeat(len(full_date_range),axis=1)

date_processing = utils.Time_emb(full_date_range)
date_array = date_processing.values
date_array_expanded = np.expand_dims(date_array, axis=0)
date_emb = np.repeat(date_array_expanded, num_sites, axis=0)

# 拼接数据
x = np.concatenate((x_init_trans, c_init,date_init), axis=-1)
x_exog = np.concatenate((x_trans, c,date_emb), axis=-1)


nx = x.shape[2]
ny = y_init.shape[-1]

model = MODEL_FACTORY[BACKEND](
        nx, ny,
        hyper_params['hidden_size'],
        hyper_params['num_layers'],
        hyper_params['pred_len'],
        hyper_params['drop_rate'])


model_files = glob.glob(os.path.join(model_file, "*.pt"))
if not model_files:
    raise FileNotFoundError("未能找到训练保存的模型文件，请检查 train_G 是否成功保存。")
# 按照文件修改时间排序，获取最新的模型
latest_model_path = max(model_files, key=os.path.getmtime)
print(f">>> 加载原始模型进行插补: {latest_model_path}")
model_raw = torch.load(latest_model_path)

pred =  rolling_forecast(model_raw,x,10,3,device,exog_future=x_exog)


out = inverse_scalar(np.array(pred.squeeze(0).cpu()),train_stats['y_mean'],train_stats['y_std'])
tp_out = pd.DataFrame(out[:,:,0],index=sites_ID['P_nm'],columns=range(1,11,1)).T
do_out = pd.DataFrame(out[:,:,1],index=sites_ID['P_nm'],columns=range(1,11,1)).T
tp_out.to_csv(os.path.join(dir_output,'forecast_tp.csv'))
do_out.to_csv(os.path.join(dir_output,'forecast_do.csv'))