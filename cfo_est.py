import os
import numpy as np
import scipy.io as sio
import pickle

# ===== 配置参数 =====
data_num = 2000
cyc_period = 10
sts_length = 16  # ✅ 关键修改：每个 STS 是 16 个点
rxnode_name = "2025_12_25_wifi_IQ"
base_path = r'C:\Users\arise\Desktop\wifi_audio'

# 带宽目录（现在 sts_length 固定为 16，但采样率可能不同）
bandwidth_dirs = {
    '20MHz': 'Config_wifi_20M',
    '10MHz': 'Config_wifi_10M',
    '5MHz': 'Config_wifi_5M'
}

# 如果你知道各带宽对应的采样率，填在这里（单位：Hz）
# 如果不确定，可设为 None，只保存归一化 CFO
sampling_rates = {
    '20MHz': 20e6,  # 20 MHz
    '10MHz': 20e6,  # 10 MHz
    '5MHz': 20e6  # 5 MHz
}

device_list = [
    'hackrf_5453',
    'hackrf_5c63',
    'hackrf_7353',
    'hackrf_70cf',
    'hackrf_9583',
    'hackrf_8783'
]

def compensate_cfo(iq_signal, cfo_hz: float, fs: float):
    """
    对 IQ 信号进行 CFO 补偿，保持输入输出 shape 一致。
    支持 (N,), (N,1), (1,N) 等形状。
    """
    # 保存原始 shape
    original_shape = iq_signal.shape

    # 转为 1D 进行计算
    x = np.asarray(iq_signal).flatten()  # (160,1) → (160,)
    n = np.arange(x.size)
    correction = np.exp(-1j * 2 * np.pi * cfo_hz / fs * n)
    corrected_1d = x * correction

    # 恢复原始 shape
    return corrected_1d.reshape(original_shape)

# ===== CFO 估计函数（使用 sts_length=16）=====
def estimate_cfo_multi_sts(iq_signal, cyc_period, sts_length):
    total_angle = 0.0
    count = 0
    for k in range(cyc_period - 1):
        start1 = k * sts_length
        end1 = (k + 1) * sts_length
        start2 = (k + 1) * sts_length
        end2 = (k + 2) * sts_length

        if end2 > len(iq_signal):
            break

        s1 = iq_signal[start1:end1]
        s2 = iq_signal[start2:end2]
        P = np.sum(s2 * np.conj(s1))  # ✅ 交换顺序：s2 * conj(s1)
        total_angle += np.angle(P)
        count += 1

    if count == 0:
        return np.nan
    cfo_norm = (total_angle / count) / (2 * np.pi * sts_length)
    return cfo_norm


if __name__ == "__main__":
    # ===== 主程序 =====
    all_results = {}  # {bandwidth: {device: [cfo_hz or cfo_norm]}}

    for bw_name, config_dir in bandwidth_dirs.items():
        print(f"\nProcessing {bw_name}...")
        all_results[bw_name] = {}

        config_path = os.path.join(base_path, config_dir, rxnode_name)
        fs = sampling_rates.get(bw_name, None)  # 可能为 None

        for device in device_list:
            device_path = os.path.join(config_path, device)
            if not os.path.exists(device_path):
                print(f"  ⚠️  Device folder not found: {device}")
                continue

            cfo_list = []
            mat_files = sorted([f for f in os.listdir(device_path) if f.endswith('.mat')])
            mat_files = mat_files[:data_num]

            print(f"  Processing {device} ({len(mat_files)} files)...")

            for mat_file in mat_files:
                try:
                    mat_path = os.path.join(device_path, mat_file)
                    data = sio.loadmat(mat_path)

                    # 尝试提取 IQ 信号
                    iq_signal = None
                    for key in ['iq_signal', 'data', 'signal', 'x']:
                        if key in data:
                            iq_signal = data[key].flatten()
                            break
                    if iq_signal is None:
                        for key, val in data.items():
                            if not key.startswith('__') and isinstance(val, np.ndarray):
                                iq_signal = val.flatten()
                                break



                    # 估计归一化 CFO
                    cfo_norm = estimate_cfo_multi_sts(iq_signal, cyc_period, sts_length)

                    if np.isnan(cfo_norm):
                        cfo_list.append(np.nan)
                    else:
                        # 如果知道采样率，转为 Hz；否则保留归一化值
                        if fs is not None:
                            cfo_hz = cfo_norm * fs
                            cfo_list.append(cfo_hz)
                        else:
                            cfo_list.append(cfo_norm)  # 归一化值

                except Exception as e:
                    print(f"    ❌ Error in {mat_file}: {e}")
                    cfo_list.append(np.nan)

            all_results[bw_name][device] = np.array(cfo_list)

    # ===== 保存结果 =====
    save_dir = os.path.join(os.getcwd(), 'CFO_Results_STS16')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'cfo_estimates_sts16.pkl'), 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\n✅ Done! Results saved to: {save_dir}")

    # ===== 打印统计（如果是 Hz）=====
    print("\n" + "=" * 60)
    for bw, devices in all_results.items():
        unit = "Hz" if sampling_rates.get(bw) is not None else "cycles/sample"
        print(f"\n{bw} CFO Statistics ({unit}):")
        for dev, cfo_arr in devices.items():
            valid = cfo_arr[~np.isnan(cfo_arr)]
            if len(valid) > 0:
                print(f"  {dev}: mean={np.mean(valid):.4f}, std={np.std(valid):.4f}")
            else:
                print(f"  {dev}: no valid data")