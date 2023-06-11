import os
import mne
import numpy as np

# 设置输入文件夹和输出文件夹路径
input_folder = 'F:\EEG数据集\兰州大学抑郁症静息态EEG\EEG_set\HC'  # 输入文件夹路径
output_folder = 'F:\EEG数据集\代码\Clustering-Fusion\EEG_pre\HC_pre'  # 输出文件夹路径

# 获取输入文件夹下的所有子文件夹
subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

# 定义频段边界（以Hz为单位）
freq_bands = {'delta': (0.5, 4),
              'theta': (4, 8),
              'alpha': (8, 13),
              'beta': (13, 30)}


for subfolder in subfolders:
    files = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.endswith('.set')]

    # 循环处理每个EEG数据文件
    for file in files:
        # 读取set格式的EEG信号
        raw = mne.io.read_raw_eeglab(file, preload=True)

        #预处理
        #没有Cz作为参考电极，没有选择90个通道，没有ICA去眼电
        raw.filter(1,40)
        raw.notch_filter(50)
        # 设置参考方式为平均参考
        raw.set_eeg_reference('average')

        # 将信号分为四个频段
        freq_band_data = {}
        for band, (fmin, fmax) in freq_bands.items():
            raw_band = raw.copy().filter(fmin, fmax)
            freq_band_data[band] = raw_band.get_data()

        # 创建输出文件夹
        output_subfolder = os.path.join(output_folder, os.path.basename(subfolder))
        os.makedirs(output_subfolder, exist_ok=True)

        # 保存每个频段的数据为单独的文件
        for band, data in freq_band_data.items():
            output_file = os.path.join(output_subfolder, f"{os.path.splitext(os.path.basename(file))[0]}_{band}.set")
            # mne.io.export.export_raw(data, output_file, fmt='auto', overwrite=True)

            # mne.io.write_raw_eeglab(output_file, data)
            
            """info = mne.create_info(ch_names=[], ch_types=128, sfreq=250)
            data = np.array(data)
            raw_data = mne.io.RawArray(data)
            raw_data.save(output_file, overwrite=True)"""

            output_file = os.path.join(output_subfolder, f"{os.path.splitext(os.path.basename(file))[0]}_{band}.npy")
            np.save(output_file, data)