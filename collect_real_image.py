# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:28:39 2023

@author: wcfda
"""
import os
import pickle
import numpy as np
import pandas as pd
from pandas.core.series import Series
pd.options.mode.chained_assignment = None
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def collect_WT_data(WT_idx, start_time, end_time, features):
    
    if start_time >= end_time: 
        raise ValueError('The end time must be after the start time')
    
    WT_NO = (WT_idx - 176)%25 + 1
    WT_batch =  (WT_idx - 176)//25 + 10
    y = start_time.year
    m = start_time.month
    t = datetime(y, m, 1, 0)
    
    while t <= end_time:
        data_fp = r'C:/Users/wcfda/Desktop/Study/Projects/ViT_WTPM/codes/data/'\
            '{:02d}/t_wtdata_realdata_{:04d}{:02d}.txt'.format(WT_batch, y, m)
        WT_data = pd.read_csv(data_fp, index_col = 0, sep = '\t')
        WTNO_data = WT_data['WTNO']
        target_WT_data = WT_data[WTNO_data == WT_NO]
        target_feat_data = target_WT_data[['DateAcqTime'] + features]
        if y == start_time.year and m == start_time.month:
            data = target_feat_data.copy()
        else:
            data = pd.concat([data, target_feat_data], ignore_index = True)
        print('batch:{:02d}, WTNO:{:02d}, month:{:04d}{:02d}'.format(WT_batch, WT_NO, y, m))

        m += 1
        if m > 12: y += 1; m = 1
        t = datetime(y, m, 1, 0)
    
    data.dropna(inplace = True)
    date_str_data = data['DateAcqTime']
    date_obj_data = []
    date_format = '%Y/%m/%d %H:%M:%S'
    for date_str in date_str_data:
        date_obj = datetime.strptime(date_str, date_format)
        date_obj_data.append(date_obj)
    data['DateAcqTime'] = date_obj_data
    target_data = data[(data['DateAcqTime'] >= start_time) & 
                       (data['DateAcqTime'] <= end_time)]
    
    return target_data

def sigmoid_f(x):
    return 1/(np.exp(-x) + 1)

def get_outlier_factor(feat_data, LB, UB):
    if type(feat_data) == Series:
        factors = np.zeros_like(feat_data)
        factors[feat_data < LB] = (LB - feat_data[feat_data < LB])/(UB - LB)
        factors[feat_data > UB] = (feat_data[feat_data > UB] - UB)/(UB - LB)
        outlier_factors = sigmoid_f(factors**2)
    else:
        raise TypeError('(feat_data) should be a Series object')
    
    return outlier_factors

def round_nonzero(value, digits = 0):
    if value >= 1 or value <= -1:
        power = 0
        rounded_base = round(value, digits)
    elif value == 0:
        power = -2
        rounded_base = 0
    else:
        power = int(np.floor(np.log10(abs(value))))
        amplified_value = value/10**power
        rounded_base = round(amplified_value, digits)
    
    return rounded_base, power

def translate_names(name_list):
    ENG_names = list(name_correspondence.keys())
    CHN_names = list(name_correspondence.values())
    translated_names = []
    for name in name_list:
        if name in ENG_names: 
            trans_name = name_correspondence[name]
        elif name in CHN_names:
            trans_name = ENG_names[CHN_names.index(name)]
        else:
            raise ValueError(f'there is not a corresponding translation for {name}')
        translated_names.append(trans_name)
    
    return translated_names
   
def collect_WT_data2(WT_idx, features, start_time = None, end_time = None):
    
    data_dir = 'C:/Users/wcfda/Desktop/Study/Projects/ViT_WTPM/codes/data/ANhui/D{}.xlsx'
    data = pd.read_excel(data_dir.format(WT_idx))
    features = ['DateAcqTime'] + features
    features_chinese = translate_names(features)
    feat_data = data[features_chinese]
    feat_data.columns = translate_names(feat_data.columns)
    feat_data = feat_data.dropna()
    
    timestamp = feat_data['DateAcqTime']
    if start_time is None:
        start_time = timestamp.min()
    if end_time is None:
        end_time = timestamp.max() + timedelta(seconds = 1)
    if start_time >= end_time: 
        raise ValueError('The end time must be after the start time')
    target_data = feat_data[(timestamp >= start_time) & (timestamp < end_time)]
    
    return target_data

def find_risky_Tstamp(feature_x, feature_y, q = 0.95):
    x_data = WT_data[feature_x]
    x_min, x_max = x_data.min(), x_data.max()
    x_LB = int(np.floor(x_min))
    x_UB = int(np.ceil(x_max))
    
    risky_Tstamps = []
    for x in range(x_LB, x_UB):
        data_x = WT_data[(x_data >= x) & (x_data < x + 1)]
        if data_x.shape[0] >= 100:
            y_thres = data_x[feature_y].quantile(q)
            risky_data_x = data_x[data_x[feature_y] > y_thres]
            risky_Tstamps_x = list(risky_data_x['DateAcqTime'])
            risky_Tstamps += risky_Tstamps_x
    
    return risky_Tstamps

def get_frequency(list_1d, start, end, step):
    array_1d = np.array(list_1d)
    freq = []
    periods = []
    x = start
    while x <= end:
        count = np.sum((array_1d >= x) & (array_1d < x + step))
        freq.append(count)
        periods.append([x, x + step])
        x += step
        
    return np.array(freq), np.array(periods)

def find_faulty_Tstamps(risky_Tstamps, t_start, t_end, 
                       delta_t, delta_t_obs, freq_thres):
    
    n_iter = int(delta_t/delta_t_obs)
    for i in range(n_iter):
        t_start_i = t_start - i*delta_t_obs
        risky_freq, time_periods = get_frequency(risky_Tstamps, t_start_i, t_end, delta_t)
        high_risk_periods = time_periods[risky_freq > freq_thres]
        high_risk_Tstamps = []
        for period in high_risk_periods:
            t_lb = period[0]
            t_rb = period[1]
            period_Tstamps = pd.date_range(t_lb, t_rb, freq = delta_t_obs, inclusive = 'left')
            high_risk_Tstamps += period_Tstamps.to_pydatetime().tolist()
        if i == 0:
            faulty_Tstamps = set(high_risk_Tstamps)
        else:
            faulty_Tstamps = faulty_Tstamps.intersection(set(high_risk_Tstamps))
    
    return list(faulty_Tstamps)

def visualize_WT_data(data, bounds, x_feature, time_step, save_dir, save = True, 
                      show = False, fig_size = (1.28, 1.28), cover = True):
    
    # remove wind speed outliers
    WS_data = data[x_feature]
    WS_LB, WS_UB = bounds[x_feature]
    WS_outlier_idx = np.where((WS_data < WS_LB) | (WS_data > WS_UB))[0]
    WS_outlier_prop = len(WS_outlier_idx)/len(WS_data)
    base, power = round_nonzero(WS_outlier_prop, 2)
    print(f'wind speed outlier proportion: {base}e{power + 2}%')
    data.drop(index = WS_outlier_idx, inplace = True)
    
    # standardize data and calculate the outlier factor
    features = list(bounds.keys())
    y_features = features.copy()
    y_features.remove(x_feature)
    outlier_fct_data = data[y_features].copy()
    
    for feat_name in features:
        feat_data = data[feat_name]
        LB, UB = bounds[feat_name]
        if feat_name != x_feature:
            factors = get_outlier_factor(feat_data, LB, UB)
            outlier_fct_data[feat_name] = factors
            feat_data = np.minimum(np.maximum(feat_data, LB), UB)
        data[feat_name] = (feat_data - LB)/(UB - LB)

    # visualize data
    time_data = data['DateAcqTime']
    start_time = time_data.min()
    end_time = time_data.max()
    unique_Tstamps = time_data.unique().to_pydatetime()
    obs_time_gaps = np.diff(np.sort(unique_Tstamps))
    obs_time_step = min(obs_time_gaps)
    ept_num_obs = int(time_step/obs_time_step)

    for y_feature in y_features:
        if save:
            Dir = save_dir.format(y_feature)
            if os.path.exists(Dir):
                if cover:
                    start_idx = 0
                else:
                    img_name_list = os.listdir(Dir)
                    img_idx_list = [int(name.split('.')[0]) for name in img_name_list]
                    start_idx = max(img_idx_list) + 1 if len(img_idx_list) > 0 else 0
            else:
                os.makedirs(Dir)
                start_idx = 0

        count = 0
        t0 = start_time
        while t0 <= end_time:
            t1 = t0 + time_step
            idx = np.where((time_data >= t0) & (time_data < t1))[0]
            actual_num_obs = len(idx)
            if actual_num_obs > 0.7*ept_num_obs:
                plot_data = data[[x_feature, y_feature]].iloc[idx]
                factors = outlier_fct_data[y_feature].iloc[idx]
                color_list = [str(f) for f in factors]
                # plot WT performance data
                plt.style.use('dark_background')
                f, ax = plt.subplots(1, 1, figsize = fig_size)
                ax.scatter(plot_data[x_feature], plot_data[y_feature], c = color_list, s = 3)
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.axis('off')
                if show: 
                    plt.show()
                if save: 
                    f.savefig(Dir + '{}.png'.format(start_idx + count), dpi = 100)
                plt.close()
            else:
                count -= 1
            # reassign values for the next time interval
            count += 1
            t0 = t1
            


    
    
if __name__ == '__main__':
    
    img_dir = r'C:/Users/wcfda/Desktop/Study/Projects/'\
        'ViT_WTPM/codes/image generation/img_data/{}/{}/{} state/{} images/'
    # ------------------------------------------------------------------------- Case1: blade breakage
    WT_idx = 198
    source = 'Ningxia'
    features = ['WindSpeed', 'PitchAngle1', 'PitchAngle2', 'PitchAngle3']

    start_time = datetime(2017, 7, 1, 0, 0, 0)
    end_time = datetime(2018, 8, 31, 23, 59, 59)
    WT_data = collect_WT_data(WT_idx, start_time, end_time, features)
    for feat_name in features:
        print(feat_name, 
              WT_data[feat_name].min(), 
              WT_data[feat_name].quantile(0.01),
              WT_data[feat_name].quantile(0.99),
              WT_data[feat_name].max())
    bounds = {features[0]: [0, 25], 
              features[1]: [0, 90],
              features[2]: [0, 90],
              features[3]: [0, 90]}
    
    start_time = datetime(2017, 8, 1, 0, 0, 0)
    end_time = datetime(2017, 10, 31, 23, 59, 59)
    healthy_data = collect_WT_data(WT_idx, start_time, end_time, features)
    save_dir = img_dir.format(f'{source}_{WT_idx}', {}, 'healthy', 'real')
    visualize_WT_data(healthy_data, bounds, 'WindSpeed', timedelta(hours = 3), save_dir)
    
    start_time = datetime(2018, 8, 1, 21, 0, 0)
    end_time = datetime(2018, 8, 2, 14, 59, 59)
    faulty_data = collect_WT_data(WT_idx, start_time, end_time, features)
    save_dir = img_dir.format(f'{source}_{WT_idx}', {}, 'faulty', 'real')
    visualize_WT_data(faulty_data, bounds, 'WindSpeed', timedelta(hours = 3), save_dir)
    
    # ------------------------------------------------------------------------- Case2: missing bolts and nuts
    WT_idx = 230
    source = 'Ningxia'
    features = ['WindSpeed', 'Nace_Vib_X', 'Nace_Vib_Y']

    start_time = datetime(2017, 7, 1, 0, 0, 0)
    end_time = datetime(2018, 8, 31, 23, 59, 59)
    WT_data = collect_WT_data(WT_idx, start_time, end_time, features)
    for feat_name in features:
        print(feat_name, 
              WT_data[feat_name].min(), 
              WT_data[feat_name].quantile(0.01),
              WT_data[feat_name].quantile(0.99),
              WT_data[feat_name].max(),
              )
    bounds = {features[0]: [0, 25], 
              features[1]: [0, 0.01],
              features[2]: [0, 0.015]}
    
    start_time = datetime(2017, 8, 1, 0, 0, 0)
    end_time = datetime(2017, 10, 31, 23, 59, 59)
    healthy_data = collect_WT_data(WT_idx, start_time, end_time, features)
    save_dir = img_dir.format(f'{source}_{WT_idx}', {}, 'healthy', 'real')
    visualize_WT_data(healthy_data, bounds, 'WindSpeed', timedelta(hours = 3), save_dir)
    
    start_time = datetime(2018, 8, 1, 0, 0, 0)
    end_time = datetime(2018, 8, 2, 23, 59, 59)
    faulty_data = collect_WT_data(WT_idx, start_time, end_time, features)
    save_dir = img_dir.format(f'{source}_{WT_idx}', {}, 'faulty', 'real')
    visualize_WT_data(faulty_data, bounds, 'WindSpeed', timedelta(hours = 3), save_dir)
    
    # ------------------------------------------------------------------------- Case3: Gearbox overheating
    name_correspondence = {
        'DateAcqTime': '时间',
        'wind_Spd_10m': '风速',
        'T_GBO_Visu': '齿箱油温',
        'T_GBS_Out_Visu': '齿箱轴1温度',
        'T_GBS_In_Visu': '齿箱轴2温度'
        }

    WT_idx = 5
    source = 'Anhui'
    features = ['wind_Spd_10m', 'T_GBO_Visu']
    info_dir = 'C:/Users/wcfda/Desktop/Study/Projects/ViT_WTPM/codes/data/ANhui/'
    
    WT_data = collect_WT_data2(WT_idx, features)
    t_start = WT_data['DateAcqTime'].min()
    t_end = WT_data['DateAcqTime'].max()
    delta_t = timedelta(days = 1)
    
    GB_risky_Tstamps = find_risky_Tstamp('wind_Spd_10m', 'T_GBO_Visu')
    risky_freq, _ = get_frequency(GB_risky_Tstamps, t_start, t_end, delta_t)
    plt.plot(risky_freq)
    plt.axhline(y = 15)

    freq_thres = 15
    delta_t_obs = timedelta(minutes = 10)
    GB_faulty_Tstamps = find_faulty_Tstamps(GB_risky_Tstamps, t_start, t_end, delta_t, delta_t_obs, freq_thres)
    with open (info_dir + 'GB_fault_timestamps', 'wb') as fp:
        pickle.dump(GB_faulty_Tstamps, fp)

    with open (info_dir + 'GB_fault_timestamps', 'rb') as fp:
        GB_faulty_Tstamps = pickle.load(fp)
    GB_faulty_dates = set()
    for t in GB_faulty_Tstamps:
        date = datetime(t.year, t.month, t.day)
        GB_faulty_dates.add(date)
    Tstamp_data = WT_data['DateAcqTime']
    date_data = [datetime(t.year, t.month, t.day) for t in Tstamp_data]
    is_faulty = [date in GB_faulty_dates for date in date_data]
    GB_faulty_idx = np.where(is_faulty)[0]
    with open (info_dir + 'GB_fault_indices', 'wb') as fp:
        pickle.dump(GB_faulty_idx, fp)

    healthy_data = WT_data.drop(index = GB_faulty_idx)
    faulty_data = WT_data.iloc[GB_faulty_idx]
    bounds = {features[0]: [0, 25], 
              features[1]: [0, 70]}

    save_dir = img_dir.format(f'{source}_{WT_idx}', {}, 'healthy', 'real')
    visualize_WT_data(healthy_data, bounds, 'wind_Spd_10m', timedelta(days = 1), save_dir)
    save_dir = img_dir.format(f'{source}_{WT_idx}', {}, 'faulty', 'real')
    visualize_WT_data(faulty_data, bounds, 'wind_Spd_10m', timedelta(days = 1), save_dir)

