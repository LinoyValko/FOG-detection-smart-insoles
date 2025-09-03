from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import save_file, uplaod_file, read_csv_skip_empty
import os
import sys

# def calculate_COP_window(window_data, side):
#     sensor_d = {
#         'hallux': (0.25, 0.90),
#         'toes': (0.10, 0.85),
#         'met1': (0.30, 0.70),
#         'met3': (0.175, 0.675),
#         'met5': (0.05, 0.65),
#         'arch': (0.10, 0.40),
#         'heelin': (0.25, 0.10),
#         'heelout': (0.125, 0.10)
#     }
#
#     sensors_labels = [f'{sensor}_{side}' for sensor in sensor_d.keys()]
#
#     forces_matrices = window_data[sensors_labels]
#
#     D_ml = np.array([coord[0] for coord in sensor_d.values()])
#     D_ap = np.array([coord[1] for coord in sensor_d.values()])
#
#     ml_cop = (forces_matrices * D_ml).sum().sum() / forces_matrices.sum().sum()
#     ap_cop = (forces_matrices * D_ap).sum().sum() / forces_matrices.sum().sum()
#
#     return ml_cop, ap_cop

# def calculate_velocity_and_acceleration_COP_window(features_df, step_size=0.2):
#     """
#     מוסיפה מהירות (velocity) ותאוצה (acceleration) בין חלונות זמן עוקבים, עבור כל רגל ולכל ציר.
#     """
#     df = features_df.copy()
#     df = df.sort_values('window_num').reset_index(drop=True)
#     for axis in ['ml', 'ap']:
#         for side in ['left', 'right']:
#             COP_column = f'COP_{axis}_{side}'
#             vel_column = f'COP_Velocity_{axis}_{side}'
#             acc_column = f'COP_Acceleration_{axis}_{side}'
#
#             df[vel_column] = np.gradient(df[COP_column], step_size)
#             df[acc_column] = np.gradient(df[vel_column], step_size)
#
#     print(f"✅ נוספו עמודות מהירות ותאוצה בין חלונות (Δt = {step_size}s).")
#     return df

# def calculate_velocity_and_acceleration_COP_row(df):
#     df = df.copy()
#
#     df['Time_by_Counter'] = pd.to_datetime(df['Time_by_Counter'], errors='coerce')
#     df['Time_diff'] = df['Time_by_Counter'].diff().dt.total_seconds()
#
#     for axis in ['ml', 'ap']:
#         for side in ['left', 'right']:
#             COP_column = f'COP_{axis}_{side}'
#             vel_column = f'COP_Velocity_{axis}_{side}'
#             acc_column = f'COP_Acceleration_{axis}_{side}'
#
#             df[vel_column] = df[COP_column].diff() / df['Time_diff']
#             df[acc_column] = df[vel_column].diff() / df['Time_diff']
#
#         COP_column = f'COP_{axis}'
#         vel_column = f'COP_Velocity_{axis}'
#         acc_column = f'COP_Acceleration_{axis}'
#
#         df[vel_column] = df[COP_column].diff() / df['Time_diff']
#         df[acc_column] = df[vel_column].diff() / df['Time_diff']
#
#     return df

# def extract_COP_features_window(df, window_col='window_num', expected_window_size=100):
#     """
#     מחשב פיצרים מבוססי COP עבור כל חלון (100 שורות) לפי נקודת ההתחלה בפועל.
#     מתקן את בעיית חפיפה בין חלונות.
#     """
#     features = []
#
#     # רק שורות שיש להן מספר חלון תקין
#     df = df[df[window_col] != -1].copy()
#     df = df.reset_index(drop=True)
#
#     # מציאת שורות התחלה – איפה שמספר החלון משתנה
#     window_change_indices = df[df[window_col] != df[window_col].shift(1)].index.tolist()
#
#     for start_idx in window_change_indices:
#         window_num = df.loc[start_idx, window_col]
#
#         # שליפת 100 שורות מהשורה הראשונה של החלון
#         window_df = df.loc[start_idx:start_idx + expected_window_size - 1]
#
#         # בדיקת גודל (יכול להיות שבסוף הקובץ יהיו פחות מ-100 שורות)
#         if window_df.shape[0] < expected_window_size:
#             continue
#
#         try:
#             ml_left, ap_left = calculate_COP_window(window_df, side='left')
#         except Exception:
#             ml_left, ap_left = np.nan, np.nan
#
#         try:
#             ml_right, ap_right = calculate_COP_window(window_df, side='right')
#         except Exception:
#             ml_right, ap_right = np.nan, np.nan
#
#         features.append({
#             'window_num': window_num,
#             'COP_ml_left': ml_left,
#             'COP_ap_left': ap_left,
#             'COP_ml_right': ml_right,
#             'COP_ap_right': ap_right
#         })
#
#     features_df = pd.DataFrame(features)
#     print(f"✅ נוצרו {len(features_df)} חלונות COP לפי 100 שורות מלאות לכל חלון.")
#
#     # הוספת מהירות ותאוצה בין חלונות:
#     features_df = calculate_velocity_and_acceleration_COP_window(features_df)
#     return features_df

# def extract_GRF_features_window(df, df_full_features, window_col='window_num', expected_window_size=100):
#     """
#     מחשב GRF ממוצע וחלק יחסי לכל רגל ומוסיף את הפיצ'רים ל-DataFrame של תכונות קיימות לפי window_num.
#     """
#     features = []
#     df = df[df[window_col] != -1].copy()
#     df = df.reset_index(drop=True)
#
#     pressure_cols = {
#         'left': [f'{sensor}_left' for sensor in ['hallux', 'toes', 'met1', 'met3', 'met5', 'arch', 'heelin', 'heelout']],
#         'right': [f'{sensor}_right' for sensor in ['hallux', 'toes', 'met1', 'met3', 'met5', 'arch', 'heelin', 'heelout']],
#     }
#
#     window_change_indices = df[df[window_col] != df[window_col].shift(1)].index.tolist()
#
#     for start_idx in window_change_indices:
#         window_num = df.loc[start_idx, window_col]
#         window_df = df.loc[start_idx:start_idx + expected_window_size - 1]
#
#         if window_df.shape[0] < expected_window_size:
#             continue
#
#         row = {'window_num': window_num}
#
#         for side in ['left', 'right']:
#             try:
#                 grf_series = window_df[pressure_cols[side]].sum(axis=1)
#                 row[f'GRF_{side}'] = grf_series.mean()
#
#             except Exception as e:
#                 print(f"⚠️ שגיאה בחישוב GRF בחלון {window_num}, צד {side}: {e}")
#                 row[f'GRF_{side}'] = np.nan
#
#         # חישוב חלק יחסי
#         total_grf = row.get('GRF_right', 0) + row.get('GRF_left', 0)
#         if total_grf > 0:
#             row['fraction_GRF_right'] = row['GRF_right'] / total_grf
#             row['fraction_GRF_left'] = row['GRF_left'] / total_grf
#         else:
#             row['fraction_GRF_right'] = np.nan
#             row['fraction_GRF_left'] = np.nan
#
#         features.append(row)
#
#     grf_df = pd.DataFrame(features)
#
#     # מיזוג עם df של פיצ'רים קיימים לפי window_num
#     features_df = pd.merge(df_full_features, grf_df, on='window_num', how='left')
#
#     print(f"✅ נוצרו {len(features_df)} שורות GRF ממוצע לחלונות.")
#     return features_df




# def extract_feature_columns_row(df):
#     feature_cols = [
#         'patient_num', 'COP_ml', 'COP_ap',
#         'COP_ml_left', 'COP_ap_left', 'COP_ml_right', 'COP_ap_right',
#         'COP_Velocity_ml', 'COP_Velocity_ap', 'COP_Acceleration_ml', 'COP_Acceleration_ap',
#         'COP_Velocity_ml_left', 'COP_Velocity_ap_left', 'COP_Velocity_ml_right', 'COP_Velocity_ap_right',
#         'COP_Acceleration_ml_left', 'COP_Acceleration_ap_left', 'COP_Acceleration_ml_right', 'COP_Acceleration_ap_right',
#         'GRF_left', 'GRF_right',
#         'fraction_GRF_left', 'fraction_GRF_right'
#     ]
#
#     # שמירה רק על עמודות שנמצאות בפועל ב-DataFrame
#     feature_cols_present = [col for col in feature_cols if col in df.columns]
#
#     return df[feature_cols_present].copy()


def calculate_COP_row(row, side):
    if side in ['left', 'right']:
        sensor_d = {
            'hallux': (0.25, 0.90),
            'toes': (0.10, 0.85),
            'met1': (0.30, 0.70),
            'met3': (0.175, 0.675),
            'met5': (0.05, 0.65),
            'arch': (0.10, 0.40),
            'heelin': (0.25, 0.10),
            'heelout': (0.15, 0.10)
        }
    else:
        sensor_d = {
            'R_hallux': (0.25, 0.90),
            'R_toes': (0.10, 0.85),
            'R_met1': (0.30, 0.70),
            'R_met3': (0.175, 0.675),
            'R_met5': (0.05, 0.65),
            'R_arch': (0.10, 0.40),
            'R_heelin': (0.25, 0.10),
            'R_heelout': (0.15, 0.10),
            'L_hallux': (0.475, 0.90),
            'L_toes': (0.60, 0.85),
            'L_met1': (0.45, 0.70),
            'L_met3': (0.55, 0.675),
            'L_met5': (0.675, 0.65),
            'L_arch': (0.65, 0.40),
            'L_heelin': (0.50, 0.10),
            'L_heelout': (0.60, 0.10),
        }

    sensors_labels = list(sensor_d.keys())

    try:
        forces_row = np.array([row[col] for col in sensors_labels])
        D_ml = np.array([coord[0] for coord in sensor_d.values()])
        D_ap = np.array([coord[1] for coord in sensor_d.values()])

        total_force = forces_row.sum()
        if total_force == 0:
            return np.nan, np.nan

        ml_cop = np.sum(forces_row * D_ml) / total_force
        ap_cop = np.sum(forces_row * D_ap) / total_force

        return ml_cop, ap_cop
    except Exception as e:
        print(f"⚠️Error in calculating COP in line: {e}")
        return np.nan, np.nan

def calculate_velocity_and_acceleration_COP_row(df, side):
    df = df.copy()

    if side in ['left', 'right']:  # מצב של רגל אחת
        df['Time_by_Counter'] = pd.to_datetime(df['Time_by_Counter'], errors='coerce')
        df['Time_diff'] = df['Time_by_Counter'].diff().dt.total_seconds()

        for axis in ['ml', 'ap']:
            COP_col = f'COP_{axis}'

            if COP_col in df.columns:
                df[f'COP_Velocity_{axis}'] = df[COP_col].diff() / df['Time_diff']
                df[f'COP_Acceleration_{axis}'] = df[f'COP_Velocity_{axis}'].diff() / df['Time_diff']
            else:
                print(f"⚠️ Missing column: {COP_col}")

        return df

    else:  # מצב של שתי רגליים
        df['L_Time_by_Counter'] = pd.to_datetime(df.get('L_Time_by_Counter'), errors='coerce')
        df['R_Time_by_Counter'] = pd.to_datetime(df.get('R_Time_by_Counter'), errors='coerce')

        df['L_Time_diff'] = df['L_Time_by_Counter'].diff().dt.total_seconds()
        df['R_Time_diff'] = df['R_Time_by_Counter'].diff().dt.total_seconds()

        for axis in ['ml', 'ap']:
            for side in ['left', 'right']:
                if side == 'left':
                    side = 'L'
                elif side == 'right':
                    side = 'R'
                COP_col = f'{side}_COP_{axis}'
                time_diff_col = 'L_Time_diff' if side == 'L' else 'R_Time_diff'

                if COP_col in df.columns:
                    df[f'{side}_COP_Velocity_{axis}'] = df[COP_col].diff() / df[time_diff_col]
                    df[f'{side}_COP_Acceleration_{axis}'] = df[f'{side}_COP_Velocity_{axis}'].diff() / df[time_diff_col]
                else:
                    print(f"⚠️ Missing column: {COP_col}")
        return df

def calculate_GRF_row(df, side):
    sensors = ['hallux', 'toes', 'met1', 'met3', 'met5', 'arch', 'heelin', 'heelout']

    if side in ['left', 'right']:
        pressure_cols = sensors
        df['GRF'] = df[pressure_cols].sum(axis=1)

    else:
        pressure_cols = {
            'L': [f'L_{sensor}' for sensor in ['hallux', 'toes', 'met1', 'met3', 'met5', 'arch', 'heelin', 'heelout']],
            'R': [f'R_{sensor}' for sensor in ['hallux', 'toes', 'met1', 'met3', 'met5', 'arch', 'heelin', 'heelout']]
        }

        df['L_GRF'] = df[pressure_cols['L']].sum(axis=1)
        df['R_GRF'] = df[pressure_cols['R']].sum(axis=1)

        df['L_fraction_GRF'] = df['L_GRF'] / (df['L_GRF'] + df['R_GRF'])
        df['R_fraction_GRF'] = df['R_GRF'] / (df['L_GRF'] + df['R_GRF'])

    return df


def extract_features_row(df, side):
    if side == 'both':
        df['L_COP_ml'], df['L_COP_ap'] = zip(*df.apply(lambda row: calculate_COP_row(row, side), axis=1))
        df['R_COP_ml'], df['R_COP_ap'] = zip(*df.apply(lambda row: calculate_COP_row(row, side), axis=1))
    else: # אם דאטה של רגל אחת
        df['COP_ml'], df['COP_ap'] = zip(*df.apply(lambda row: calculate_COP_row(row, side), axis=1))

    df = calculate_velocity_and_acceleration_COP_row(df, side)
    df = calculate_GRF_row(df, side)

    return df


def main(patient_num, T_num, study):
    data_path = fr'data\{study}\PRE_tagging_data'
    save_path = fr'data\PRE_features'
    features_dfs = []
    sides = ['left', 'right', 'both']

    for side in sides:
        data_folder = f'PRE_{side}_only_task_tagging_data'
        base_filename = f'{patient_num}_T{T_num}_{side}'
        save_data_folder = f'PRE_{side}_only_task_features'

        data = pd.read_csv(os.path.join(data_path, data_folder, base_filename + '.csv'))

        features_df = extract_features_row(data, side)

        columns_to_save = ['L_COP_ml', 'L_COP_ap', 'R_COP_ml', 'R_COP_ap', 'L_Time_diff', 'R_Time_diff',
                           'COP_ml', 'COP_ap', 'Time_diff', 'COP_Velocity_ml', 'COP_Acceleration_ml',
                           'COP_Velocity_ap', 'COP_Acceleration_ap', 'GRF',
                           'L_COP_Velocity_ml', 'L_COP_Acceleration_ml', 'R_COP_Velocity_ml', 'R_COP_Acceleration_ml',
                           'L_COP_Velocity_ap', 'L_COP_Acceleration_ap', 'R_COP_Velocity_ap', 'R_COP_Acceleration_ap',
                           'L_GRF', 'R_GRF', 'L_fraction_GRF', 'R_fraction_GRF', 'Task', 'State', 'FOG']
        columns_to_save = [col for col in columns_to_save if col in features_df.columns]

        features_dfs.append(features_df[columns_to_save])
        features_df[columns_to_save].to_csv(os.path.join(save_path, save_data_folder, base_filename + '_features.csv'), index=False)

    # איחוד של דאטה ימין ודאטה שמאל
    features_one_leg = pd.concat(features_dfs[:2], axis=0, ignore_index=True)
    one_leg_save_path = os.path.join(save_path, 'PRE_one_leg_only_task_features', f'{patient_num}_T{T_num}_one_leg_features.csv')
    features_one_leg.to_csv(one_leg_save_path, index=False)


if __name__ == '__main__':
    study = sys.argv[3]
    patient_num = sys.argv[1]
    T_num = sys.argv[2]
    # study = 'FOG_COA'
    # patient_num = '005'
    # T_num = 1
    main(patient_num, T_num, study)



