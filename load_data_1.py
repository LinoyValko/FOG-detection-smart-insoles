import pandas as pd
import os
import glob
from datetime import datetime
from utils import save_file, read_csv_skip_empty
import numpy as np
import sys

def upload_files_per_patient(trial_name, patient_num, file_side, file_type, data_dir, T_num):
    # Find files
    files_list = glob.glob(os.path.join(data_dir, trial_name, patient_num + f"*{T_num}", file_type, f'*{file_side}*'))
    # Read, concatenate, and save files
    data_frames = {}
    for file in files_list:
        df = read_csv_skip_empty(file)
        if df is not None:
            df = df.drop(0)
            df = df.drop(df.index[-1])
            data_frames[file] = df

    return data_frames

def FOG_HOME_file_preparation(df):
    df = df.drop(index=df.index[:2])
    df = df.drop(columns=df.columns[10:])

    new_columns_names = ['phonetime', 'counter', 'hallux', 'toes', 'met1', 'met3', 'met5', 'arch', 'heelin', 'heelout']

    df.columns = new_columns_names

    return df

def clean_data(df, patient_num, file_side, trial_name):
    # Removes spaces from column names
    df = df.rename(columns=lambda x: x.strip())

    if trial_name == 'FOG_HOME':
        df = FOG_HOME_file_preparation(df)

    # Filter the rows that have values over 200 and less than 0 (according to the insoles manual)
    # df = df[(df.iloc[:, 1:9] <= 200).all(axis=1) & (df.iloc[:, 1:9] >= 0).all(axis=1)]
    sensor_labels = ['hallux', 'toes', 'met1', 'met3', 'met5', 'arch', 'heelin', 'heelout']
    df_filtered = df[(df[sensor_labels] <= 200).all(axis=1) &
                     (df[sensor_labels] >= 0).all(axis=1)].copy()

    # clipping thr data
    df_filtered.loc[:, sensor_labels] = df[sensor_labels].clip(lower=20, upper=120)

    return df_filtered

def create_time_column(df):
    df = df.astype(float)
    df['phonetime_ms'] = df['phonetime']
    df['phonetime'] = df['phonetime'].apply(lambda time: datetime.fromtimestamp(int(time) / 1000))

    # Sort the DataFrame by the 'counter' column in ascending order
    df = df.sort_values(by='counter', ascending=True).reset_index(drop=True)
    initial_counter = df['counter'].min()
    df['counter'] = df['counter'] - initial_counter

    start_time = df['phonetime'].min()
    df['Time_by_Counter'] = start_time + pd.to_timedelta((df['counter']) / 100, unit='s')

    # df = df.drop(columns='phonetime')

    # צור עמודה של זמן במילישניות לצורך חישוב מדויק
    df['time_ms'] = df['Time_by_Counter'].astype('int64') // 10 ** 6

    return df

def extract_medication_state(df, file_name):
    if '_on.csv' in file_name:
        state_medication = 'on'
    elif '_off.csv' in file_name:
        state_medication = 'off'
    else:
        state_medication = 'unknown'

    df['medication_state'] = state_medication

    return df, state_medication

def pre_preparation_data(trial_name, patient_num, file_side, file_type, data_dir, T_num):
    data_frames = upload_files_per_patient(trial_name, patient_num, file_side, file_type, data_dir, T_num)

    if not data_frames:
        print(f"Patient Number {patient_num} has no valid {file_side} data.")
        df = pd.DataFrame()

    else:
        dfs = []
        for file_name, df in data_frames.items():
            print(file_name)
            if not df.empty:
                df_clean = clean_data(df, patient_num, file_side, trial_name)
                df_with_time = create_time_column(df_clean)
                print(df_with_time.columns)
                df_normalized = normalization_by_sum_of_pressures(df_with_time)

                df_with_state, state_medication = extract_medication_state(df_normalized, file_name)

                # Resampling & Linear interpolation
                # df = resample_and_interpolate(df_with_time)

                # save clean data
                file_name_save = patient_num + file_side + file_type + f'_T{T_num}' + '_' + state_medication + '_clean'
                data_dir_save = os.path.join(data_dir, trial_name, 'PRE_clean_data')
                save_file(df_with_state, file_name_save, data_dir_save)
                # df_with_state.to_csv(os.path.join(data_dir, trial_name, 'new', file_name_save))

                dfs.append(df_with_state)

        df = pd.concat(dfs, ignore_index=True)

        all_file_name_save = patient_num + file_side + file_type + f'_T{T_num}'
        all_data_dir_save = os.path.join(data_dir, trial_name, 'PRE_data_by_foot')
        save_file(df, all_file_name_save, all_data_dir_save)

        print(f'Patient number: {patient_num}, {file_side} side, has shape data: {df.shape}')
    return df

def normalization_by_sum_of_pressures(df):
    pressure_columns = ['hallux', 'toes', 'met1', 'met3', 'met5', 'arch', 'heelin', 'heelout']

    # סכימת לחצים לכל שורה
    df['sum_pressure'] = df[pressure_columns].sum(axis=1)

    # לחץ מקסימלי כללי
    max_pressure = df['sum_pressure'].max()

    # נרמול כל סנסור לפי הלחץ המקסימלי
    df[pressure_columns] = df[pressure_columns] / max_pressure

    df = df.drop(columns='sum_pressure')
    return df

def calculate_file_time_duration(df):
    gross_duration = df["Time"].max() - df["Time"].min()

    df["Time_Diff"] = df["Time"].diff()

    net_duration = df["Time_Diff"][df["Time_Diff"] <= pd.Timedelta(milliseconds=10)].sum()

    print(f"Gross Duration: {gross_duration}")
    print(f"Net Duration: {net_duration}")

    return gross_duration, net_duration

def resample_and_interpolate(df, freq='10ms', max_gap='1min'):
    # Converting the time column to a Datetime index
    df = df.set_index("Time_by_Counter").sort_index()

    # Resampling – creates new samples every 10ms, calculates an average for values within each 10ms time window
    resampled = df.resample(freq).mean()

    # Identifying large gaps between samples
    missing_mask = resampled.isna().all(axis=1)
    time_diffs = resampled.index.to_series().diff()
    large_gaps = time_diffs > pd.Timedelta(max_gap)

    # Marking areas where interpolation is not done
    resampled.loc[large_gaps, :] = np.nan

    # Linear interpolation – calculates missing values (NaN) according to the trend of the data
    # Uses previous and next values to calculate the missing value in a straight line
    # Suitable for all numeric columns
    resampled = resampled.interpolate(method='linear', limit_area="inside")

    print(f"Number of samples before resampling: {df.shape[0]}")
    print(f"Number of samples after resampling: {resampled.shape[0]}")
    print(f"Number of large gaps that were NOT interpolated: {large_gaps.sum()}")

    return resampled.reset_index()

def debug_merge_diagnostics(df_right, df_left, patient_num, trial_name, data_dir, tolerance_ms=10):
    # Ensure datetime format
    df_right['Time_by_Counter'] = pd.to_datetime(df_right['Time_by_Counter'], errors='coerce')
    df_left['Time_by_Counter'] = pd.to_datetime(df_left['Time_by_Counter'], errors='coerce')

    # Print time range
    print(f"\n📅 Patient: {patient_num}")
    print(f"\n🕒 Right time range: {df_right['Time_by_Counter'].min()} → {df_right['Time_by_Counter'].max()}")
    print(f"🕒 Left time range : {df_left['Time_by_Counter'].min()} → {df_left['Time_by_Counter'].max()}")

    # Basic counts
    unique_right = df_right['Time_by_Counter'].nunique()
    unique_left = df_left['Time_by_Counter'].nunique()
    intersection = len(set(df_right['Time_by_Counter']).intersection(set(df_left['Time_by_Counter'])))
    only_in_right = len(df_right[~df_right['Time_by_Counter'].isin(df_left['Time_by_Counter'])])
    only_in_left = len(df_left[~df_left['Time_by_Counter'].isin(df_right['Time_by_Counter'])])

    print(f"🔢 Unique Right: {unique_right}")
    print(f"🔢 Unique Left : {unique_left}")
    print(f"🔁 Intersection: {intersection}")
    print(f"⏳ Right-only rows: {only_in_right}")
    print(f"⏳ Left-only rows : {only_in_left}")

    # Convert to int64 milliseconds for fast delta computation
    right_times = df_right['Time_by_Counter'].dropna().values.astype('datetime64[ms]').astype(np.int64)
    left_times = df_left['Time_by_Counter'].dropna().values.astype('datetime64[ms]').astype(np.int64)

    # Search nearest indices
    idxs = np.searchsorted(right_times, left_times)
    idxs = np.clip(idxs, 1, len(right_times) - 1)

    prev = right_times[idxs - 1]
    next_ = right_times[idxs]
    deltas = np.minimum(abs(left_times - prev), abs(left_times - next_))
    deltas_ms = deltas.astype(float)

    mean_diff = deltas_ms.mean()
    std_diff = deltas_ms.std()
    matched = (deltas_ms <= tolerance_ms).sum()

    print(f"\n📊 Time delta between Left → Right (ms):")
    print(f"   Mean: {mean_diff:.2f} ms, Std: {std_diff:.2f} ms")
    print(f"   ✅ Matched within {tolerance_ms}ms: {matched} of {len(deltas_ms)}")

    # Save metadata
    metadata = {
        "patient_num": patient_num,
        "num_right_timestamps": len(df_right),
        "num_left_timestamps": len(df_left),
        "unique_right": unique_right,
        "unique_left": unique_left,
        "intersection": intersection,
        "only_in_right": only_in_right,
        "only_in_left": only_in_left,
        "mean_time_diff_ms": mean_diff,
        "std_time_diff_ms": std_diff,
        f"matched_within_{tolerance_ms}ms": matched
    }

    metadata_df = pd.DataFrame([metadata])
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, trial_name,'PRE_timing_metadata', f"{patient_num}_timing_metadata.csv")
    metadata_df.to_csv(save_path, index=False)
    print(f"\n💾 Metadata saved to: {save_path}")

    return metadata_df

# Functions to combine right and left data files
def find_time_match_idx(df_1, df_2, side):
    df_1['time_ms'] = pd.to_numeric(df_1['time_ms'], errors='coerce')
    df_2['time_ms'] = pd.to_numeric(df_2['time_ms'], errors='coerce')

    df_2_sorted = df_2.sort_values('time_ms').reset_index()
    df_1_sorted = df_1.sort_values('time_ms').reset_index()

    df_2_times = df_2_sorted['time_ms'].values
    df_1_times = df_1_sorted['time_ms'].values

    indices = np.searchsorted(df_2_times, df_1_times)

    best_matches = []
    for i, t in enumerate(df_1_times):
        idx = indices[i]
        candidates = []
        if idx > 0:
            candidates.append((abs(t - df_2_times[idx - 1]), idx - 1))
        if idx < len(df_2_times):
            candidates.append((abs(t - df_2_times[idx]), idx))

        best_idx = min(candidates)[1]
        best_matches.append(df_2_sorted.loc[best_idx, 'index'])

    df_1[f'{side}_idx_match'] = best_matches

    # df_1['Time_by_Counter'].apply(lambda time: datetime.fromtimestamp(int(time) / 1000))
    # df_2['Time_by_Counter'].apply(lambda time: datetime.fromtimestamp(int(time) / 1000))

    return df_1
def find_agreements(df_right, df_left):
    df_r = df_right.copy().reset_index(drop=True)
    df_l = df_left.copy().reset_index(drop=True)

    matched_rows = []
    # print(df_r[['Left_idx_match']].head())
    # print(df_l[['Right_idx_match']].head())
    # print(df_r.index)
    # print(df_l.index)

    for idx_right, row_right in df_r.iterrows():
        idx_left = row_right['Left_idx_match']
        if idx_left >= 0 and idx_left < len(df_l):
            if df_l.loc[idx_left, 'Right_idx_match'] == idx_right:
                df_r.at[idx_right, 'agreement'] = 1
                df_l.at[idx_left, 'agreement'] = 1

                combined_row = pd.concat([
                    row_right.add_prefix('R_'),
                    df_l.loc[idx_left].add_prefix('L_')
                ])
                matched_rows.append(combined_row)

    df_matched = pd.DataFrame(matched_rows).reset_index(drop=True)

    return df_matched, df_r, df_l
def insert_unmatched_rows(df_unmatched, df_matched, side):
    # print(f'df_matched columns: {df_matched.columns}')
    if side == 'right':
        prefix = 'R_'
        other_side = 'L_'
    else:
        prefix = 'L_'
        other_side = 'R_'

    df_unmatched_prefixed = df_unmatched.add_prefix(prefix)
    # print(f'df_unmatched_prefixed columns: {df_unmatched_prefixed.columns}')
    for col in df_matched.columns:
        if col.startswith(other_side) and col not in df_unmatched_prefixed.columns:
            df_unmatched_prefixed[col] = np.nan

    df_r_unmatched_prefixed = df_unmatched_prefixed[df_matched.columns]
    df_combined = pd.concat([df_matched, df_r_unmatched_prefixed], ignore_index=True)
    # print(f'df_combined columns: {df_combined.columns}')
    # print(df_combined['R_Time_by_Counter'])
    df_combined = df_combined.sort_values(f'{prefix}Time_by_Counter').reset_index(drop=True)

    return df_combined, df_unmatched_prefixed

def combine_right_left(df_right, df_left):
    # 1- insert column with zeros to describe the agreement
    df_right.insert(0, 'agreement', 0)
    df_left.insert(0, 'agreement', 0)

    # 2- Insert to right the match index of the left foot closest time
    df_right = find_time_match_idx(df_right, df_left, 'Left')

    # 3- Insert to left the match index of the right foot closest time
    df_left = find_time_match_idx(df_left, df_right, 'Right')

    # 4- find agreements
    df_matched, df_r, df_l = find_agreements(df_right, df_left)

    # 5- find unmatched rows in right and left
    df_r_unmatched = df_r[df_r['agreement'] != 1]
    df_l_unmatched = df_l[df_l['agreement'] != 1]

    # 6- insert unmatched right rows
    df_combined_right, df_r_unmatched_prefixed = insert_unmatched_rows(df_r_unmatched, df_matched, 'right')

    # 7- duplicate the right time to the left time in the nan values
    df_combined_right['L_Time_by_Counter'] = df_combined_right['L_Time_by_Counter'].fillna(df_combined_right['R_Time_by_Counter'])

    # 8- insert unmatched left rows
    df_combined, df_l_unmatched_prefixed = insert_unmatched_rows(df_l_unmatched, df_combined_right, 'left')

    # 9- delete the left time values in rows with only right data
    l_columns = [col for col in df_combined.columns if col.startswith('L_') and col != 'L_Time_by_Counter']
    mask_only_time = df_combined[l_columns].isna().all(axis=1)
    df_combined.loc[mask_only_time, 'L_Time_by_Counter'] = np.nan

    # 10- create delta column
    df_combined['delta_time'] = (df_combined['R_Time_by_Counter'] - df_combined['L_Time_by_Counter']).abs()

    return df_combined


def check_sampling_quality(df):
    # Checking for duplicate times
    duplicates = df["Time_by_Counter"].duplicated().sum()
    print(f"Number of duplicates in time: {duplicates}")

    # Testing samples with missing data
    missing = df.isna().sum().sum()
    print(f"Number of missing values: {missing}")

    # Checking for unusual jumps in time
    time_diff = df["Time_by_Counter"].diff().dropna().dt.total_seconds()
    large_jumps = (time_diff > 0.01).sum()  # קפיצות גדולות מ-10 מילי שניות
    print(f"Number of large jumps between samples (>10ms): {large_jumps}")

    # View statistics of time differences
    print(f"Time difference statistics:")
    print(time_diff.describe())
def check_counter_jumps(df, counter_column="counter"):
    """
    בודק סטטיסטיקת קפיצות בעמודת COUNTER ומוודא שכל קפיצה היא בדיוק 1.

    :param df: DataFrame שמכיל את עמודת ה-Counter
    :param counter_column: שם העמודה שמכילה את ה-Counter (ברירת מחדל "COUNTER")
    """
    if counter_column not in df.columns:
        print(f"⚠️ עמודת {counter_column} לא נמצאה ב-DataFrame!")
        return

    # חישוב ההפרשים בין ערכים עוקבים בעמודת COUNTER
    counter_diff = df[counter_column].diff().dropna()

    # הצגת סטטיסטיקה כללית
    print("📊 סטטיסטיקה של הפרשי COUNTER:")
    print(counter_diff.describe())

    # ספירת מספר ההופעות של כל קפיצה
    counter_jump_counts = counter_diff.value_counts().sort_index()

def main(patient_num, T_num, study):
    data_dir = r'data'

    pre_right = pre_preparation_data(study, patient_num, 'Right', 'PRE', data_dir, T_num)
    pre_left = pre_preparation_data(study, patient_num, 'Left', 'PRE', data_dir, T_num)

    pre_data = combine_right_left(pre_right, pre_left)
    data_dir_save = os.path.join(data_dir, study, 'PRE_combine_data')
    save_file(pre_data, f'{patient_num}_T{T_num}_combine', data_dir_save)




if __name__ == '__main__':
    # study = sys.argv[3]
    # patient_num = sys.argv[1]
    # T_num = sys.argv[2]
    patient_num = '015'
    T_num = 2
    study = 'FOG_COA'


    main(patient_num, T_num, study)


