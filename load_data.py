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

    # Apply special preparation for FOG_HOME trial
    if trial_name == 'FOG_HOME':
        df = FOG_HOME_file_preparation(df)

    # Define relevant sensor columns
    sensor_labels = ['hallux', 'toes', 'met1', 'met3', 'met5', 'arch', 'heelin', 'heelout']

    # Keep only rows where sensor values are within [0, 200]
    df_filtered = df[(df[sensor_labels] <= 200).all(axis=1) &
                     (df[sensor_labels] >= 0).all(axis=1)].copy()

    # Clip the sensor values to the range [20, 120] to remove outliers
    df_filtered.loc[:, sensor_labels] = df[sensor_labels].clip(lower=20, upper=120)

    return df_filtered

# Create proper time columns from raw phonetime and counter values
def create_time_column(df):
    # Convert all values to float
    df = df.astype(float)
    df['phonetime_ms'] = df['phonetime']
    df['phonetime'] = df['phonetime'].apply(lambda time: datetime.fromtimestamp(int(time) / 1000))

    # Sort the DataFrame by the 'counter' column in ascending order
    df = df.sort_values(by='counter', ascending=True).reset_index(drop=True)

    # Normalize the counter so it starts from zero
    initial_counter = df['counter'].min()
    df['counter'] = df['counter'] - initial_counter

    # Create a continuous time column based on the counter (100 Hz sampling)
    start_time = df['phonetime'].min()
    df['Time_by_Counter'] = start_time + pd.to_timedelta((df['counter']) / 100, unit='s')


    # Create a time column in milliseconds for accurate calculation
    df['time_ms'] = df['Time_by_Counter'].astype('int64') // 10 ** 6

    return df

def extract_medication_state(df, file_name):
    # Determine the medication state (on/off/unknown) from the file name
    # and add it as a new column in the dataframe
    if '_on.csv' in file_name:
        state_medication = 'on'
    elif '_off.csv' in file_name:
        state_medication = 'off'
    else:
        state_medication = 'unknown'

    df['medication_state'] = state_medication

    return df, state_medication

def pre_preparation_data(trial_name, patient_num, file_side, file_type, data_dir, T_num):
    # Load raw files for a given patient, clean them, create time column,
    # normalize sensor values, and attach medication state information
    data_frames = upload_files_per_patient(trial_name, patient_num, file_side, file_type, data_dir, T_num)

    if not data_frames:
        # No files were found for this patient and side
        print(f"Patient Number {patient_num} has no valid {file_side} data.")
        df = pd.DataFrame()

    else:
        dfs = []
        for file_name, df in data_frames.items():
            print(file_name)
            if not df.empty:
                # Step 1: Clean raw data
                df_clean = clean_data(df, patient_num, file_side, trial_name)

                # Step 2: Create proper time columns
                df_with_time = create_time_column(df_clean) 
                print(df_with_time.columns)

                # Step 3: Normalize sensor pressures by total sum
                df_normalized = normalization_by_sum_of_pressures(df_with_time) 

                # Step 4: Add medication state column
                df_with_state, state_medication = extract_medication_state(df_normalized, file_name) 

                # Step 5: Save cleaned data for this file
                file_name_save = patient_num + file_side + file_type + f'_T{T_num}' + '_' + state_medication + '_clean'
                data_dir_save = os.path.join(data_dir, trial_name, 'PRE_clean_data')
                save_file(df_with_state, file_name_save, data_dir_save)

                dfs.append(df_with_state)

        # Concatenate all files for this patient/side into one dataframe
        df = pd.concat(dfs, ignore_index=True)

        # Save combined data for this patient/side
        all_file_name_save = patient_num + file_side + file_type + f'_T{T_num}'
        all_data_dir_save = os.path.join(data_dir, trial_name, 'PRE_data_by_foot')
        save_file(df, all_file_name_save, all_data_dir_save)

        # Log summary info
        print(f'Patient number: {patient_num}, {file_side} side, has shape data: {df.shape}')
    return df

def normalization_by_sum_of_pressures(df):

    # Normalize pressure sensor values by the maximum sum of pressures across all rows
    pressure_columns = ['hallux', 'toes', 'met1', 'met3', 'met5', 'arch', 'heelin', 'heelout']

    # Calculate the sum of pressures per row
    df['sum_pressure'] = df[pressure_columns].sum(axis=1)

    # Find the maximum overall pressure
    max_pressure = df['sum_pressure'].max()

    # Normalize each sensor by dividing by the maximum pressure
    df[pressure_columns] = df[pressure_columns] / max_pressure

    # Remove the temporary sum column
    df = df.drop(columns='sum_pressure')
    return df

def calculate_file_time_duration(df):

    # Calculate the gross and net duration of the recording based on the Time column
    gross_duration = df["Time"].max() - df["Time"].min()

    # Calculate time differences between consecutive rows
    df["Time_Diff"] = df["Time"].diff()

    # Net duration: sum only the valid intervals (<= 10 ms)
    net_duration = df["Time_Diff"][df["Time_Diff"] <= pd.Timedelta(milliseconds=10)].sum()

    # Print summary of durations
    print(f"Gross Duration: {gross_duration}")
    print(f"Net Duration: {net_duration}")

    return gross_duration, net_duration

def resample_and_interpolate(df, freq='10ms', max_gap='1min'):

    # Resample and interpolate the data to ensure consistent time intervals
    # freq: target frequency (e.g., 10 ms), max_gap: maximum gap allowed for interpolation

    # Convert the time column to a datetime index for resampling
    df = df.set_index("Time_by_Counter").sort_index()

    # Resample – creates new rows every 'freq' interval, averaging values within each window
    resampled = df.resample(freq).mean()

    # Identify missing samples (NaNs) and large gaps between samples
    missing_mask = resampled.isna().all(axis=1)
    time_diffs = resampled.index.to_series().diff()
    large_gaps = time_diffs > pd.Timedelta(max_gap)

    # For large gaps, keep them as NaN (do not interpolate)
    resampled.loc[large_gaps, :] = np.nan

    # Linear interpolation for missing values within allowed gaps
    # Uses the trend of previous and next values to fill in the missing data
    resampled = resampled.interpolate(method='linear', limit_area="inside")

    # Print summary of resampling results
    print(f"Number of samples before resampling: {df.shape[0]}")
    print(f"Number of samples after resampling: {resampled.shape[0]}")
    print(f"Number of large gaps that were NOT interpolated: {large_gaps.sum()}")

    return resampled.reset_index()

def debug_merge_diagnostics(df_right, df_left, patient_num, trial_name, data_dir, tolerance_ms=10):
    # Ensure both dataframes have datetime format for 'Time_by_Counter'
    df_right['Time_by_Counter'] = pd.to_datetime(df_right['Time_by_Counter'], errors='coerce')
    df_left['Time_by_Counter'] = pd.to_datetime(df_left['Time_by_Counter'], errors='coerce')

    # Print the time ranges for each side (right and left foot)
    print(f"\nPatient: {patient_num}")
    print(f"\nRight time range: {df_right['Time_by_Counter'].min()} → {df_right['Time_by_Counter'].max()}")
    print(f"Left time range : {df_left['Time_by_Counter'].min()} → {df_left['Time_by_Counter'].max()}")

    # Count unique timestamps and intersections
    unique_right = df_right['Time_by_Counter'].nunique()
    unique_left = df_left['Time_by_Counter'].nunique()
    intersection = len(set(df_right['Time_by_Counter']).intersection(set(df_left['Time_by_Counter'])))
    only_in_right = len(df_right[~df_right['Time_by_Counter'].isin(df_left['Time_by_Counter'])])
    only_in_left = len(df_left[~df_left['Time_by_Counter'].isin(df_right['Time_by_Counter'])])

    print(f"Unique Right: {unique_right}")
    print(f"Unique Left : {unique_left}")
    print(f"Intersection: {intersection}")
    print(f"Right-only rows: {only_in_right}")
    print(f"Left-only rows : {only_in_left}")

    # Convert datetime to int64 (milliseconds) for efficient delta calculations
    right_times = df_right['Time_by_Counter'].dropna().values.astype('datetime64[ms]').astype(np.int64)
    left_times = df_left['Time_by_Counter'].dropna().values.astype('datetime64[ms]').astype(np.int64)

     # For each left timestamp, find nearest right timestamps using searchsorted
    idxs = np.searchsorted(right_times, left_times)
    idxs = np.clip(idxs, 1, len(right_times) - 1)

    prev = right_times[idxs - 1]
    next_ = right_times[idxs]
    deltas = np.minimum(abs(left_times - prev), abs(left_times - next_))
    deltas_ms = deltas.astype(float)

    # Compute time delta statistics
    mean_diff = deltas_ms.mean()
    std_diff = deltas_ms.std()
    matched = (deltas_ms <= tolerance_ms).sum()

    print(f"\nTime delta between Left → Right (ms):")
    print(f"Mean: {mean_diff:.2f} ms, Std: {std_diff:.2f} ms")
    print(f"Matched within {tolerance_ms}ms: {matched} of {len(deltas_ms)}")

    # Collect metadata results into a dictionary
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

    # Save metadata into a CSV file inside the PRE_timing_metadata folder
    metadata_df = pd.DataFrame([metadata])
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, trial_name,'PRE_timing_metadata', f"{patient_num}_timing_metadata.csv")
    metadata_df.to_csv(save_path, index=False)
    print(f"\nMetadata saved to: {save_path}")

    return metadata_df

# Functions to combine right and left data files
def find_time_match_idx(df_1, df_2, side):

    # Ensure both dataframes have numeric time values in milliseconds
    df_1['time_ms'] = pd.to_numeric(df_1['time_ms'], errors='coerce')
    df_2['time_ms'] = pd.to_numeric(df_2['time_ms'], errors='coerce')

    # Sort both dataframes by time_ms for alignment
    df_2_sorted = df_2.sort_values('time_ms').reset_index()
    df_1_sorted = df_1.sort_values('time_ms').reset_index()

    # Extract arrays of times
    df_2_times = df_2_sorted['time_ms'].values
    df_1_times = df_1_sorted['time_ms'].values

    # Find approximate insertion indices of df_1 times within df_2 times
    indices = np.searchsorted(df_2_times, df_1_times)

    best_matches = []
    for i, t in enumerate(df_1_times):
        idx = indices[i]
        candidates = []
        # Compare with the closest time before
        if idx > 0:
            candidates.append((abs(t - df_2_times[idx - 1]), idx - 1))
        # Compare with the closest time after
        if idx < len(df_2_times):
            candidates.append((abs(t - df_2_times[idx]), idx))
        # Choose the closest match
        best_idx = min(candidates)[1]
        # Save the original index from df_2
        best_matches.append(df_2_sorted.loc[best_idx, 'index'])

    # Add new column with matching indices from the other dataframe
    df_1[f'{side}_idx_match'] = best_matches

    return df_1


def find_agreements(df_right, df_left):
    # Create copies of right and left DataFrames with reset index
    df_r = df_right.copy().reset_index(drop=True)
    df_l = df_left.copy().reset_index(drop=True)

    matched_rows = []

    # Iterate through all rows of the right DataFrame
    for idx_right, row_right in df_r.iterrows():
        idx_left = row_right['Left_idx_match']
        # Check if index is valid
        if idx_left >= 0 and idx_left < len(df_l):
            # Confirm bidirectional agreement (right matches left and left matches right)
            if df_l.loc[idx_left, 'Right_idx_match'] == idx_right:
                # Mark both rows as agreement
                df_r.at[idx_right, 'agreement'] = 1
                df_l.at[idx_left, 'agreement'] = 1

                # Combine right and left row into a single row with prefixes
                combined_row = pd.concat([
                    row_right.add_prefix('R_'),
                    df_l.loc[idx_left].add_prefix('L_')
                ])
                matched_rows.append(combined_row)
    # Create DataFrame of matched rows
    df_matched = pd.DataFrame(matched_rows).reset_index(drop=True)

    return df_matched, df_r, df_l

def insert_unmatched_rows(df_unmatched, df_matched, side):
    # Insert rows that do not have a matching pair into the matched DataFrame

    # Determine prefix based on which side is unmatched
    if side == 'right':
        prefix = 'R_'
        other_side = 'L_'
    else:
        prefix = 'L_'
        other_side = 'R_'

    # Add prefix to unmatched DataFrame columns
    df_unmatched_prefixed = df_unmatched.add_prefix(prefix)
 
    # Ensure all columns from the other side exist (fill with NaN if missing)
    for col in df_matched.columns:
        if col.startswith(other_side) and col not in df_unmatched_prefixed.columns:
            df_unmatched_prefixed[col] = np.nan

    # Align column order to df_matched
    df_r_unmatched_prefixed = df_unmatched_prefixed[df_matched.columns]
    # Concatenate matched and unmatched rows
    df_combined = pd.concat([df_matched, df_r_unmatched_prefixed], ignore_index=True)
    # Sort by time column of the chosen side
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
   # Check jump statistics in the counter column and verify increments are equal to 1

    if counter_column not in df.columns:
        print(f" Column {counter_column} not found in DataFrame!")
        return

    # Calculate differences between consecutive counter values
    counter_diff = df[counter_column].diff().dropna()

    # Print descriptive statistics for counter differences
    print("Counter difference statistics:")
    print(counter_diff.describe())

    # Count occurrences of each jump size
    counter_jump_counts = counter_diff.value_counts().sort_index()
    print("\nCounter jump counts:")
    print(counter_jump_counts)

def main(patient_num, T_num, study):
    # Set base directory for data
    data_dir = r'data'

    # Run preprocessing for right and left foot data
    pre_right = pre_preparation_data(study, patient_num, 'Right', 'PRE', data_dir, T_num)
    pre_left = pre_preparation_data(study, patient_num, 'Left', 'PRE', data_dir, T_num)

    # Combine right and left foot data into a single dataframe
    pre_data = combine_right_left(pre_right, pre_left)

    # Save combined dataframe to output directory
    data_dir_save = os.path.join(data_dir, study, 'PRE_combine_data')
    save_file(pre_data, f'{patient_num}_T{T_num}_combine', data_dir_save)


if __name__ == '__main__':
     # Example execution (can be replaced with CLI arguments)
    patient_num = '015'
    T_num = 2
    study = 'FOG_COA'

    # Run the pipeline
    main(patient_num, T_num, study)


