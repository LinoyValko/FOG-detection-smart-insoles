import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import re
import glob
import sys


def load_annotation_txt(file_path):
    # Try reading with automatic delimiter detection
    try:
        df = pd.read_csv(file_path, sep=None, engine='python')
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # If file opened but only one column exists – handle compressed header
    if len(df.columns) == 1 and isinstance(df.columns[0], str):
        raw_cols = df.columns[0].split(',')

        if len(raw_cols) > 1:
            # Clean the file: rebuild columns and skip the first row
            df = df.iloc[1:].copy()
            df.columns = raw_cols
            df.reset_index(drop=True, inplace=True)

    return df

def fix_task_column(annotation_df):
    # Map task numbers to short task codes
    task_map = {
        '1. Walk': 'stwalk',
        '2. Dual-task walk': 'dwalk',
        '3. Carrying': 'stcarr',
        '4. Turns': 'stturn',
        '5. Dual-task turns': 'dtturn',
        '6. Box shuffle': 'stshuf',
        '7. Box Agility': 'stagil',
        '8. Doorway': 'stdoor'
    }

    annotation_df['task'] = annotation_df['Task'].map(task_map)
    return annotation_df

def find_start_trial_time_opals(patient_num, T_num, task, state, study):
    # Locate Opals timing Excel file for the given patient and trial
    base_pattern = fr'N:\Gait-Neurodynamics by Names\Shahar\INSOLES project\Pycharm Insoles\data\{study}\{patient_num}*_T{T_num}\{patient_num}_opals_time.xlsx'
    matches = glob.glob(base_pattern)
    if not matches:
        raise FileNotFoundError(f"No matching file found for: {base_pattern}")

    # Read Opals timing data
    opals_time_path = matches[0]
    opals_time_data = pd.read_excel(opals_time_path)

    # Extract the start time for the selected task & state
    start_trial_opals = opals_time_data[(opals_time_data['trial'] == task) & (opals_time_data['state'] == state)]['Date Time']
    start_trial_time = pd.to_datetime(start_trial_opals.iloc[0])

    return start_trial_time

def insert_date_time_to_annotations(annotation_df, patient_num, T_num, task, state, study, start_trial_time):
    # Add absolute datetime columns by combining trial start time with relative offsets
    def calc_begin_time(row):
        return start_trial_time + pd.to_timedelta(row['Begin Time - hh:mm:ss.ms'])

    def calc_end_time(row):
        return start_trial_time + pd.to_timedelta(row['End Time - hh:mm:ss.ms'])

    annotation_df['Begin Date Time'] = annotation_df.apply(calc_begin_time, axis=1)
    annotation_df['End Date Time'] = annotation_df.apply(calc_end_time, axis=1)

    return annotation_df

def nan_value_FOG_column(annotation_df, task=None):
    # Handle missing FOG annotations differently for each study
    def nan_value_FOG_column_FOG_COA(annotation_df, task):
        # For FOG_COA: if 1 missing row → fill with task name
        # If 2 missing rows → keep the longest duration row
        if annotation_df['FOG'].isna().sum() == 1:
            annotation_df['FOG'] = annotation_df['FOG'].fillna(task)

        elif annotation_df['FOG'].isna().sum() == 2:
            annotation_df['Duration_TD'] = pd.to_timedelta(annotation_df['Duration - hh:mm:ss.ms'], errors='coerce')

            na_rows = annotation_df[annotation_df['FOG'].isna()]
            if not na_rows.empty:
                keep_idx = na_rows['Duration_TD'].idxmax()
                annotation_df['FOG'] = annotation_df['FOG'].astype('object')
                annotation_df.at[keep_idx, 'FOG'] = task

               # Drop the less relevant annotation row
                drop_idxs = na_rows.index.difference([keep_idx])
                annotation_df.drop(index=drop_idxs, inplace=True)

            annotation_df = annotation_df.drop('Duration_TD', axis=1)

        return annotation_df

    def del_duplicate_tasks_STEP_UP(annotation_df):
        # For STEP_UP: keep the FOG column as is.
        # If multiple rows exist for the same task without FOG, keep only the longest one.
        df = annotation_df.copy()

        df['FOG'] = df['FOG'].astype('object')
        df['task'] = df['task'].astype('object')

        # Convert duration to timedelta
        df['Duration_TD'] = pd.to_timedelta(df['Duration - hh:mm:ss.ms'], errors='coerce')

        # Identify duplicate tasks with missing FOG
        mask_no_fog = df['FOG'].isna()
        duplicated_tasks = (
            df[mask_no_fog]
            .sort_values(['task', 'Duration_TD'], ascending=[True, False])
            .duplicated(subset=['task'], keep='first')
        )

        idx_to_drop = df[mask_no_fog].loc[duplicated_tasks].index
        df = df.drop(index=idx_to_drop)

        # Fill missing FOG with task name
        df.loc[df['FOG'].isna(), 'FOG'] = df.loc[df['FOG'].isna(), 'task']
        df = df.drop('Duration_TD', axis=1)
        df = df.drop('task', axis=1)

        return df

    if task:
        annotation_df = nan_value_FOG_column_FOG_COA(annotation_df, task)
    else:
        annotation_df = del_duplicate_tasks_STEP_UP(annotation_df)

    return annotation_df


def clean_annotation_data(annotation_df, patient_num, T_num, study, task=None, state=None):
    # Standard cleaning pipeline for annotation data
    # 1. Fix task names
    annotation_df = fix_task_column(annotation_df)

    # 2. Drop irrelevant columns
    cols_to_drop = ['Begin Time - ss.msec', 'Task', 'End Time - ss.msec', 'Duration - ss.msec', 'default', 'Core']
    annotation_df = annotation_df.drop(cols_to_drop, axis=1)

    # 3. Choose timing method depending on study
    if study == 'FOG_COA':
        start_time = find_start_trial_time_opals(patient_num, T_num, task, state, study)

    else:  # STEP_UP
        start_time = find_start_video_time(patient_num, T_num, study)
        task = None

    # 4. Insert absolute time columns
    annotation_df = insert_date_time_to_annotations(annotation_df, patient_num, T_num, task, state, study, start_time)

     # 5. Handle missing FOG values
    annotation_df = nan_value_FOG_column(annotation_df, task)

    return annotation_df, start_time


def find_shift_time_opals_VS_insoles(patient_num, T_num, state, study):
    # Calculate the time difference (delta) between Opals and Insoles tapping events
    meta_data_path = fr'N:\Gait-Neurodynamics by Names\Shahar\INSOLES project\Pycharm Insoles\data\{study}\{study}_tapping_metadata.xlsx'
    meta_data = pd.read_excel(meta_data_path, dtype={'subject': str, 'T_num': int})

    start_stwalk_time = find_start_trial_time_opals(patient_num, T_num, 'stwalk', state, study)

    row = meta_data[
        (meta_data['subject'].astype(str) == str(patient_num)) &
        (meta_data['T_num'].astype(int) == int(T_num))
        ]

    if not row.empty:
        tapping_opals_time = row[f'Opals tapping {state}'].iloc[0]
        tapping_insoles_time = row[f'Insoles ACC tapping {state}'].iloc[0]

        # Align Opals tapping time with start walk
        tapping_opals_time = pd.to_timedelta(tapping_opals_time) + start_stwalk_time
        tapping_insoles_time = pd.to_timedelta(tapping_insoles_time)

        tapping_delta = tapping_insoles_time - pd.to_timedelta(tapping_opals_time.strftime('%H:%M:%S.%f'))
        return tapping_delta
    else:
        raise ValueError(f"Patient number {patient_num} not found in meta data!")


def find_start_video_time(patient_num, T_num, study):
    # Build path to the tapping metadata Excel file
    meta_data_path = fr'N:\Gait-Neurodynamics by Names\Shahar\INSOLES project\Pycharm Insoles\data\{study}\{study}_tapping_metadata.xlsx'
    meta_data = pd.read_excel(meta_data_path, dtype={'subject': str})

    # Filter metadata for the current patient and trial
    row = meta_data[
        (meta_data['subject'].astype(str) == str(patient_num)) &
        (meta_data['T_num'].astype(int) == int(T_num))
        ]

    if not row.empty:
        # Extract tapping times from Insoles and Video
        tapping_insoles_time = row[f'Insoles ACC tapping'].iloc[0]
        tapping_video_time = row[f'Video tapping'].iloc[0]

        # Convert Insoles tapping time to datetime (with the experiment date)
        tapping_insoles_datetime = row['Date'].iloc[0] + pd.to_timedelta(tapping_insoles_time)

        # Align start video time relative to insoles tapping
        start_video_time = tapping_insoles_datetime - pd.to_timedelta(tapping_video_time)

        return start_video_time

    else:
        raise ValueError(f"Patient number {patient_num} not found in meta data!")

def insert_insoles_time_to_annotation_df(annotation_df, tapping_delta=None):
    # Shift annotation times into the insoles timeline
    # If tapping_delta is provided, adjust Begin/End times by this offset
    if tapping_delta:
        annotation_df['Insoles Begin Time'] = annotation_df['Begin Date Time'] + tapping_delta
        annotation_df['Insoles End Time'] = annotation_df['End Date Time'] + tapping_delta
    else:
        # If no delta → just copy the original Date Times
        annotation_df['Insoles Begin Time'] = annotation_df['Begin Date Time']
        annotation_df['Insoles End Time'] = annotation_df['End Date Time']

    return annotation_df

def save_annotation_data(annotation_df, patient_num, T_num, task, state, base_pattern_path):
    # Save per-task annotations to CSV (used in FOG_COA)
    columns_to_save = ['FOG', 'Insoles Begin Time', 'Insoles End Time', 'Duration - hh:mm:ss.ms']
    # Keep only columns that exist
    columns_to_save = [col for col in columns_to_save if col in annotation_df.columns]

    # Build save folder: .../Annotations/<state>
    save_folder = os.path.join(glob.glob(base_pattern_path)[0], "Annotations", state)
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, f"annotation_{task}.csv")

    annotation_df[columns_to_save].to_csv(save_path, index=False)

    print(f'Annotations file: annotation_{task}.csv saved to: {save_folder}')

def extract_task_name(filename):
    # Extract task code (e.g., stwalk, dtturn) from filename using regex
    match = re.search(r'_(st\w+?|d\w+?)(?:_|$)', filename)
    return match.group(1) if match else 'unknown'

def save_all_annotations(folder_path, state):
    # Merge all task annotation CSV files from one state (off/on)
    saved_csv_files = glob.glob(os.path.join(folder_path, "annotation_*.csv"))
    all_annotations = []

    for csv_path in saved_csv_files:
        df = pd.read_csv(csv_path)
        df['state'] = state # Add column to mark the state
        all_annotations.append(df)

    if all_annotations:
        # Concatenate all annotation DataFrames
        merged_df = pd.concat(all_annotations, ignore_index=True)
        merged_save_path = os.path.join(folder_path, f"{state}_annotations.csv")
        merged_df.to_csv(merged_save_path, index=False)
        print(f"{state} annotations merged and saved to: {merged_save_path}")
        return merged_df

def save_annotation_data_STEP_UP(folder_path, annotation_df):
    # Save STEP_UP annotations (different structure vs FOG_COA)
    columns_to_save = ['FOG', 'Insoles Begin Time', 'Insoles End Time', 'Duration - hh:mm:ss.ms', 'task']
    columns_to_save = [col for col in columns_to_save if col in annotation_df.columns]

    # STEP_UP doesn't distinguish "state", so fill with None
    annotation_df['state'] = None

    save_path = os.path.join(folder_path, f"annotation_df.csv")

    annotation_df[columns_to_save].to_csv(save_path, index=False)

    print(f'Annotations file: annotation_df.csv saved to: {folder_path}')

def save_combined_annotations(base_annotations_path):
    # Merge "off" and "on" annotations into a single all_annotations.csv
    off_path = os.path.join(base_annotations_path, "off")
    on_path = os.path.join(base_annotations_path, "on")

    off_df = save_all_annotations(off_path, "off")
    on_df = save_all_annotations(on_path, "on")

    # Concatenate both states
    combined_df = pd.concat([off_df, on_df], ignore_index=True)
    combined_save_path = os.path.join(base_annotations_path, "all_annotations.csv")
    combined_df.to_csv(combined_save_path, index=False)
    print(f"Combined annotations saved to: {combined_save_path}")

    return combined_df

def all_process_annotation_data_FOG_COA(patient_num, T_num):
    # Process all annotation .txt files for a given patient in the FOG_COA study
    study = 'FOG_COA'
    base_pattern_path = fr"N:\Gait-Neurodynamics by Names\Shahar\INSOLES project\Pycharm Insoles\data\{study}\{patient_num}*T{T_num}"

    def process_annotation_data_per_state(patient_num, T_num, state):
        # Locate folder for the given state (on/off)
        folder_path = os.path.join(glob.glob(base_pattern_path)[0], "Annotations", state)
        txt_annotation_files = glob.glob(os.path.join(folder_path, "*.txt"))

        for annotation_txt_path in txt_annotation_files:
            # Load annotation text file
            annotation_df = load_annotation_txt(annotation_txt_path)

            # Extract short task name from file
            task = extract_task_name(annotation_txt_path)

            # Clean data: fix tasks, add timestamps, handle missing FOG
            annotation_df, _ = clean_annotation_data(annotation_df, patient_num, T_num, study, task, state)

            # Align Opals vs Insoles time
            tapping_delta = find_shift_time_opals_VS_insoles(patient_num, T_num, state, study)
            annotation_df = insert_insoles_time_to_annotation_df(annotation_df, tapping_delta)

            # Save processed annotation to CSV
            save_annotation_data(annotation_df, patient_num, T_num, task, state, base_pattern_path)

    # Run for both states
    process_annotation_data_per_state(patient_num, T_num, 'off')
    process_annotation_data_per_state(patient_num, T_num, 'on')

    # Merge "off" and "on" annotations into one file
    save_folder_path = os.path.join(glob.glob(base_pattern_path)[0], "Annotations")
    combined_df = save_combined_annotations(save_folder_path)

    return combined_df


def all_process_annotation_data_STEP_UP(patient_num, T_num):
    # STEP_UP study: process annotation files for a patient and trial
    study = 'STEP_UP'
    base_pattern_path = fr"N:\Gait-Neurodynamics by Names\Shahar\INSOLES project\Pycharm Insoles\data\{study}\{patient_num}*T{T_num}"

    # Locate the annotations folder and read the TXT annotation file
    folder_path = os.path.join(glob.glob(base_pattern_path)[0], "Annotations")
    txt_annotation_path = glob.glob(os.path.join(folder_path, "*.txt"))[0]
    annotation_df = load_annotation_txt(txt_annotation_path)

    # Clean and prepare annotation data (add times, fix tasks, handle missing values)
    annotation_df, start_time = clean_annotation_data(annotation_df, patient_num, T_num, study)

    # Add insoles-aligned time columns
    annotation_df = insert_insoles_time_to_annotation_df(annotation_df)
    # Save the processed annotation data
    save_annotation_data_STEP_UP(folder_path, annotation_df)

    return annotation_df

def tagging_fog_and_tasks(annotations_df, df):
    # Convert all time columns to datetime
    df['R_Time_by_Counter'] = pd.to_datetime(df['R_Time_by_Counter'], errors='coerce')
    df['L_Time_by_Counter'] = pd.to_datetime(df['L_Time_by_Counter'], errors='coerce')
    annotations_df['Insoles Begin Time'] = pd.to_datetime(annotations_df['Insoles Begin Time'], errors='coerce')
    annotations_df['Insoles End Time'] = pd.to_datetime(annotations_df['Insoles End Time'], errors='coerce')

    # Print ranges for validation
    print("Task time range:", annotations_df['Insoles Begin Time'].min(), "–", annotations_df['Insoles End Time'].max())
    print("Insoles time range:", df['R_Time_by_Counter'].min(), "–", df['R_Time_by_Counter'].max())

    # Extract FOG annotation ranges
    fog_rows = annotations_df[annotations_df['FOG'] == 'FOG']
    FOG_ranges = list(zip(fog_rows['Insoles Begin Time'], fog_rows['Insoles End Time']))

    # Extract Task annotation ranges
    task_rows = annotations_df[annotations_df['FOG'] != 'FOG']
    Task_ranges = list(zip(
        task_rows['Insoles Begin Time'],
        task_rows['Insoles End Time'],
        task_rows['FOG'],
        task_rows['state']
    ))

    # Helper: check if timestamp is within any FOG interval
    def tag_fog(ts):
        for start, end in FOG_ranges:
            if pd.notna(start) and pd.notna(end) and start <= ts <= end:
                return 1
        return 0

    # Helper: assign a Task/State if timestamp falls into task interval
    def tag_task(ts_r, ts_l):
        for start, end, task, state in Task_ranges:
            if pd.notna(start) and pd.notna(end):
                if (start <= ts_r <= end) or (start <= ts_l <= end):
                    print(start, end, task)
                    return task, state
        return None, None

    # Apply FOG tagging
    df['FOG'] = df.apply(lambda row: max(tag_fog(row['R_Time_by_Counter']), tag_fog(row['L_Time_by_Counter'])), axis=1)

    # Apply Task tagging
    task_results = df.apply(lambda row: tag_task(row['R_Time_by_Counter'], row['L_Time_by_Counter']), axis=1)
    df[['Task', 'State']] = pd.DataFrame(task_results.tolist(), index=df.index)
    print(df[df['Task'].notna()])
    return df, FOG_ranges, Task_ranges

def save_tagged_insole_data(df_full, patient_num, T_num, study):
    # Save full tagging data and a subset containing only rows with tasks
    base_folder = fr'N:\Gait-Neurodynamics by Names\Shahar\INSOLES project\Pycharm Insoles\data\{study}\PRE_tagging_data(4)'
    filename_base = f"{patient_num}_T{T_num}"

    # Save full dataset with FOG and tasks
    full_path = os.path.join(base_folder, f"{filename_base}_full_tagging.csv")
    df_full.to_csv(full_path, index=False)

    # Save only rows that belong to a task
    task_only_path = os.path.join(base_folder, f"{filename_base}_tasks_only_tagging.csv")
    df_task_only = df_full[df_full['Task'].notna()]
    df_task_only.to_csv(task_only_path, index=False)

    return df_full, df_task_only

def split_tagged_data(df, patient_num, T_num, study):
    # Split data into three subsets: both legs, right only, left only
    both_legs_df = df[df['R_counter'].notna() & df['L_counter'].notna()].copy()
    right_only_df = df[df['R_counter'].notna() & df['L_counter'].isna()].copy()
    left_only_df = df[df['L_counter'].notna() & df['R_counter'].isna()].copy()

    # Columns that should be removed from all subsets
    columns_to_drop = [
        'R_agreement', 'R_Left_idx_match', 'R_counter', 'R_time_ms', 'R_medication_state',
        'L_agreement', 'L_Right_idx_match', 'L_counter', 'L_time_ms', 'L_medication_state', 'delta_time'
    ]

    # Drop redundant columns
    for df_part in [both_legs_df, right_only_df, left_only_df]:
        for col in columns_to_drop:
            if col in df_part.columns:
                df_part.drop(columns=col, inplace=True)

    # Remove columns of the opposite leg from right-only and left-only subsets
    right_only_df = right_only_df.drop(columns=[col for col in right_only_df.columns if col.startswith('L_')])
    left_only_df = left_only_df.drop(columns=[col for col in left_only_df.columns if col.startswith('R_')])

    # Rename columns to drop prefixes (R_/L_)
    right_only_df.columns = [col[2:] if col.startswith('R_') else col for col in right_only_df.columns]
    left_only_df.columns = [col[2:] if col.startswith('L_') else col for col in left_only_df.columns]

    # Save each subset separately
    save_dir = fr'N:\Gait-Neurodynamics by Names\Shahar\INSOLES project\Pycharm Insoles\data\{study}\PRE_tagging_data(4)'

    both_legs_df.to_csv(os.path.join(save_dir, 'PRE_both_only_task_tagging_data', f"{patient_num}_T{T_num}_both.csv"), index=False)
    right_only_df.to_csv(os.path.join(save_dir, 'PRE_right_only_task_tagging_data', f"{patient_num}_T{T_num}_right.csv"), index=False)
    left_only_df.to_csv(os.path.join(save_dir, 'PRE_left_only_task_tagging_data', f"{patient_num}_T{T_num}_left.csv"), index=False)

    print("Splitting the data into right leg, left leg, and both legs is complete.")
    print(f"Rows with two-legged data: {len(both_legs_df)}")
    print(f"Rows with right data only: {len(right_only_df)}")
    print(f"Rows with left data only: {len(left_only_df)}")

    return both_legs_df, right_only_df, left_only_df

def main(patient_num, T_num, study):
    # Step 1: Load and process annotations
    if study == 'FOG_COA':
        annotations_df = all_process_annotation_data_FOG_COA(patient_num, T_num)
    else:   # study == STEP_UP
        annotations_df = all_process_annotation_data_STEP_UP(patient_num, T_num)

    # Step 2: Tag insoles data with FOG and tasks
    df_path = fr'N:\Gait-Neurodynamics by Names\Shahar\INSOLES project\Pycharm Insoles\data\{study}\PRE_combine_data(3)\{patient_num}_T{T_num}_combine.csv'
    df = pd.read_csv(df_path, low_memory=False)
    df_tagging, FOG_ranges, Task_ranges = tagging_fog_and_tasks(annotations_df, df)

    # Step 3: Save tagged datasets
    df_full, df_task_only = save_tagged_insole_data(df_tagging, patient_num, T_num, study)

    # Step 4: Split task-only data into subsets
    both_legs_df, right_only_df, left_only_df = split_tagged_data(df_task_only, patient_num, T_num, study)


if __name__ == '__main__':
    study_name = sys.argv[1]
    patient_num = sys.argv[2]
    T_num = sys.argv[3]

    main(patient_num, T_num, study_name)
