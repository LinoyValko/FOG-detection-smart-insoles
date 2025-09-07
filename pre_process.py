import glob
import pandas as pd
import numpy as np
import os
import sys

def _fill_one_side(df, side, step_ms=10):
    # Fill missing counters and times for one side (R or L)
    counter_col = f'{side}_counter'
    time_ms_col = f'{side}_time_ms'
    time_dt_col = f'{side}_Time_by_Counter'

    # Select only columns related to this side
    cols = [c for c in [counter_col, time_ms_col, time_dt_col] if c in df.columns]
    s = df[cols].copy()

    # Ensure counter is Int64 (nullable)
    if counter_col not in s:
        return df  # No such side in dataframe
    s[counter_col] = pd.to_numeric(s[counter_col], errors='coerce').astype('Int64')

    # Keep only valid counters and set them as index
    g = s.dropna(subset=[counter_col]).set_index(counter_col).sort_index()
    g = g[~g.index.duplicated(keep='first')]

    if g.empty:
        return df

    # Build full counter range for this side
    cmin, cmax = int(g.index.min()), int(g.index.max())
    full_idx = pd.RangeIndex(cmin, cmax + 1, name=counter_col)
    g = g.reindex(full_idx)  

    # Convert time columns if present
    if time_ms_col in g.columns:
        g[time_ms_col] = pd.to_numeric(g[time_ms_col], errors='coerce')
    if time_dt_col in g.columns:
        g[time_dt_col] = pd.to_datetime(g[time_dt_col], errors='coerce')

    # Fill missing time_ms values based on the first known anchor
    if time_ms_col in g.columns and g[time_ms_col].notna().any():
        idx0 = g[time_ms_col].first_valid_index()
        t0   = float(g.loc[idx0, time_ms_col])
        mask = g[time_ms_col].isna()
        g.loc[mask, time_ms_col] = t0 + (g.index[mask] - idx0) * float(step_ms)

    # Fill missing datetime values based on the first known anchor
    if time_dt_col in g.columns and g[time_dt_col].notna().any():
        idx0 = g[time_dt_col].first_valid_index()
        t0   = g.loc[idx0, time_dt_col]
        mask = g[time_dt_col].isna()
        g.loc[mask, time_dt_col] = t0 + pd.to_timedelta((g.index[mask] - idx0) * step_ms, unit='ms')

    # Merge back to the original dataframe (outer join keeps rows from the other side)
    filled_side = g.reset_index()
    out = df.merge(filled_side, on=counter_col, how='outer', suffixes=('', '_new'))

    # Prefer new values when original values are missing
    for col in [time_ms_col, time_dt_col]:
        if col in out.columns and f'{col}_new' in out.columns:
            out[col] = out[col].combine_first(out[f'{col}_new'])
            out.drop(columns=[f'{col}_new'], inplace=True)

    # Ensure counter remains Int64
    out[counter_col] = pd.to_numeric(out[counter_col], errors='coerce').astype('Int64')

    return out

def fill_counters_both(df, step_ms=10):
    # Fill missing counters for both right and left sides
    out = df.copy()
    
    if any(c.startswith('R_') for c in out.columns):
        out = _fill_one_side(out, 'R', step_ms=step_ms)
    if any(c.startswith('L_') for c in out.columns):
        out = _fill_one_side(out, 'L', step_ms=step_ms)

     # Sort rows based on the earliest available time
    candidate_times = []
    if 'R_time_ms' in out: candidate_times.append('R_time_ms')
    if 'L_time_ms' in out: candidate_times.append('L_time_ms')
    if candidate_times:
        out['_sort_time'] = out[candidate_times].min(axis=1)
        out = out.sort_values('_sort_time', kind='stable').drop(columns=['_sort_time']).reset_index(drop=True)
    else:
        out = out.sort_values(['R_counter','L_counter'], na_position='first').reset_index(drop=True)

    return out

def tagging_fog_and_tasks(annotations_df, df):
    # Tag dataframe rows with FOG events and task labels based on annotation ranges

    # Convert all relevant time columns to datetime
    df['R_Time_by_Counter'] = pd.to_datetime(df['R_Time_by_Counter'], errors='coerce')
    df['L_Time_by_Counter'] = pd.to_datetime(df['L_Time_by_Counter'], errors='coerce')
    annotations_df['Insoles Begin Time'] = pd.to_datetime(annotations_df['Insoles Begin Time'], errors='coerce')
    annotations_df['Insoles End Time'] = pd.to_datetime(annotations_df['Insoles End Time'], errors='coerce')

    print("Tasks range:", annotations_df['Insoles Begin Time'].min(), "–", annotations_df['Insoles End Time'].max())
    print("Insoles range:", df['R_Time_by_Counter'].min(), "–", df['R_Time_by_Counter'].max())

    # Extract FOG and task ranges
    fog_rows = annotations_df[annotations_df['FOG'] == 'FOG']
    FOG_ranges = list(zip(fog_rows['Insoles Begin Time'], fog_rows['Insoles End Time']))

    task_rows = annotations_df[annotations_df['FOG'] != 'FOG']
    Task_ranges = list(zip(
        task_rows['Insoles Begin Time'],
        task_rows['Insoles End Time'],
        task_rows['FOG']
    ))

    # Tagging helper: FOG
    def tag_fog(ts):
        for start, end in FOG_ranges:
            if pd.notna(start) and pd.notna(end) and start <= ts <= end:
                return 1
        return 0

    # Tagging helper: Task
    def tag_task(ts_r, ts_l):
        for start, end, task in Task_ranges:
            if pd.notna(start) and pd.notna(end):
                if (start <= ts_r <= end) or (start <= ts_l <= end):
                    return task
        return None

    # Apply FOG tags
    df['FOG'] = df.apply(lambda row: max(tag_fog(row['R_Time_by_Counter']), tag_fog(row['L_Time_by_Counter'])), axis=1)

    # Apply Task tags
    task_results = df.apply(lambda row: tag_task(row['R_Time_by_Counter'], row['L_Time_by_Counter']), axis=1)
    df[['Task']] = pd.DataFrame(task_results.tolist(), index=df.index)
    return df, FOG_ranges, Task_ranges

def save_tagged_insole_data(df_full, patient_num, T_num, study):
    # Save tagged insole data to CSV files (full and tasks-only)

    base_folder = fr'data\{study}\PRE_tagging_data'
    filename_base = f"{patient_num}_T{T_num}"

    full_path = os.path.join(base_folder, f"{filename_base}_full_tagging.csv")
    df_full.to_csv(full_path, index=False)

    task_only_path = os.path.join(base_folder, f"{filename_base}_tasks_only_tagging.csv")
    df_task_only = df_full[df_full['Task'].notna()]
    df_task_only.to_csv(task_only_path, index=False)

    return df_full, df_task_only

def split_tagged_data(df, patient_num, T_num, study):
    # Split tagged data into both legs, right-only, and left-only datasets

    both_legs_df = df[df['R_counter'].notna() & df['L_counter'].notna()].copy()
    right_only_df = df[df['R_counter'].notna() & df['L_counter'].isna()].copy()
    left_only_df = df[df['L_counter'].notna() & df['R_counter'].isna()].copy()

    # Columns to drop from all parts
    columns_to_drop = [
        'R_agreement', 'R_Left_idx_match', 'R_counter', 'R_time_ms', 'R_medication_state',
        'L_agreement', 'L_Right_idx_match', 'L_counter', 'L_time_ms', 'L_medication_state', 'delta_time'
    ]

     # Drop unnecessary columns
    for df_part in [both_legs_df, right_only_df, left_only_df]:
        for col in columns_to_drop:
            if col in df_part.columns:
                df_part.drop(columns=col, inplace=True)

    # Drop opposite side columns
    right_only_df = right_only_df.drop(columns=[col for col in right_only_df.columns if col.startswith('L_')])
    left_only_df = left_only_df.drop(columns=[col for col in left_only_df.columns if col.startswith('R_')])

    # Remove prefixes (R_ / L_) from column names
    right_only_df.columns = [col[2:] if col.startswith('R_') else col for col in right_only_df.columns]
    left_only_df.columns = [col[2:] if col.startswith('L_') else col for col in left_only_df.columns]

    # Save results to separate CSVs
    save_dir = fr'data\{study}\PRE_tagging_data'

    both_legs_df.to_csv(os.path.join(save_dir, 'PRE_both_only_task_tagging_data', f"{patient_num}_T{T_num}_both.csv"), index=False)
    right_only_df.to_csv(os.path.join(save_dir, 'PRE_right_only_task_tagging_data', f"{patient_num}_T{T_num}_right.csv"), index=False)
    left_only_df.to_csv(os.path.join(save_dir, 'PRE_left_only_task_tagging_data', f"{patient_num}_T{T_num}_left.csv"), index=False)

    print("Splitting the data into right leg, left leg, and both legs is complete.")
    print(f"Rows with two-legged data: {len(both_legs_df)}")
    print(f"Rows with right data only: {len(right_only_df)}")
    print(f"Rows with left data only: {len(left_only_df)}")

    return both_legs_df, right_only_df, left_only_df


def main(study, patient_num, T_num):
    # Main pipeline for filling counters, tagging events, and splitting datasets

    # Load combined data
    df_path = fr'data\{study}\PRE_combine_data\{patient_num}_T{T_num}_combine.csv'
    df = pd.read_csv(df_path, low_memory=False)

    # Fill counters for both sides
    df_filled = fill_counters_both(df, step_ms=10)

    # Save filled data
    save_path = fr'data\{study}\PRE_combine_data\{patient_num}_T{T_num}_combine_filled.csv'
    df_filled.to_csv(save_path, index=False)

    # Load annotations file
    annotation_pattern = fr'data\{study}\{patient_num}*\Annotations'
    annotation_path = glob.glob(os.path.join(annotation_pattern, 'all_annotations.csv'))[0]
    annotations = pd.read_csv(annotation_path, low_memory=False)

    # Tag data with FOG and task events
    tagging_df, _, _ = tagging_fog_and_tasks(annotations, df_filled)

    # Save tagged insole data (full and tasks-only)
    df_full, df_task_only = save_tagged_insole_data(tagging_df, patient_num, T_num, study)

    # Split into right, left, and both legs datasets
    both_legs_df, right_only_df, left_only_df = split_tagged_data(df_task_only, patient_num, T_num, study)

if __name__ == "__main__":
    # Read arguments from command line
    study = sys.argv[3]
    patient_num = sys.argv[1]
    T_num = sys.argv[2]

    # Run the main function
    main(study, patient_num, T_num)