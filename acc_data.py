import matplotlib.pyplot as plt
import pandas as pd
import os
from load_data_1 import create_time_column
from utils import save_file, read_csv_skip_empty
import glob

def read_and_clean_acc(patient_num, T_num, study):

    # Define base directory for study data
    data_dir = f'data\{study}'

    # Find matching ACC folder using glob pattern
    folder_pattern = os.path.join(data_dir, f"{patient_num}_*_{T_num}", "ACC")
    matching_folders = glob.glob(folder_pattern)

    if not matching_folders:
        raise FileNotFoundError(f"No matching ACC folder found for: {folder_pattern}")

    # Use the first matching folder
    folder_path = matching_folders[0]
    files_name = os.listdir(folder_path)
    clean_dfs = []

    # Iterate over all ACC files in the folder
    for file_name in files_name:
        file_path = os.path.join(folder_path, file_name)

        # Load CSV and strip whitespace from column names
        df = pd.read_csv(file_path)
        df = df.rename(columns=lambda x: x.strip())

        # Create proper time column
        df_with_time_column = create_time_column(df)
        clean_dfs.append(df_with_time_column)

        # Save cleaned file to output folder
        save_file_name = file_name.replace('.csv', '_clean.csv')
        save_dir = os.path.join(data_dir, 'ACC_cleaned', patient_num)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_file_name)
        df_with_time_column.to_csv(save_path, index=False)

    return clean_dfs, files_name

def plot_acc(acc_dfs, files_name):

    # Plot acceleration data (X, Y, Z) for left side files
    for i, acc_df in enumerate(acc_dfs):
        if 'Left' in files_name[i]:
            plt.plot(acc_df['Time_by_Counter'], acc_df['acc_x'], label='X')
            plt.plot(acc_df['Time_by_Counter'], acc_df['acc_y'], label='Y')
            plt.plot(acc_df['Time_by_Counter'], acc_df['acc_z'], label='Z')
            plt.title(files_name[i])
            plt.xlabel('Time_by_Counter')
            plt.ylabel('Acceleration')
            plt.legend()
            plt.grid(True)

    plt.tight_layout()
    plt.show()

# Example patients and study
patient_num_FOG_COA = ['002_T1', '003_T0', '008_T0', '009_T1', '011_T0', '012_T0']
study = 'STEP_UP'

# Run preprocessing for each patient
for patient in patient_num_FOG_COA:
    patient_num = patient[:3] # extract patient ID
    T_num = patient[-2:]  # extract trial number
    acc_dfs, files_name = read_and_clean_acc(patient_num, T_num, study)


