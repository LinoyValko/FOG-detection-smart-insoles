from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import save_file, uplaod_file, read_csv_skip_empty
import os
import sys


def calculate_COP_row(row, side):
    # Calculate Center of Pressure (COP) for a single row based on sensor positions
    if side in ['left', 'right']: # Single leg case
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
    else:  # Both legs case, prefixed with R_ and L_
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
        print(f"Error in calculating COP in line: {e}")
        return np.nan, np.nan

def calculate_velocity_and_acceleration_COP_row(df, side):
    # Calculate velocity and acceleration of COP for each axis (ml, ap)
    df = df.copy()

    if side in ['left', 'right']:  # Single leg case
        df['Time_by_Counter'] = pd.to_datetime(df['Time_by_Counter'], errors='coerce')
        df['Time_diff'] = df['Time_by_Counter'].diff().dt.total_seconds()

        for axis in ['ml', 'ap']:
            COP_col = f'COP_{axis}'

            if COP_col in df.columns:
                df[f'COP_Velocity_{axis}'] = df[COP_col].diff() / df['Time_diff']
                df[f'COP_Acceleration_{axis}'] = df[f'COP_Velocity_{axis}'].diff() / df['Time_diff']
            else:
                print(f"Missing column: {COP_col}")

        return df

    else:    # Both legs case
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
                    print(f"Missing column: {COP_col}")
        return df

def calculate_GRF_row(df, side):
    # Calculate Ground Reaction Force (GRF) for one or both legs
    sensors = ['hallux', 'toes', 'met1', 'met3', 'met5', 'arch', 'heelin', 'heelout']

    if side in ['left', 'right']:  # Single leg
        pressure_cols = sensors
        df['GRF'] = df[pressure_cols].sum(axis=1)

    else: # Both legs
        pressure_cols = {
            'L': [f'L_{sensor}' for sensor in ['hallux', 'toes', 'met1', 'met3', 'met5', 'arch', 'heelin', 'heelout']],
            'R': [f'R_{sensor}' for sensor in ['hallux', 'toes', 'met1', 'met3', 'met5', 'arch', 'heelin', 'heelout']]
        }

        df['L_GRF'] = df[pressure_cols['L']].sum(axis=1)
        df['R_GRF'] = df[pressure_cols['R']].sum(axis=1)

        # Relative GRF contributions
        df['L_fraction_GRF'] = df['L_GRF'] / (df['L_GRF'] + df['R_GRF'])
        df['R_fraction_GRF'] = df['R_GRF'] / (df['L_GRF'] + df['R_GRF'])

    return df


def extract_features_row(df, side):
     # Extract COP, velocity, acceleration, and GRF features
    if side == 'both':
        df['L_COP_ml'], df['L_COP_ap'] = zip(*df.apply(lambda row: calculate_COP_row(row, side), axis=1))
        df['R_COP_ml'], df['R_COP_ap'] = zip(*df.apply(lambda row: calculate_COP_row(row, side), axis=1))
    else: # Single leg case
        df['COP_ml'], df['COP_ap'] = zip(*df.apply(lambda row: calculate_COP_row(row, side), axis=1))

    df = calculate_velocity_and_acceleration_COP_row(df, side)
    df = calculate_GRF_row(df, side)

    return df


def main(patient_num, T_num, study):
    # Main pipeline for feature extraction from tagged insole data
    data_path = fr'data\{study}\PRE_tagging_data'
    save_path = fr'data\PRE_features'
    features_dfs = []
    sides = ['left', 'right', 'both']

    for side in sides:
        # Load tagged data
        data_folder = f'PRE_{side}_only_task_tagging_data'
        base_filename = f'{patient_num}_T{T_num}_{side}'
        save_data_folder = f'PRE_{side}_only_task_features'

        data = pd.read_csv(os.path.join(data_path, data_folder, base_filename + '.csv'))

        # Extract features for this side
        features_df = extract_features_row(data, side)

        # Keep only relevant columns
        columns_to_save = ['L_COP_ml', 'L_COP_ap', 'R_COP_ml', 'R_COP_ap', 'L_Time_diff', 'R_Time_diff',
                           'COP_ml', 'COP_ap', 'Time_diff', 'COP_Velocity_ml', 'COP_Acceleration_ml',
                           'COP_Velocity_ap', 'COP_Acceleration_ap', 'GRF',
                           'L_COP_Velocity_ml', 'L_COP_Acceleration_ml', 'R_COP_Velocity_ml', 'R_COP_Acceleration_ml',
                           'L_COP_Velocity_ap', 'L_COP_Acceleration_ap', 'R_COP_Velocity_ap', 'R_COP_Acceleration_ap',
                           'L_GRF', 'R_GRF', 'L_fraction_GRF', 'R_fraction_GRF', 'Task', 'State', 'FOG']
        columns_to_save = [col for col in columns_to_save if col in features_df.columns]

        # Save features for this side
        features_dfs.append(features_df[columns_to_save])
        features_df[columns_to_save].to_csv(os.path.join(save_path, save_data_folder, base_filename + '_features.csv'), index=False)

    # Merge left and right features into one dataset
    features_one_leg = pd.concat(features_dfs[:2], axis=0, ignore_index=True)
    one_leg_save_path = os.path.join(save_path, 'PRE_one_leg_only_task_features', f'{patient_num}_T{T_num}_one_leg_features.csv')
    features_one_leg.to_csv(one_leg_save_path, index=False)


if __name__ == '__main__':
    # Read arguments from command line
    study = sys.argv[3]
    patient_num = sys.argv[1]
    T_num = sys.argv[2]

     # Run pipeline
    main(patient_num, T_num, study)



