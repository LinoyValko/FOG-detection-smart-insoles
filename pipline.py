import os, glob, re
import sys
import seaborn as sns
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Input
from tensorflow.keras.models import Model
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import (Input, LSTM, SpatialDropout1D,
                                     Dropout, TimeDistributed, Dense)
from tensorflow.keras.regularizers import l2
from collections import defaultdict
import tensorflow as tf
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# Delete n first rows if exist NAN/ INF
def drop_first_n_if_nan(df: pd.DataFrame, feature_cols, n=2):
    # Check first n rows for NaN/Inf in selected feature columns
    bad_mask = (~np.isfinite(df.loc[:n-1, feature_cols])).any(axis=1)
    if bad_mask.any():
        # Drop rows with bad values and reset index
        df = df.drop(index=df.index[:n][bad_mask]).reset_index(drop=True)
    return df


def load_data(side):
    # Define features of interest
    features_keywords = ['Time_diff', 'COP_ml', 'COP_ap', 'COP_Velocity_ml', 'COP_Acceleration_ml', 'COP_Velocity_ap',
                         'COP_Acceleration_ap', 'GRF']
    trials, labels, subject_ids = [], [], []

    # Look for feature files under PRE_features folder
    features_path = fr'data\PRE_features\PRE_{side}_only_task_features'
    pattern = os.path.join(features_path, '*_features.csv')

    for file_name in glob.glob(pattern):
         # Filter relevant feature columns
        df = pd.read_csv(file_name)
        features_columns = [c for c in df.columns if any(k in c for k in features_keywords)]
        # Clean first rows if they contain NaN/Inf
        df = drop_first_n_if_nan(df, features_columns, n=2)

        # Extract X (features) and y (FOG labels)
        X_df = df.loc[:, features_columns].astype('float32')
        y_sr = df['FOG'].astype('int32')

        trials.append(X_df)
        labels.append(y_sr)

        # Use part of filename as subject identifier
        subject = '_'.join(os.path.basename(file_name).split('_')[:4])
        subject_ids.append(subject)

    print(f'For {side} data, loaded {len(trials)} trials from {len(set(subject_ids))} participants')
    return trials, labels, subject_ids


def mark_overlap_in_dataframe(df, window=11):
    # Mark transitions between 0 and 1 in the FOG column as 0.5
    df = df.copy()
    df['FOG_orig'] = df['FOG']  # Keep original FOG column for later reference
    print(df.columns)
    y = df['FOG'].values
    transition_points = np.where(np.diff(y) != 0)[0] # indices where label changes

    for idx in transition_points:
        start = max(0, idx - window + 1) # mark a small window before transition
        end = min(len(df), idx + window + 1) # and after transition
        df.loc[start:end, 'FOG'] = 0.5   # set overlap marker
    return df

def get_segments(y, value=1):
    # Find continuous segments where y == value
    idx = np.where(y == value)[0]
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0] + 1 # points where sequence breaks
    return np.split(idx, breaks)

def clean_overlap_and_summarize(df, subject_id, min_fog_length=26):
    # Remove overlap regions (FOG == 0.5) and summarize before/after cleaning
    df = df.copy()

    # "Before": take original FOG column if it exists, otherwise use current
    if 'FOG_orig' in df.columns:
        before_series = df['FOG_orig'].to_numpy()
    else:
        # If no original column, replace overlap markers with 0 (not FOG)
        before_series = df['FOG'].replace(-1, 0).to_numpy()

    # Count rows and segments before cleaning
    fog_rows_before    = int((before_series == 1).sum())
    nofog_rows_before  = int((before_series == 0).sum())
    fog_segs_before    = len(get_segments(before_series, value=1))

    # Remove overlap rows (FOG == 0.5)
    df_clean = df[df['FOG'] != 0.5].reset_index(drop=True)
    final_fog = df_clean['FOG'].to_numpy()

    # Count rows and segments after cleaning
    fog_rows_after    = int((final_fog == 1).sum())
    nofog_rows_after  = int((final_fog == 0).sum())
    fog_segs_after    = len(get_segments(final_fog, value=1))

    # Drop helper column if it exists
    if 'FOG_orig' in df_clean.columns:
        df_clean = df_clean.drop(columns='FOG_orig')

    # Build summary dictionary
    summary = {
        "Subject": subject_id,
        "FOG_segments_before": fog_segs_before,
        "FOG_segments_after": fog_segs_after,
        "FOG_rows_before": fog_rows_before,
        "FOG_rows_after": fog_rows_after,
        "NOFOG_rows_before": nofog_rows_before,
        "NOFOG_rows_after": nofog_rows_after,
    }
    return df_clean, summary


def pre_process_data(trials, labels, subject_ids):
    save_dir = 'data/summary_data_process'
    summaries = []  # store summary stats for each subject
    clean_dfs = []  # store cleaned dataframes
    processed = {}  # store processed dataframes before cleaning
    
    for X, y, pid in zip(trials, labels, subject_ids):
        # Build dataframe for this subject
        df = pd.DataFrame(X)
        df['FOG'] = y

        # Mark transition overlaps (FOG=0.5)
        marked_overlap_df =  mark_overlap_in_dataframe(df)
        processed[pid] = marked_overlap_df
        print(marked_overlap_df.columns)

        # Clean overlaps and short FOG episodes, collect summary
        clean_df, summary = clean_overlap_and_summarize(marked_overlap_df, pid)
        clean_dfs.append(clean_df)
        summaries.append(summary)

    # Build summary dataframe across all subjects
    summary_df = pd.DataFrame(summaries)
    summary_df['Participant'] = np.arange(1, len(summary_df) + 1)
    # Save summary to CSV
    summary_path = os.path.join(save_dir, 'data_cleaning_overlaps_and_short_fogs_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    return clean_dfs, processed, summary_df

def zscore_normalise(train_Xs, val_Xs):
    # Concatenate all training sequences to compute mean/std
    concat = np.concatenate([x for xs, _ in train_Xs for x in [xs]], axis=0)

    scaler = StandardScaler()
    scaler.fit(concat)

    # Apply z-score normalization to each sequence
    def norm(seq):
        return scaler.transform(seq)

    norm_train = [(norm(x), y) for x, y in train_Xs]
    norm_val = [(norm(x), y) for x, y in val_Xs]

    return norm_train, norm_val

def make_model(n_layers=2, units=16, n_feat=16):
    # Build simple LSTM sequence model
    INIT_LR = 1e-2
    inp = Input(shape=(None, n_feat))
    x = inp
    for _ in range(n_layers):
        x = LSTM(units, return_sequences=True, activation='tanh')(x)
    out = TimeDistributed(Dense(3, activation='softmax'))(x)  # 3-class output
    model = Model(inp, out)

    # Compile with Adam optimizer and sparse categorical crossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
    )
    return model


def lr_schedule(epoch, lr):
    # Halve the learning rate every 5 epochs
    HALVE_EVERY = 5                
    if epoch and epoch % HALVE_EVERY == 0:
        return lr * 0.5
    return lr

def evaluate_sequence(model, val_Xy):
    # Evaluate model performance on validation sequences
    tp=tn=fp=fn=0

    for X,y in val_Xy:
        # Predict sequence labels (take argmax of class probs)
        pred = model.predict(X[None,...], verbose=0)[0].argmax(-1)
        mask = (y==0) | (y==1)          # Only evaluate 0/1 labels (ignore 0.5)
        if not np.any(mask):
            continue
        y_use   = y[mask].astype(int)   
        pred_use= pred[mask] 
        
        # Update confusion matrix counts           
        tp += np.sum((pred_use==1) & (y_use==1))
        tn += np.sum((pred_use==0) & (y_use==0))
        fp += np.sum((pred_use==1) & (y_use==0))
        fn += np.sum((pred_use==0) & (y_use==1))

    # Compute metrics with epsilon for stability
    eps=1e-6
    sens = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    prec = tp / (tp + fp + eps)
    f1   = 2 * prec * sens / (prec + sens + eps)
    return sens, spec, prec, f1


def save_training_plot(history, test_pid, side, run_dir):
    plt.figure(figsize=(12, 4))
    plt.suptitle(f'Training Progress – {test_pid} ({side})')

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # Save plot to file instead of showing
    filename = f"train_plot_{test_pid}_{side}.png"
    full_path = os.path.join(run_dir, filename)
    plt.savefig(full_path)
    plt.close()

def save_lofo_summary(results, side, run_dir):
    # Build dataframe with results from each test subject
    df_results = pd.DataFrame([
        {
            'Participant held out': test_id,
            "PID": pid,
            'FOG data': FOG_rows_after,
            'Non-FOG data': NOFOG_rows_after,
            "Sensitivity (%)": sens,
            "Specificity (%)": spec,
            "Precision (%)": prec,
            "F1 score": f1
        }
        for (test_id, pid, FOG_rows_after, NOFOG_rows_after, sens, spec, prec, f1) in results
    ])

    percent_cols = ["Sensitivity (%)", "Specificity (%)", "Precision (%)"]
    f1_col = "F1 score"

    # Compute mean and standard deviation across participants
    means = df_results[percent_cols + [f1_col]].mean()
    sds = df_results[percent_cols + [f1_col]].std(ddof=1)

    # Format values for display (percentages and F1 with decimals)
    df_disp = df_results.copy()
    for c in percent_cols:
        df_disp[c] = (df_disp[c] * 100).map(lambda v: f"{v:.1f}")
    df_disp[f1_col] = df_disp[f1_col].map(lambda v: f"{v:.2f}")

    # Add mean ± SD row
    mean_sd_row = {
        'Participant held out': 'Mean (SD)',
        'PID': ''
    }
    for c in percent_cols:
        mean_sd_row[c] = f"{means[c] * 100:.1f} ± {sds[c] * 100:.1f}"
    mean_sd_row[f1_col] = f"{means[f1_col]:.2f} ± {sds[f1_col]:.2f}"

    df_disp = pd.concat([df_disp, pd.DataFrame([mean_sd_row])], ignore_index=True)

    # Print nicely formatted results
    print(df_disp.to_string(index=False))

    # Save results to CSV
    csv_path = os.path.join(run_dir, f"lofo_results_{side}.csv")
    df_disp.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

def build_contiguous_segments_for_lofo(
    # Convert time_diff column to milliseconds
    data_by_pid: Dict[str, pd.DataFrame],
    label_col: str = "FOG",
    time_diff_col: str = "R_Time_diff",
    time_unit: str = "ms",          # "ms" or "sec"
    expected_step_ms: float = 10.0, # expected step between samples
    tol_ms: float = 1.0,            # tolerance for irregular sampling
    min_points: int = 1,            # minimum points per segment
    min_duration_ms: float = 0.0,   # minimum duration per segment
    feature_cols: Optional[List[str]] = None,
    y_dtype: str = "int32"          ## "int32" for binary labels, "float32" if 0.5 exists
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    # Build contiguous time-based segments for each participant (pid).
    # Returns: { pid: [(X[T,F], y[T]), ...] } where each segment is continuous in time.

    # Convert given time series column to milliseconds

    def _to_ms(series: pd.Series) -> np.ndarray:
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        return arr if time_unit.lower() in ("ms","millisecond","milliseconds") else (arr * 1000.0)

    out: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    for pid, df in data_by_pid.items():
        print(df.columns)

        # Ensure required columns exist
        if time_diff_col not in df.columns:
            raise ValueError(f"{pid}: missing '{time_diff_col}'")
        if label_col not in df.columns:
            raise ValueError(f"{pid}: missing '{label_col}'")

        # Select features (exclude label and time_diff unless specified explicitly)
        if feature_cols is None:
            exclude = {label_col, time_diff_col}
            feat_cols = [c for c in df.columns if c not in exclude]
        else:
            feat_cols = feature_cols

        # Convert time differences to ms and compute cumulative timeline
        dt_ms = _to_ms(df[time_diff_col])
        time_ms = np.nan_to_num(dt_ms, nan=expected_step_ms).cumsum()

        # Identify breaks in the timeline (discontinuities or NaN values)
        breaks = np.where((dt_ms[1:] > (expected_step_ms + tol_ms)) | np.isnan(dt_ms[1:]))[0] + 1
        starts = np.r_[0, breaks]
        ends   = np.r_[breaks - 1, len(df) - 1]

        segments_xy: List[Tuple[np.ndarray, np.ndarray]] = []
        X_all = df[feat_cols].to_numpy(dtype=float)
        y_all = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=float)

        for s, e in zip(starts, ends):
            # Segment length and duration
            n_points_seg = e - s + 1
            duration_ms = max(0.0, time_ms[e] - (time_ms[s-1] if s > 0 else 0.0))

            # Skip too short segments
            if n_points_seg < min_points or duration_ms < min_duration_ms:
                continue

            # Extract X (features) and y (labels) for the segment
            X_seg = X_all[s:e+1]
            y_seg = y_all[s:e+1]

            # Cast labels to correct dtype
            if y_dtype == "int32":
                y_seg = y_seg.astype(np.int32, copy=False)
            else:
                y_seg = y_seg.astype(np.float32, copy=False)

            segments_xy.append((X_seg, y_seg))

        # Store segments for this participant
        out[pid] = segments_xy

    return out

def filter_subjects(data, drop_ids, exact=False):
    # Filter out specific subjects (pids) from the dataset.
    if exact:
        to_remove = {str(x) for x in drop_ids}
        keep = {pid: v for pid, v in data.items() if str(pid) not in to_remove}
        removed = [pid for pid in data if str(pid) in to_remove]
    else:
        # Regex for safe partial match: matches numeric tokens, avoids substring issues
        pat = re.compile(rf'(?<!\d)({"|".join(map(re.escape, drop_ids))})(?!\d)')
        keep = {pid: v for pid, v in data.items() if not pat.search(str(pid))}
        removed = [pid for pid in data if pat.search(str(pid))]
    return keep, removed


def lofo_training(continues_segments, summary_df, side, run_name):
    # Perform Leave-One-Freezer-Out (LOFO) training:
    # For each subject (pid), hold them out as validation and train on all others.
    pids = list(continues_segments.keys())
    results = []

    # Create run directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d")
    run_dir = os.path.join("runs", f"{timestamp}_{run_name}")
    os.makedirs(run_dir, exist_ok=True)

    for test_pid in pids:
        # Build train & validation datasets
        train_Xy = [xy for pid in pids if pid != test_pid
                    for xy in continues_segments[pid]]
        val_Xy = continues_segments[test_pid]

        # Normalize with z-score (fit only on training set)
        train_Xy, val_Xy = zscore_normalise(train_Xy, val_Xy)

        # Convert to tf.data datasets (batch=1, keep original sequence lengths)
        num_features = train_Xy[0][0].shape[1]
        def gen_train():
            for X,y in train_Xy:
                yield X[None,...].astype('float32'), y[None,...].astype('float32')
        def gen_val():
            for X,y in val_Xy:
                yield X[None,...].astype('float32'), y[None,...].astype('float32')

        train_ds = (
            tf.data.Dataset.from_generator(
                gen_train,
                output_signature=(
                    tf.TensorSpec(shape=(1, None, num_features), dtype=tf.float32),
                    tf.TensorSpec(shape=(1, None), dtype=tf.float32)))
            .prefetch(tf.data.AUTOTUNE))

        val_ds = tf.data.Dataset.from_generator(
            gen_val,
            output_signature=(
                tf.TensorSpec(shape=(1, None, num_features), dtype=tf.float32),
                tf.TensorSpec(shape=(1, None), dtype=tf.float32)))

        # Train model
        model = make_model(n_layers=2, units=16, n_feat=num_features)
        history = model.fit(train_ds,
                  epochs=30,
                  batch_size=1,
                  validation_data=val_ds,
                  callbacks=[LearningRateScheduler(lr_schedule)],
                  verbose=1)

        # Save training progress plot
        save_training_plot(history, test_pid, side, run_dir)

        # Evaluate on full validation sequences
        test_id = summary_df[summary_df["Subject"] == test_pid]["Participant"].iloc[0]
        FOG_rows_after = summary_df[summary_df["Subject"] == test_pid]["FOG_rows_after"].iloc[0]
        NOFOG_rows_after = summary_df[summary_df["Subject"] == test_pid]["NOFOG_rows_after"].iloc[0]
        sens, spec, prec, f1 = evaluate_sequence(model, val_Xy)
        results.append((test_id, test_pid, FOG_rows_after, NOFOG_rows_after, sens, spec, prec, f1))
        print(f'PID {test_pid}: sensitivity={sens:.1%}, specificity={spec:.1%}')

    # Save aggregated results
    save_lofo_summary(results, side, run_dir)


def main(side, run_name):
    # 1. Load data and preprocess
    trials, labels, subject_ids = load_data(side)
    clean_dfs, data_by_pid, summary_df = pre_process_data(trials, labels, subject_ids)

    # 2. Build contiguous segments from time-series
    continues_segments = build_contiguous_segments_for_lofo(data_by_pid)

    # 3. Filter out problematic subjects
    filtered_segments, removed = filter_subjects(continues_segments, ('003', '008', '009', '014'))
    print('Removed:', removed)

    # 4. Run LOFO training
    lofo_training(filtered_segments, summary_df, side, run_name)

if __name__ == "__main__":
    side = 'both'
    run_name = 'like_article'
    main(side, run_name)



