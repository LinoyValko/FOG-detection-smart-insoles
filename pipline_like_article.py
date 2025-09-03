import os, glob, re
import sys
import seaborn as sns

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


# Delete n first rows if exist NAN/ INF
def drop_first_n_if_nan(df: pd.DataFrame, feature_cols, n=2):
    """
    מחפש NaN/Inf בשורות הראשונות (0 ... n‑1) בטורים feature_cols.
    אם נמצא – מוחק אותן ומדפיס התראה.
    """
    bad_mask = (~np.isfinite(df.loc[:n-1, feature_cols])).any(axis=1)
    if bad_mask.any():
        # print(f'⚠️  Found NaN/Inf in first {n} rows — dropping {bad_mask.sum()} rows')
        df = df.drop(index=df.index[:n][bad_mask]).reset_index(drop=True)
    return df

# Prepare_data_&_save_summary_instances_data
def load_data(side):
    features_keywords = ['Time_diff', 'COP_ml', 'COP_ap', 'COP_Velocity_ml', 'COP_Acceleration_ml', 'COP_Velocity_ap',
                         'COP_Acceleration_ap', 'GRF']
    trials, labels, subject_ids = [], [], []

    features_path = fr'data\PRE_features\PRE_{side}_only_task_features'
    pattern = os.path.join(features_path, '*_features.csv')

    for file_name in glob.glob(pattern):
        df = pd.read_csv(file_name)
        features_columns = [c for c in df.columns if any(k in c for k in features_keywords)]
        df = drop_first_n_if_nan(df, features_columns, n=2)

        X_df = df.loc[:, features_columns].astype('float32')
        y_sr = df['FOG'].astype('int32')

        trials.append(X_df)
        labels.append(y_sr)

        subject = '_'.join(os.path.basename(file_name).split('_')[:4])
        subject_ids.append(subject)

    print(f'For {side} data, loaded {len(trials)} trials from {len(set(subject_ids))} participants')
    return trials, labels, subject_ids

# Create training balance segments with 1:1 balancing
# def balance_segments(X, y):
#     balanced = []
#     fog_idx = np.where(y == 1)[0]
#     if fog_idx.size == 0:
#         return balanced
#
#     seg_breaks = np.where(np.diff(fog_idx) > 1)[0] + 1
#     fog_segments = np.split(fog_idx, seg_breaks)
#
#     T = len(y)
#     for seg in fog_segments:
#         start, end = seg[0], seg[-1]
#         dur = end - start + 1
#         need_pre, need_post = dur // 2, dur - dur // 2
#
#         pre_start = max(start - need_pre, 0)
#         post_end  = min(end + need_post + 1, T)
#
#         # השלמות קצה לשמירת יחס 1:1
#         have_pre  = start - pre_start
#         have_post = post_end - end - 1
#         missing   = dur - (have_pre + have_post)
#
#         if missing:
#             extra_post = min(missing, T - post_end)
#             post_end  += extra_post
#             missing   -= extra_post
#         if missing:
#             extra_pre  = min(missing, pre_start)
#             pre_start -= extra_pre
#             missing   -= extra_pre
#         # אם עדיין חסר → דלג
#         if missing:
#             continue
#
#         xs = X[pre_start:post_end]
#         ys = y[pre_start:post_end]
#
#         n0 = np.sum(ys == 0)
#         n1 = np.sum(ys == 1)
#         if n0 != n1:
#             continue           # לא מאוזן? דלג.
#
#         balanced.append((xs, ys))
#
#     return balanced
#
# # Build 2 data dict: balance for training & raw for validation
# def prepare_datasets(trials, labels, subject_ids):
#     train_balance   = defaultdict(list)
#     raw_validation  = defaultdict(list)
#
#     for X, y, pid in zip(trials, labels, subject_ids):
#         raw_validation[pid].append((X, y))
#
#         balance_segs = balance_segments(X, y)
#         if balance_segs:
#             train_balance[pid].extend(balance_segs)
#
#     return train_balance, raw_validation

def mark_overlap_in_dataframe(df, window=11):
    """
    מסמן אזורי מעבר בין 0 ל-1 בעמודת FOG בתוך DataFrame, ע"י סימון הערכים כ־-1.
    """
    df = df.copy()
    df['FOG_orig'] = df['FOG']  # שימור העמודה המקורית
    print(df.columns)
    y = df['FOG'].values
    transition_points = np.where(np.diff(y) != 0)[0]

    for idx in transition_points:
        start = max(0, idx - window + 1)
        end = min(len(df), idx + window + 1)
        df.loc[start:end, 'FOG'] = 0.5
    return df

def get_segments(y, value=1):
    """
    מזהה קבוצות רציפות של value בתוך וקטור.
    """
    idx = np.where(y == value)[0]
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    return np.split(idx, breaks)

def clean_overlap_and_summarize(df, subject_id, min_fog_length=26):
    """
    מסיר חפיפות (FOG==-1) ומוחק אפיזודות FOG קצרות מ-min_fog_length.
    מחזיר את הדאטה לאחר הסינון + סיכום מצומצם.
    """
    df = df.copy()

    # --- "לפני": לפי המקור אם יש, אחרת לפי FOG הנוכחי ---
    if 'FOG_orig' in df.columns:
        before_series = df['FOG_orig'].to_numpy()
    else:
        # אם אין FOG_orig, נחשב 'לפני' לפי המצב הנוכחי, מתעלמים מ- -1 כסתם לא-Fog
        before_series = df['FOG'].replace(-1, 0).to_numpy()

    fog_rows_before    = int((before_series == 1).sum())
    nofog_rows_before  = int((before_series == 0).sum())
    fog_segs_before    = len(get_segments(before_series, value=1))

    # --- הסרת חפיפות ---
    df_clean = df[df['FOG'] != 0.5].reset_index(drop=True)
    final_fog = df_clean['FOG'].to_numpy()

    # --- "אחרי" ---
    fog_rows_after    = int((final_fog == 1).sum())
    nofog_rows_after  = int((final_fog == 0).sum())
    fog_segs_after    = len(get_segments(final_fog, value=1))

    # ניקוי העמודה העזר (לא חובה)
    if 'FOG_orig' in df_clean.columns:
        df_clean = df_clean.drop(columns='FOG_orig')

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
    summaries = []
    clean_dfs = []
    processed = {}
    for X, y, pid in zip(trials, labels, subject_ids):
        df = pd.DataFrame(X)
        df['FOG'] = y

        marked_overlap_df =  mark_overlap_in_dataframe(df)
        processed[pid] = marked_overlap_df
        print(marked_overlap_df.columns)
        clean_df, summary = clean_overlap_and_summarize(marked_overlap_df, pid)
        clean_dfs.append(clean_df)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df['Participant'] = np.arange(1, len(summary_df) + 1)
    summary_path = os.path.join(save_dir, 'data_cleaning_overlaps_and_short_fogs_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    # ---------- גרף 1: FOG שורות לפני ואחרי ----------
    # plt.figure(figsize=(10, 5))
    # sns.barplot(data=summary_df, x="Subject", y="FOG_rows_before", color="salmon", label="Before")
    # sns.barplot(data=summary_df, x="Subject", y="FOG_rows_after", color="seagreen", label="After")
    # plt.title("FOG Rows: Before vs After Cleaning")
    # plt.xticks(rotation=45)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "fog_rows_before_after.png"))
    # plt.close()

    # ---------- גרף 2: NOFOG שורות לפני ואחרי ----------
    # plt.figure(figsize=(10, 5))
    # sns.barplot(data=summary_df, x="Subject", y="NOFOG_rows_before", color="lightblue", label="Before")
    # sns.barplot(data=summary_df, x="Subject", y="NOFOG_rows_after", color="mediumblue", label="After")
    # plt.title("NOFOG Rows: Before vs After Cleaning")
    # plt.xticks(rotation=45)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "nofog_rows_before_after.png"))
    # plt.close()

    # ---------- גרף 3: אחוז שורות שהוסרו ----------
    # plt.figure(figsize=(10, 5))
    # sns.barplot(data=summary_df, x="Subject", y="Percent_rows_removed_total", color="gray")
    # plt.title("Total % of Rows Removed per Subject")
    # plt.ylabel("Percent Removed")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "percent_rows_removed.png"))
    # plt.close()

    # ---------- גרף 4: כמות אפיזודות קצרות שהוסרו ----------
    # plt.figure(figsize=(10, 5))
    # sns.barplot(data=summary_df, x="Subject", y="Removed_short_fog_segments", color="orange")
    # plt.title("Short FOG Segments Removed per Subject")
    # plt.ylabel("Removed Segments")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "short_fog_segments_removed.png"))
    # plt.close()

    return clean_dfs, processed, summary_df


# Normalization Z-score
def zscore_normalise(train_Xs, val_Xs):
    concat = np.concatenate([x for xs, _ in train_Xs for x in [xs]], axis=0)

    scaler = StandardScaler()
    scaler.fit(concat)

    def norm(seq):
        return scaler.transform(seq)

    norm_train = [(norm(x), y) for x, y in train_Xs]
    norm_val = [(norm(x), y) for x, y in val_Xs]

    return norm_train, norm_val

# LSTM model
def make_model(n_layers=2, units=16, n_feat=16):
    INIT_LR = 1e-2
    inp = Input(shape=(None, n_feat))
    x = inp
    for _ in range(n_layers):
        x = LSTM(units, return_sequences=True, activation='tanh')(x)
    out = TimeDistributed(Dense(3, activation='softmax'))(x)  # *** 3 מחלקות ***
    model = Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
    )
    return model


def lr_schedule(epoch, lr):
    HALVE_EVERY = 5                # epochs
    if epoch and epoch % HALVE_EVERY == 0:
        return lr * 0.5
    return lr

# Metric calculation
def evaluate_sequence(model, val_Xy):
    tp=tn=fp=fn=0
    for X,y in val_Xy:
        pred = model.predict(X[None,...], verbose=0)[0].argmax(-1)
        mask = (y==0) | (y==1)           # מתעלמים מ-0.5 במדדים בינאריים
        if not np.any(mask):
            continue
        y_use   = y[mask].astype(int)    # 0/1 בלבד
        pred_use= pred[mask]             # תחזית מחושבת מתוך 3 מחלקות (0/1/2) → כאן 0/1
        tp += np.sum((pred_use==1) & (y_use==1))
        tn += np.sum((pred_use==0) & (y_use==0))
        fp += np.sum((pred_use==1) & (y_use==0))
        fn += np.sum((pred_use==0) & (y_use==1))

    eps=1e-6
    sens = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    prec = tp / (tp + fp + eps)
    f1   = 2 * prec * sens / (prec + sens + eps)
    return sens, spec, prec, f1



def save_training_plot(history, test_pid, side, run_dir):
    plt.figure(figsize=(12, 4))
    plt.suptitle(f'Training Progress – {test_pid} ({side})')

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # שמירה לקובץ במקום הצגה
    filename = f"train_plot_{test_pid}_{side}.png"
    full_path = os.path.join(run_dir, filename)
    plt.savefig(full_path)
    plt.close()
def save_lofo_summary(results, side, run_dir):
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

    # ממוצעים וסטיות תקן (גלם)
    means = df_results[percent_cols + [f1_col]].mean()
    sds = df_results[percent_cols + [f1_col]].std(ddof=1)

    # טבלת תצוגה בפורמט כמו בתמונה
    df_disp = df_results.copy()
    for c in percent_cols:
        df_disp[c] = (df_disp[c] * 100).map(lambda v: f"{v:.1f}")
    df_disp[f1_col] = df_disp[f1_col].map(lambda v: f"{v:.2f}")

    mean_sd_row = {
        'Participant held out': 'Mean (SD)',
        'PID': ''
    }
    for c in percent_cols:
        mean_sd_row[c] = f"{means[c] * 100:.1f} ± {sds[c] * 100:.1f}"
    mean_sd_row[f1_col] = f"{means[f1_col]:.2f} ± {sds[f1_col]:.2f}"

    df_disp = pd.concat([df_disp, pd.DataFrame([mean_sd_row])], ignore_index=True)

    # הדפסה נאה ושמירה
    print(df_disp.to_string(index=False))

    csv_path = os.path.join(run_dir, f"lofo_results_{side}.csv")
    df_disp.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")



from typing import Dict, List, Optional, Tuple

def build_contiguous_segments_for_lofo(
    data_by_pid: Dict[str, pd.DataFrame],
    label_col: str = "FOG",
    time_diff_col: str = "R_Time_diff",
    time_unit: str = "ms",          # "ms" או "sec"
    expected_step_ms: float = 10.0, # צעד צפוי בין דגימות
    tol_ms: float = 1.0,            # טולרנס לשונות קטנה
    min_points: int = 1,            # מינימום נק' במקטע
    min_duration_ms: float = 0.0,   # מינימום משך מקטע
    feature_cols: Optional[List[str]] = None,
    y_dtype: str = "int32"          # "int32" אם 0/1; "float32" אם יש גם 0.5
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    מחזיר continues_segments בפורמט:
      { pid : [ (X[T,F], y[T]), (X,y), ... ] }
    מייצר מקטע חדש רק כשיש שבר ברצף הזמן (לפי expected_step_ms+tol_ms).
    """

    def _to_ms(series: pd.Series) -> np.ndarray:
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        return arr if time_unit.lower() in ("ms","millisecond","milliseconds") else (arr * 1000.0)

    out: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    for pid, df in data_by_pid.items():
        print(df.columns)
        if time_diff_col not in df.columns:
            raise ValueError(f"{pid}: missing '{time_diff_col}'")
        if label_col not in df.columns:
            raise ValueError(f"{pid}: missing '{label_col}'")

        # בחירת פיצ'רים
        if feature_cols is None:
            exclude = {label_col, time_diff_col}
            feat_cols = [c for c in df.columns if c not in exclude]
        else:
            feat_cols = feature_cols

        # וקטור dt במילישניות + זמן מצטבר (לנוחות סינון לפי משך)
        dt_ms = _to_ms(df[time_diff_col])
        time_ms = np.nan_to_num(dt_ms, nan=expected_step_ms).cumsum()

        # שברי זמן: כל מקום שבו dt גדול מהצעד+טולרנס או NaN
        breaks = np.where((dt_ms[1:] > (expected_step_ms + tol_ms)) | np.isnan(dt_ms[1:]))[0] + 1
        starts = np.r_[0, breaks]
        ends   = np.r_[breaks - 1, len(df) - 1]

        segments_xy: List[Tuple[np.ndarray, np.ndarray]] = []
        X_all = df[feat_cols].to_numpy(dtype=float)
        y_all = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=float)

        for s, e in zip(starts, ends):
            n_points_seg = e - s + 1
            duration_ms = max(0.0, time_ms[e] - (time_ms[s-1] if s > 0 else 0.0))

            if n_points_seg < min_points or duration_ms < min_duration_ms:
                continue

            X_seg = X_all[s:e+1]
            y_seg = y_all[s:e+1]

            # יציקת dtype ללייבלים לפי הצורך של ה-lofo_training
            if y_dtype == "int32":
                y_seg = y_seg.astype(np.int32, copy=False)
            else:
                y_seg = y_seg.astype(np.float32, copy=False)

            segments_xy.append((X_seg, y_seg))

        out[pid] = segments_xy

    return out


import re

def filter_subjects(data, drop_ids, exact=False):
    """
    data: מילון בפורמט {pid: [(X,y), ...]}
    drop_ids: מזהים להסרה כמחרוזות
    exact: אם True – מסיר רק אם ה-pid שווה בדיוק לאחד המזהים (אחרי המרה למחרוזת).
           אם False – מסיר אם המזהה מופיע בתוך ה-pid (עם גבולות ספרה, כדי שלא יתפוס 0032).

    מחזיר: (data_filtered, removed_keys)
    """
    if exact:
        to_remove = {str(x) for x in drop_ids}
        keep = {pid: v for pid, v in data.items() if str(pid) not in to_remove}
        removed = [pid for pid in data if str(pid) in to_remove]
    else:
        # התאמה “בטוחה”: 003 לא יתפוס 0032. תופס כטוקן מספרי בתוך שם.
        pat = re.compile(rf'(?<!\d)({"|".join(map(re.escape, drop_ids))})(?!\d)')
        keep = {pid: v for pid, v in data.items() if not pat.search(str(pid))}
        removed = [pid for pid in data if pat.search(str(pid))]
    return keep, removed


# Leave‑One‑Freezer‑Out training loop
def lofo_training(continues_segments, summary_df, side, run_name):
    pids = list(continues_segments.keys())
    results = []

    timestamp = datetime.now().strftime("%Y-%m-%d")
    run_dir = os.path.join("runs", f"{timestamp}_{run_name}")
    os.makedirs(run_dir, exist_ok=True)

    for test_pid in pids:
        # Bulid train & validation data set
        train_Xy = [xy for pid in pids if pid != test_pid
                    for xy in continues_segments[pid]]
        val_Xy = continues_segments[test_pid]

        # Normalized data
        train_Xy, val_Xy = zscore_normalise(train_Xy, val_Xy)

        # Data‑Generators  (Batch=1, keep original length)
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

        # Model training
        model = make_model(n_layers=2, units=16, n_feat=num_features)
        history = model.fit(train_ds,
                  epochs=30,
                  batch_size=1,
                  validation_data=val_ds,
                  callbacks=[LearningRateScheduler(lr_schedule)],
                  verbose=1)

        # ----------  Save Plot learning graph ----------
        save_training_plot(history, test_pid, side, run_dir)

        # ----------  Evaluation on FULL raw sequence ----------
        test_id = summary_df[summary_df["Subject"] == test_pid]["Participant"].iloc[0]
        FOG_rows_after = summary_df[summary_df["Subject"] == test_pid]["FOG_rows_after"].iloc[0]
        NOFOG_rows_after = summary_df[summary_df["Subject"] == test_pid]["NOFOG_rows_after"].iloc[0]
        sens, spec, prec, f1 = evaluate_sequence(model, val_Xy)
        results.append((test_id, test_pid, FOG_rows_after, NOFOG_rows_after, sens, spec, prec, f1))
        print(f'PID {test_pid}: sensitivity={sens:.1%}, specificity={spec:.1%}')

    save_lofo_summary(results, side, run_dir)


def main(side, run_name):
    trials, labels, subject_ids = load_data(side)
    clean_dfs, data_by_pid, summary_df = pre_process_data(trials, labels, subject_ids)

    continues_segments = build_contiguous_segments_for_lofo(data_by_pid)

    filtered_segments, removed = filter_subjects(continues_segments, ('003', '008', '009', '014'))
    print('הוסרו:', removed)

    lofo_training(filtered_segments, summary_df, side, run_name)

if __name__ == "__main__":
    side = 'both'
    run_name = 'like_article'
    main(side, run_name)



