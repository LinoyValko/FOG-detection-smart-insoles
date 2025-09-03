# FOG Detection Using Smart Insoles Data

**Final MSc Project – Data Science**  
**Holon Institute of Technology (HIT)**  
**By: Linoy Valko & Shahar Simantov**  
**Supervisors: Dr. Dmitry Goldstein, Yakir Menahem**  
**In collaboration with Ichilov Hospital**  
**Submitted: August 2025**

---

## Project Overview

This project focuses on the **automatic detection of Freezing of Gait (FOG)** episodes in **Parkinson’s Disease (PD)** patients using only **smart insole pressure data**.  
Unlike traditional approaches that rely on IMU sensors, this study explores the feasibility of using insole data exclusively — providing a **non-invasive, wearable, and daily-practical solution** for clinical and real-world monitoring.

**Key Research Question:**  
> What are the optimal techniques for detecting FOG episodes in PD patients **based solely on smart insole pressure data**?

---

## Clinical Background

- **Freezing of Gait (FOG)** is a sudden and brief inability to walk, common in PD patients.
- It increases the risk of falls, limits independence, and significantly impairs quality of life.
- Existing detection methods rely heavily on **IMUs**, **manual annotation**, or clinical tests — often impractical for daily use.
- **Smart insoles** offer a compelling alternative for continuous, non-intrusive monitoring.

For full theoretical background, see [Chapter A – Theoretical Background](ספר%20פרויקט%20גמר-%20לינוי%20ולקו%20ושחר%20סימן%20טוב.pdf).

---

## Repository Structure

The pipeline consists of six main scripts, each corresponding to a specific processing stage:

| Stage | Filename | Description |
|-------|----------|-------------|
| 1 | `load_data_1.py` | Loads, cleans, clips, and normalizes pressure data for each patient-foot-trial combination, including timestamp reconstruction and medication state extraction. |
| 2 | `acc_data.py` | Loads and cleans 3D acceleration signals from foot-worn sensors, resamples them, and prepares them for synchronization. |
| 3 | `pre_process.py` | Performs data balancing, overlap removal between FOG/non-FOG, and removes short FOG episodes. Outputs clean, labeled segments per patient. |
| 4 | `add_annotations.py` | Aligns manual annotations with insole data, adds task tags, corrects missing values, and generates aligned time segments. |
| 5 | `Features_4.py` | Computes Center of Pressure (COP), Ground Reaction Force (GRF), their velocities and accelerations, per side and window. |
| 6 | `pipline_1.py`, `pipline_like_article.py` | Implements model training (LSTM-based), feature selection, data preparation, and evaluation. Compares performance across variations. |

---

## Methodology

The data was collected from controlled FOG-inducing walking tests performed at **Ichilov Hospital's Gait Lab**, including both **medicated ("ON")** and **unmedicated ("OFF")** states. The processing pipeline includes:

1. **Data Preparation**: Filtering invalid sensor values, synchronizing timestamps, and segmenting into per-task trials.  
2. **Annotation Alignment**: Mapping textual annotations into time-windows, tagging FOG/non-FOG labels.  
3. **Feature Engineering**: Extracting biomechanical metrics like COP, GRF, derivatives, and z-score normalization.  
4. **Balancing and Cleaning**: Removing short FOG segments, marking FOG/no-FOG transitions, filtering overlapping episodes.  
5. **Model Training**: LSTM-based classification on time-windowed feature sequences with participant-based cross-validation.

---

## Results & Evaluation

The LSTM-based model was evaluated using a **Leave-One-Freezer-Out (LOFO)** validation strategy, distinguishing between three classes: **FOG**, **Non-FOG**, and **Overlap**.

### Performance Metrics

| Metric         | Value   | Standard Deviation |
|----------------|---------|--------------------|
| **Sensitivity** | 84.0%   | ±13.2%              |
| **Specificity** | 93.6%   | ±4.9%               |

- **Architecture**: 2 LSTM layers × 16 units, full-sequence output
- **Classifier**: Softmax (3 classes: FOG / Non-FOG / Overlap)
- **Loss**: Sparse categorical cross-entropy
- **Optimizer**: Adam (initial LR: 0.01, halved every 5 epochs)
- **Regularization**: Gradient clipping
- **Batch size**: 1 (no padding)
- **Normalization**: Z-score (on training set only)

### Key Insights

- Performance is **comparable** to multi-sensor architectures (e.g., IMU + video).
- Overlap zones around FOG transitions (±11ms) were labeled as `0.5` to reduce ambiguity.
- Despite relying solely on **insole pressure data**, the model achieved **high accuracy** and **robust generalization**.

---

## Conclusions & Future Work

### Summary

- Developed a complete end-to-end pipeline for **FOG detection using smart insoles only**
- Achieved strong classification performance
- Demonstrated potential for **non-invasive**, **low-cost**, and **daily-use** monitoring of Parkinson’s patients

### Limitations

- Small sample size: 13 participants (4 excluded due to quality)
- Controlled lab environment only (no real-world data)
- Hardware limitations of insole pressure sensors
- Inter-subject variability affected generalization

### Future Directions

- Upgrade sensor hardware and sampling fidelity  
- Collect **real-world walking data** outside clinical settings  
- Increase cohort size and diversity  
- Implement **pre-FOG detection** for proactive intervention  
- Explore **advanced models** (e.g., transformers, attention)  
- Personalize detection with subject-specific fine-tuning  
- Improve handling of **missing data** using imputation techniques

---

## Sample Output

- Cleaned time-aligned data per patient and trial  
- Per-task annotation overlays  
- CSV files of extracted features  
- Summary plots for before/after cleaning  
- Model results (accuracy, confusion matrices)

---

## Technologies Used

- **Python 3.8+**
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `TensorFlow/Keras` for LSTM modeling
- `scikit-learn` for scaling and preprocessing

---

## Data Access

Due to privacy concerns, raw patient data are not included in this repository. 

---

