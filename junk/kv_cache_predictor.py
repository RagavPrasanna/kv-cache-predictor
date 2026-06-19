# =============================================================
# ML-Based KV Cache Size Prediction for LLM Serving
# CSD204 OS Project - Preliminary Results
# Run this in Google Colab (Free T4 GPU)
# =============================================================

# ─────────────────────────────────────────────────────────────
# CELL 1: Install dependencies
# ─────────────────────────────────────────────────────────────
# Paste this in the first Colab cell and run it

"""
!pip install datasets lightgbm scikit-learn matplotlib seaborn pandas numpy -q
"""

# ─────────────────────────────────────────────────────────────
# CELL 2: Imports
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("All imports successful!")

# ─────────────────────────────────────────────────────────────
# CELL 3: Load ShareGPT Dataset
# ─────────────────────────────────────────────────────────────

from datasets import load_dataset

print("Loading ShareGPT dataset...")
dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered",
                        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
                        split="train")

print(f"Dataset loaded: {len(dataset)} conversations")
print(f"Example entry: {dataset[0]}")

# ─────────────────────────────────────────────────────────────
# CELL 4: Simulate KV Cache Block Usage
# (Since we don't have a real GPU/vLLM, we simulate
#  KV block allocation based on the paper's parameters)
# ─────────────────────────────────────────────────────────────

# Parameters from the vLLM paper (OPT-13B on A100)
BLOCK_SIZE = 16          # tokens per KV block (default in vLLM)
BYTES_PER_TOKEN = 800    # KB per token for OPT-13B
MAX_SEQ_LEN = 2048       # maximum sequence length

def simulate_kv_blocks(input_length, output_length, block_size=BLOCK_SIZE):
    """
    Simulate the number of KV blocks a request uses.
    KV cache stores K and V vectors for every token (input + output).
    Blocks are allocated in fixed sizes of 'block_size' tokens.
    """
    total_tokens = input_length + output_length
    # Ceiling division — always rounds up to next full block
    blocks_needed = int(np.ceil(total_tokens / block_size))
    return blocks_needed

def classify_task(text):
    """
    Simple rule-based task classifier.
    In a real system this would be more sophisticated.
    """
    text_lower = text.lower()
    if len(text) < 100:
        return "short_query"
    elif any(w in text_lower for w in ["translate", "translation", "french", "german", "spanish"]):
        return "translation"
    elif any(w in text_lower for w in ["```", "code", "function", "program", "python", "java"]):
        return "code"
    elif any(w in text_lower for w in ["explain", "what is", "how does", "describe"]):
        return "explanation"
    else:
        return "general_chat"

def extract_features(conversation, temperature=0.8, max_tokens=512):
    """
    Extract features from a conversation for the ML model.
    These are the features available BEFORE or EARLY in generation.
    """
    # Get the first human message as the prompt
    human_turns = [m for m in conversation['conversations'] if m['from'] == 'human']
    gpt_turns   = [m for m in conversation['conversations'] if m['from'] == 'gpt']

    if not human_turns or not gpt_turns:
        return None

    prompt = human_turns[0]['value']
    response = gpt_turns[0]['value']

    # Approximate token count (words * 1.3 is a rough estimate)
    input_length  = int(len(prompt.split()) * 1.3)
    output_length = int(len(response.split()) * 1.3)

    # Clip to max sequence length
    input_length  = min(input_length, MAX_SEQ_LEN)
    output_length = min(output_length, MAX_SEQ_LEN - input_length)

    if input_length < 1 or output_length < 1:
        return None

    task_type = classify_task(prompt)

    # Simulate KV blocks (ground truth)
    actual_blocks = simulate_kv_blocks(input_length, output_length)

    # Feature: question marks suggest shorter answers
    question_marks = prompt.count('?')

    # Feature: instruction words suggest longer answers
    instruction_words = sum(1 for w in ["explain", "describe", "write", "list", "analyze", "compare"]
                            if w in prompt.lower())

    return {
        'input_length'      : input_length,
        'output_length'     : output_length,   # ground truth — NOT used as feature
        'actual_blocks'     : actual_blocks,   # ground truth — what we predict
        'temperature'       : temperature,
        'max_tokens'        : max_tokens,
        'task_type'         : task_type,
        'question_marks'    : question_marks,
        'instruction_words' : instruction_words,
        'prompt_length_sq'  : input_length ** 2,  # captures non-linear relationship
    }

# ─────────────────────────────────────────────────────────────
# CELL 5: Build Dataset
# ─────────────────────────────────────────────────────────────

print("Extracting features from conversations...")

records = []
temperatures = [0.3, 0.5, 0.7, 0.8, 1.0, 1.2]  # vary temperature across requests

for i, conv in enumerate(dataset):
    # Assign random temperature to simulate varied requests
    temp = np.random.choice(temperatures)
    max_tok = np.random.choice([256, 512, 1024, 2048])

    features = extract_features(conv, temperature=temp, max_tokens=max_tok)
    if features:
        records.append(features)

    if i >= 5000:   # use first 5000 conversations — enough for a course project
        break

df = pd.DataFrame(records)
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nBasic statistics:")
print(df[['input_length', 'output_length', 'actual_blocks']].describe())

# ─────────────────────────────────────────────────────────────
# CELL 6: Exploratory Data Analysis (EDA)
# These plots go in your report!
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('KV Cache Usage Analysis — ShareGPT Dataset', fontsize=14, fontweight='bold')

# Plot 1: Input vs Output length distribution
axes[0, 0].hist(df['input_length'], bins=50, alpha=0.7, color='steelblue', label='Input')
axes[0, 0].hist(df['output_length'], bins=50, alpha=0.7, color='coral', label='Output')
axes[0, 0].set_xlabel('Token Length')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Input and Output Length Distributions')
axes[0, 0].legend()

# Plot 2: Input length vs KV blocks used (scatter)
axes[0, 1].scatter(df['input_length'], df['actual_blocks'],
                   alpha=0.3, s=5, color='steelblue')
axes[0, 1].set_xlabel('Input Length (tokens)')
axes[0, 1].set_ylabel('KV Blocks Used')
axes[0, 1].set_title('Input Length vs KV Blocks Used')

# Plot 3: KV blocks by task type
task_groups = df.groupby('task_type')['actual_blocks'].mean().sort_values(ascending=False)
axes[1, 0].bar(task_groups.index, task_groups.values, color='steelblue')
axes[1, 0].set_xlabel('Task Type')
axes[1, 0].set_ylabel('Average KV Blocks Used')
axes[1, 0].set_title('Average KV Blocks by Task Type')
axes[1, 0].tick_params(axis='x', rotation=15)

# Plot 4: KV blocks distribution
axes[1, 1].hist(df['actual_blocks'], bins=50, color='steelblue', edgecolor='white')
axes[1, 1].set_xlabel('KV Blocks Used')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Distribution of KV Blocks Used Per Request')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("EDA plots saved as eda_plots.png")

# ─────────────────────────────────────────────────────────────
# CELL 7: Prepare Features for ML Models
# ─────────────────────────────────────────────────────────────

# Encode task type as numbers
le = LabelEncoder()
df['task_type_encoded'] = le.fit_transform(df['task_type'])
print("Task type encoding:")
for i, cls in enumerate(le.classes_):
    print(f"  {i} = {cls}")

# Features used for prediction
# NOTE: output_length is NOT included — that's what we're trying to predict
FEATURE_COLS = [
    'input_length',
    'temperature',
    'max_tokens',
    'task_type_encoded',
    'question_marks',
    'instruction_words',
    'prompt_length_sq',
]

X = df[FEATURE_COLS]
y = df['actual_blocks']

# Train/test split — 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ─────────────────────────────────────────────────────────────
# CELL 8: Train Models
# ─────────────────────────────────────────────────────────────

results = {}  # store results for comparison

# ── Model 1: Linear Regression (Baseline) ──────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_mae   = mean_absolute_error(y_test, lr_preds)
lr_rmse  = np.sqrt(mean_squared_error(y_test, lr_preds))
results['Linear Regression'] = {'MAE': lr_mae, 'RMSE': lr_rmse, 'preds': lr_preds}
print(f"Linear Regression  — MAE: {lr_mae:.2f} blocks | RMSE: {lr_rmse:.2f} blocks")

# ── Model 2: Quantile Regression p50, p85, p95 ─────────────
qr_models = {}
for q in [0.50, 0.85, 0.95]:
    qr = QuantileRegressor(quantile=q, alpha=0.1, solver='highs')
    qr.fit(X_train, y_train)
    qr_preds = qr.predict(X_test)
    qr_mae   = mean_absolute_error(y_test, qr_preds)
    qr_rmse  = np.sqrt(mean_squared_error(y_test, qr_preds))
    qr_models[q] = qr
    results[f'Quantile Regression (p{int(q*100)})'] = {
        'MAE': qr_mae, 'RMSE': qr_rmse, 'preds': qr_preds
    }
    print(f"Quantile Reg p{int(q*100)}     — MAE: {qr_mae:.2f} blocks | RMSE: {qr_rmse:.2f} blocks")

# ── Model 3: LightGBM (Best model) ─────────────────────────
lgbm = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1
)
lgbm.fit(X_train, y_train)
lgbm_preds = lgbm.predict(X_test)
lgbm_mae   = mean_absolute_error(y_test, lgbm_preds)
lgbm_rmse  = np.sqrt(mean_squared_error(y_test, lgbm_preds))
results['LightGBM'] = {'MAE': lgbm_mae, 'RMSE': lgbm_rmse, 'preds': lgbm_preds}
print(f"LightGBM           — MAE: {lgbm_mae:.2f} blocks | RMSE: {lgbm_rmse:.2f} blocks")

# ─────────────────────────────────────────────────────────────
# CELL 9: Results Visualization
# These plots go in your report!
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Prediction Results', fontsize=14, fontweight='bold')

models_to_plot = ['Linear Regression', 'Quantile Regression (p85)', 'LightGBM']

for ax, model_name in zip(axes, models_to_plot):
    preds = results[model_name]['preds']
    mae   = results[model_name]['MAE']

    ax.scatter(y_test, preds, alpha=0.3, s=5, color='steelblue')

    # Perfect prediction line
    max_val = max(y_test.max(), preds.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')

    ax.set_xlabel('Actual KV Blocks')
    ax.set_ylabel('Predicted KV Blocks')
    ax.set_title(f'{model_name}\nMAE = {mae:.2f} blocks')
    ax.legend()

plt.tight_layout()
plt.savefig('prediction_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Prediction results saved as prediction_results.png")

# ─────────────────────────────────────────────────────────────
# CELL 10: Feature Importance (LightGBM)
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
importance = pd.Series(lgbm.feature_importances_, index=FEATURE_COLS)
importance.sort_values().plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Feature Importance — LightGBM Model')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Feature importance plot saved as feature_importance.png")

# ─────────────────────────────────────────────────────────────
# CELL 11: Simulate Scheduling Improvement
# This shows how prediction helps the scheduler
# ─────────────────────────────────────────────────────────────

def simulate_scheduler(requests_df, model_preds, available_blocks=200, strategy='predicted'):
    """
    Simulate how many requests get admitted into a batch.

    strategy='predicted' : use ML predictions to decide admission
    strategy='max'       : always assume max blocks (existing system)
    strategy='actual'    : use actual blocks (oracle — impossible in practice)
    """
    admitted = 0
    memory_used = 0
    preemptions = 0

    if strategy == 'predicted':
        # Sort by predicted size — pack smallest first
        order = np.argsort(model_preds)
    elif strategy == 'max':
        # Existing system — assumes every request needs max blocks
        order = np.arange(len(requests_df))
    else:  # oracle
        order = np.argsort(requests_df['actual_blocks'].values)

    for idx in order:
        actual = requests_df['actual_blocks'].iloc[idx]

        if strategy == 'predicted':
            estimated = model_preds[idx]
        elif strategy == 'max':
            estimated = MAX_SEQ_LEN // BLOCK_SIZE  # worst case
        else:
            estimated = actual  # oracle knows exact size

        if memory_used + estimated <= available_blocks:
            admitted += 1
            memory_used += actual  # actual memory consumed

            # Check if we ran out despite prediction saying it would fit
            if memory_used > available_blocks:
                preemptions += 1
                memory_used -= actual
                admitted -= 1
        else:
            break

    utilization = (memory_used / available_blocks) * 100
    return admitted, utilization, preemptions

# Run simulation on test set
test_df = df.iloc[X_test.index].reset_index(drop=True)
lgbm_preds_test = lgbm.predict(X_test).astype(int)

admitted_predicted, util_predicted, preempt_predicted = simulate_scheduler(
    test_df, lgbm_preds_test, strategy='predicted')
admitted_max, util_max, preempt_max = simulate_scheduler(
    test_df, None, strategy='max')
admitted_oracle, util_oracle, preempt_oracle = simulate_scheduler(
    test_df, test_df['actual_blocks'].values, strategy='oracle')

print("\n" + "="*55)
print("SCHEDULING SIMULATION RESULTS")
print("="*55)
print(f"{'Strategy':<25} {'Admitted':>10} {'Util%':>10} {'Preemptions':>12}")
print("-"*55)
print(f"{'Existing (Max Reserve)':<25} {admitted_max:>10} {util_max:>9.1f}% {preempt_max:>12}")
print(f"{'Ours (ML Predicted)':<25} {admitted_predicted:>10} {util_predicted:>9.1f}% {preempt_predicted:>12}")
print(f"{'Oracle (Perfect)':<25} {admitted_oracle:>10} {util_oracle:>9.1f}% {preempt_oracle:>12}")
print("="*55)

# ─────────────────────────────────────────────────────────────
# CELL 12: Scheduling Results Plot
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Scheduling Simulation Results', fontsize=13, fontweight='bold')

strategies  = ['Existing\n(Max Reserve)', 'Ours\n(ML Predicted)', 'Oracle\n(Perfect)']
admitted_vals = [admitted_max, admitted_predicted, admitted_oracle]
util_vals     = [util_max, util_predicted, util_oracle]
colors        = ['coral', 'steelblue', 'green']

# Plot 1: Requests admitted
bars1 = axes[0].bar(strategies, admitted_vals, color=colors, edgecolor='white', width=0.5)
axes[0].set_ylabel('Requests Admitted per Batch')
axes[0].set_title('Batch Admission (higher = better)')
for bar, val in zip(bars1, admitted_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(val), ha='center', fontweight='bold')

# Plot 2: Memory utilization
bars2 = axes[1].bar(strategies, util_vals, color=colors, edgecolor='white', width=0.5)
axes[1].set_ylabel('GPU Memory Utilization (%)')
axes[1].set_title('Memory Utilization (higher = better)')
axes[1].set_ylim(0, 110)
for bar, val in zip(bars2, util_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('scheduling_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Scheduling results saved as scheduling_results.png")

# ─────────────────────────────────────────────────────────────
# CELL 13: Quantile Coverage Analysis
# Shows how well p85 prediction covers actual usage
# ─────────────────────────────────────────────────────────────

p50_preds = qr_models[0.50].predict(X_test)
p85_preds = qr_models[0.85].predict(X_test)
p95_preds = qr_models[0.95].predict(X_test)
actual    = y_test.values

# Coverage = % of requests where prediction >= actual (no underallocation)
coverage_p50 = np.mean(p50_preds >= actual) * 100
coverage_p85 = np.mean(p85_preds >= actual) * 100
coverage_p95 = np.mean(p95_preds >= actual) * 100

# Over-allocation = how much extra memory we waste on average
overalloc_p50 = np.mean(np.maximum(p50_preds - actual, 0))
overalloc_p85 = np.mean(np.maximum(p85_preds - actual, 0))
overalloc_p95 = np.mean(np.maximum(p95_preds - actual, 0))

print("\n" + "="*55)
print("QUANTILE COVERAGE ANALYSIS")
print("="*55)
print(f"{'Quantile':<15} {'Coverage':>12} {'Avg Over-alloc':>16}")
print("-"*55)
print(f"{'p50 (median)':<15} {coverage_p50:>11.1f}% {overalloc_p50:>13.2f} blk")
print(f"{'p85':<15} {coverage_p85:>11.1f}% {overalloc_p85:>13.2f} blk")
print(f"{'p95':<15} {coverage_p95:>11.1f}% {overalloc_p95:>13.2f} blk")
print("="*55)
print("\np85 is the sweet spot — covers most requests with minimal waste")

# ─────────────────────────────────────────────────────────────
# CELL 14: Final Summary Table
# ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("FINAL RESULTS SUMMARY")
print("="*55)
print(f"\nDataset        : ShareGPT ({len(df)} conversations)")
print(f"Block size     : {BLOCK_SIZE} tokens/block")
print(f"Train/Test     : 80/20 split\n")

print("── Prediction Accuracy ──────────────────────────")
for name, res in results.items():
    print(f"  {name:<35} MAE = {res['MAE']:.2f} blocks")

print("\n── Scheduling Improvement ───────────────────────")
improvement = ((admitted_predicted - admitted_max) / admitted_max) * 100
print(f"  Requests admitted improvement : +{improvement:.1f}% over existing system")
util_improvement = util_predicted - util_max
print(f"  Memory utilization improvement: +{util_improvement:.1f}% over existing system")

print("\n── Key Finding ──────────────────────────────────")
print(f"  p85 quantile regression covers {coverage_p85:.1f}% of requests")
print(f"  with only {overalloc_p85:.2f} blocks average over-allocation")
print("="*55)
