import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
import joblib

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

print('Libraries loaded!')

df_normal = pd.read_csv('final-normal-data-set.csv')
df_attack = pd.read_csv('final-anormal-data-set.csv')


df_normal['label'] = 0
df_attack['label'] = 1

df = pd.concat([df_normal, df_attack], ignore_index=True)

print(f'Normal samples: {len(df_normal)}')
print(f'Attack samples: {len(df_attack)}')
print(f'Total samples: {len(df)}')
print(f'\nDataset shape: {df.shape}')
print(f'\nClass distribution:')
print(df['label'].value_counts())
print(f'\n0 = Normal traffic')
print(f'1 = Cryptomining attack traffic')

label_map = {0: 'Normal', 1: 'Cryptomining Attack'}
df['label_name'] = df['label'].map(label_map)


signal_columns = [
    'processcount_running',
    'processcount_sleeping',
    'processcount_thread',
    'processcount_total',
    'cpu_total',
    'cpu_user',
    'cpu_system',
    'load_min1',
    'load_min5',
    'load_min15',
    'mem_percent',
    'mem_used',
    'mem_free',
    'memswap_percent',
    'memswap_used'
]


def build_baseline_profile(normal_frame, columns):
    profile = {}
    for column in columns:
        if column in normal_frame.columns:
            series = normal_frame[column].dropna()
            if not series.empty:
                std = float(series.std(ddof=0))
                profile[column] = {
                    'mean': float(series.mean()),
                    'std': std if std > 0 else 1.0,
                    'p95': float(series.quantile(0.95)),
                    'min': float(series.min()),
                    'max': float(series.max())
                }
    return profile


baseline_profile = build_baseline_profile(df_normal, signal_columns)


def describe_case(row, baseline, top_n=4):
    scored_signals = []
    for column, stats in baseline.items():
        if column in row.index:
            value = float(row[column])
            deviation = (value - stats['mean']) / stats['std'] if stats['std'] else 0.0
            distance = abs(deviation)
            scored_signals.append((column, value, stats['mean'], stats['p95'], deviation, distance))

    scored_signals.sort(key=lambda item: item[5], reverse=True)
    top_signals = scored_signals[:top_n]

    triggers = []
    for column, value, mean, p95, deviation, _ in top_signals:
        if column.startswith('processcount_') and value > p95:
            triggers.append(f"{column} higher than usual ({value:.1f} vs baseline p95 {p95:.1f})")
        elif column.startswith('cpu_') and value > p95:
            triggers.append(f"{column} above the normal range ({value:.1f} vs baseline p95 {p95:.1f})")
        elif column.startswith('load_') and value > p95:
            triggers.append(f"{column} elevated ({value:.2f} vs baseline p95 {p95:.2f})")
        elif column == 'mem_percent' and value > p95:
            triggers.append(f"memory usage is high ({value:.1f}% vs baseline p95 {p95:.1f}%)")

    if not triggers and top_signals:
        for column, value, mean, p95, deviation, _ in top_signals[:2]:
            triggers.append(f"{column} deviates from baseline by {deviation:.1f} standard deviations")

    return top_signals, triggers


def build_local_summary(prediction_label, prediction_probability, top_signals, triggers):
    trigger_text = '; '.join(triggers) if triggers else 'the feature profile differs from the normal baseline'
    signal_text = ', '.join([f"{column}={value:.2f}" for column, value, *_ in top_signals]) if top_signals else 'no strong signals were identified'
    return (
        f"Risk level: {prediction_label}. Confidence: {prediction_probability:.1%}. "
        f"This case looks suspicious because {trigger_text}. "
        f"Top signals: {signal_text}. "
        f"Recommended action: review the running processes, check for sustained CPU or load spikes, and verify whether the host is performing unexpected background work."
    )


def generate_claude_report(case_payload):
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key or Anthropic is None:
        return None

    client = Anthropic(api_key=api_key)
    model_name = os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet-latest')

    prompt = f"""
You are writing a short security summary for an end user.

Return exactly 3 short paragraphs:
1. A plain-language summary of the case.
2. A personalized explanation of why the system thinks it may be cryptomining.
3. A few practical next steps.

Be careful and do not overstate certainty. Use the provided evidence only.

Case label: {case_payload['prediction_label']}
Model confidence: {case_payload['prediction_probability']:.1%}
Top signals:
{case_payload['signals']}

Observed triggers:
{case_payload['triggers']}

Baseline context:
{case_payload['baseline_context']}
""".strip()

    response = client.messages.create(
        model=model_name,
        max_tokens=300,
        temperature=0.2,
        messages=[{'role': 'user', 'content': prompt}]
    )

    return ''.join(block.text for block in response.content if hasattr(block, 'text')).strip()

df['label_name'].value_counts().plot(kind='bar', color=['#1D9E75','#D85A30'])
plt.title('Network traffic class distribution')
plt.xlabel('Traffic Type')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('chart1_distribution.png', dpi=150)
plt.show()
print('✓ Saved chart1_distribution.png')

label_col = 'label'
X = df.drop(columns=[label_col, 'timestamp'])
y = df[label_col]
X = X.select_dtypes(include='number')

leaky_cols = [
    # Hardware fingerprints
    'memswap_total', 'memswap_free', 'memswap_used',
    'memswap_percent', 'memswap_sin', 'memswap_sout',
    'fs_/_free', 'fs_/_size', 'fs_/_used', 'fs_/_percent',
    'mem_cached', 'mem_total', 'mem_free', 'mem_used',
    'mem_available', 'mem_active', 'mem_inactive',
    'mem_buffers', 'mem_shared', 'mem_percent',

    # Zero variance
    'cpu_guest', 'cpu_guest_nice', 'cpu_irq', 'cpu_steal',

    # Duplicate per-core columns
    'percpu_0_cpu_number', 'percpu_0_guest', 'percpu_0_guest_nice',
    'percpu_0_irq', 'percpu_0_steal', 'percpu_0_idle',
    'percpu_0_softirq', 'percpu_0_iowait', 'percpu_0_total',
    'percpu_0_system', 'percpu_0_user', 'percpu_0_nice',

    # Session fingerprints
    'network_lo_cumulative_tx', 'network_lo_cumulative_rx',
    'network_lo_cumulative_cx',

    # Partition specific
    'diskio_sda1_write_bytes', 'diskio_sda1_read_bytes',
    'diskio_sda1_time_since_update', 'diskio_sda_read_bytes',

    # Confirmed fingerprints
    'cpu_idle',
    'network_lo_time_since_update',
    'cpu_softirq'
]

X = X.drop(columns=[c for c in leaky_cols if c in X.columns])
X = X.fillna(X.median())

print(f"Features after dropping leaky cols: {X.shape[1]}")

from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Features: {X.shape[1]}')
print(f'Train: {len(X_train)} | Test: {len(X_test)}')

from sklearn.metrics import roc_auc_score

print("\nAUC for every remaining feature:")
aucs = []
for col in X_train.columns:
    auc = roc_auc_score(y_train, X_train[col])
    auc = max(auc, 1 - auc)
    aucs.append((col, auc))

for col, auc in sorted(aucs, key=lambda x: -x[1]):
    print(f"  {auc:.4f}  {col}")

print("\n[DEBUG 1] Constant columns (zero variance):")
constant_cols = [col for col in X_train.columns if X_train[col].nunique() == 1]
print(f"  Found {len(constant_cols)}: {constant_cols}")

print("\n[DEBUG 2] Perfect separators (zero overlap between normal/attack):")
perfect_cols = []
for col in X_train.columns:
    normal_vals = X_train[y_train == 0][col]
    attack_vals = X_train[y_train == 1][col]
    if normal_vals.max() < attack_vals.min() or attack_vals.max() < normal_vals.min():
        perfect_cols.append(col)
        print(f"      Normal  → min={normal_vals.min():.4f}, max={normal_vals.max():.4f}")
        print(f"      Attack  → min={attack_vals.min():.4f}, max={attack_vals.max():.4f}")
if not perfect_cols:
    print("  None found.")

print("\n[DEBUG 3] Per-feature AUC (>0.95 = likely data leakage):")
leaky_cols = []
for col in X_train.columns:
    try:
        auc = roc_auc_score(y_train, X_train[col])
        auc = max(auc, 1 - auc)
        if auc > 0.95:
            leaky_cols.append((col, auc))
    except:
        pass
if leaky_cols:
    for col, auc in sorted(leaky_cols, key=lambda x: -x[1]):
        print(f"{col}: AUC={auc:.4f}")
else:
    print("  None found.")

print("\n[DEBUG 4] mem_percent sanity check:")
if 'mem_percent' in df.columns:
    print(f"  min={df['mem_percent'].min()}, max={df['mem_percent'].max()}")
    print(f"  Values > 100: {(df['mem_percent'] > 100).sum()}")
    print(f"  Values < 0:   {(df['mem_percent'] < 0).sum()}")
else:
    print("  Column not present.")

print(X_train.columns.tolist())

print("\n[DEBUG 5] Hardware fingerprint check (mean per class):")
fingerprint_cols = ['mem_total', 'memswap_total', 'mem_free', 'mem_used']
for col in fingerprint_cols:
    if col in X_train.columns:
        normal_mean = X_train[y_train == 0][col].mean()
        attack_mean = X_train[y_train == 1][col].mean()
        print(f"  {col}: Normal={normal_mean:.0f}, Attack={attack_mean:.0f}")

print("\n[DEBUG 6] Top 10 features by class mean ratio (higher = more suspicious):")
ratios = []
for col in X_train.columns:
    n = X_train[y_train == 0][col].mean()
    a = X_train[y_train == 1][col].mean()
    ratio = max(n, a) / (min(abs(n), abs(a)) + 0.0001)
    ratios.append((col, n, a, ratio))
ratios.sort(key=lambda x: -x[3])
for col, n, a, r in ratios[:10]:
    print(f"  {col}: Normal={n:.4f}, Attack={a:.4f}, Ratio={r:.1f}x")

print("="*60)

most_variable = X_train.std().idxmax()
threshold = X_train[most_variable].mean() + X_train[most_variable].std()
baseline_preds = (X_test[most_variable] > threshold).astype(int)

baseline_acc = accuracy_score(y_test, baseline_preds)
print(f'Baseline accuracy: {baseline_acc:.3f}')

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
print('Decision Tree accuracy:', round(accuracy_score(y_test, dt_preds), 3))
print('Decision Tree F1:', round(f1_score(y_test, dt_preds, average='weighted'), 3))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print('Random Forest accuracy:', round(accuracy_score(y_test, rf_preds), 3))
print('Random Forest F1:', round(f1_score(y_test, rf_preds, average='weighted'), 3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ConfusionMatrixDisplay.from_predictions(y_test, dt_preds, ax=axes[0], cmap='Blues')
axes[0].set_title('Decision Tree')
ConfusionMatrixDisplay.from_predictions(y_test, rf_preds, ax=axes[1], cmap='Greens')
axes[1].set_title('Random Forest')
plt.tight_layout()
plt.savefig('chart2_confusion_matrices.png', dpi=150)
plt.show()
print('✓ Saved chart2_confusion_matrices.png')

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"{'Model':<20} | {'Accuracy':<10} | {'F1':<10} | {'Precision':<10} | {'Recall':<10}")
print("-"*80)
for name, preds in [('Baseline', baseline_preds), ('Decision Tree', dt_preds), ('Random Forest', rf_preds)]:
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    print(f"{name:<20} | {acc:<10.3f} | {f1:<10.3f} | {prec:<10.3f} | {rec:<10.3f}")
print("="*80)

import joblib
joblib.dump(rf, 'cryptomining_detector.pkl')


def build_case_payload(row, prediction_label, prediction_probability):
    top_signals, triggers = describe_case(row, baseline_profile)
    signals_text = '\n'.join(
        [
            f"- {column}: value={value:.2f}, baseline_mean={mean:.2f}, baseline_p95={p95:.2f}, z_score={deviation:.2f}"
            for column, value, mean, p95, deviation, _ in top_signals
        ]
    )
    baseline_context = '\n'.join(
        [
            f"- {column}: mean={stats['mean']:.2f}, p95={stats['p95']:.2f}, range={stats['min']:.2f} to {stats['max']:.2f}"
            for column, stats in list(baseline_profile.items())[:6]
        ]
    )
    return {
        'prediction_label': prediction_label,
        'prediction_probability': prediction_probability,
        'signals': signals_text,
        'triggers': '\n'.join(f'- {trigger}' for trigger in triggers) if triggers else '- no strong trigger exceeded baseline thresholds',
        'baseline_context': baseline_context,
        'top_signals': top_signals,
        'triggers_list': triggers,
    }


def summarize_case(row, prediction_label, prediction_probability):
    case_payload = build_case_payload(row, prediction_label, prediction_probability)
    claude_summary = None
    try:
        claude_summary = generate_claude_report(case_payload)
    except Exception as error:
        print(f"Claude summary unavailable: {error}")

    if claude_summary:
        return claude_summary

    return build_local_summary(
        prediction_label,
        prediction_probability,
        case_payload['top_signals'],
        case_payload['triggers_list']
    )


print("\n" + "="*80)
print("SAMPLE USER REPORT")
print("="*80)

sample_row = X_test.iloc[0]
sample_frame = pd.DataFrame([sample_row])
sample_probability = float(rf.predict_proba(sample_frame)[0][1])
sample_prediction = int(rf.predict(sample_frame)[0])
sample_label = 'Potential Cryptomining' if sample_prediction == 1 else 'Likely Normal Activity'
print(summarize_case(sample_row, sample_label, sample_probability))


print(f"Decision Tree depth: {dt.get_depth()}")
print(f"Decision Tree leaves: {dt.get_n_leaves()}")

# What features is each model using?
print("\nDecision Tree feature importances:")
for feat, imp in sorted(zip(X_train.columns, dt.feature_importances_), key=lambda x: -x[1]):
    print(f"  {imp:.4f}  {feat}")

print("\nRandom Forest feature importances:")
for feat, imp in sorted(zip(X_train.columns, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {imp:.4f}  {feat}")