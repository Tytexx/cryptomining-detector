import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
import joblib

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

df['label_name'].value_counts().plot(kind='bar', color=['#1D9E75','#D85A30'])
plt.title('Network traffic class distribution')
plt.xlabel('Traffic Type')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('chart1_distribution.png', dpi=150)
plt.show()
print('Saved chart1_distribution.png')

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
print('Saved chart2_confusion_matrices.png')

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

joblib.dump(rf, 'cryptomining_detector.pkl')


print(f"Decision Tree depth: {dt.get_depth()}")
print(f"Decision Tree leaves: {dt.get_n_leaves()}")

print("\nRandom Forest feature importances:")
for feat, imp in sorted(zip(X_train.columns, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {imp:.4f}  {feat}")