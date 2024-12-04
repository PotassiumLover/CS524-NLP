import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if not os.path.exists('graphs'):
    os.makedirs('graphs')

df = pd.read_csv('evaluation_results.csv')
numeric_cols = df.columns.drop('Name')
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x='Name', y='Accuracy', data=df, palette="Blues_d")
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.ylim(0, 1.05)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('graphs/model_accuracy_comparison.png')
plt.close()
metrics_other = df[['Name', 'Precision_Other Authors', 'Recall_Other Authors', 'F1-Score_Other Authors']]
metrics_other_melted = pd.melt(metrics_other, id_vars=['Name'], var_name='Metric', value_name='Score')
plt.figure(figsize=(10, 6))
sns.barplot(x='Name', y='Score', hue='Metric', data=metrics_other_melted, palette="Set2")
plt.title('Metrics for Other Authors')
plt.ylabel('Score')
plt.xlabel('Model')
plt.ylim(0, 1.05)
plt.xticks(rotation=45)
plt.legend(title='Metric')
plt.tight_layout()
plt.savefig('graphs/metrics_other_authors.png')
plt.close()
metrics_gk = df[['Name', 'Precision_G.K. Chesterton', 'Recall_G.K. Chesterton', 'F1-Score_G.K. Chesterton']]
metrics_gk_melted = pd.melt(metrics_gk, id_vars=['Name'], var_name='Metric', value_name='Score')
plt.figure(figsize=(10, 6))
sns.barplot(x='Name', y='Score', hue='Metric', data=metrics_gk_melted, palette="Set1")
plt.title('Metrics for G.K. Chesterton')
plt.ylabel('Score')
plt.xlabel('Model')
plt.ylim(0, 1.05)
plt.xticks(rotation=45)
plt.legend(title='Metric')
plt.tight_layout()
plt.savefig('graphs/metrics_gk_chesterton.png')
plt.close()