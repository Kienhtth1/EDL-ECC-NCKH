import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import METRICS_NAMES


def plot_eda_charts(Y, label_names, ds_name, out_dir):
    num_labels = len(label_names)
    label_freq = Y.sum(axis=0)
    label_ratio = label_freq / len(Y)

    # 1. Bar chart phân bố nhãn
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    palette_1 = sns.color_palette('husl', num_labels)
    palette_2 = sns.color_palette('coolwarm', num_labels)

    axes[0].bar(label_names, label_freq, color=palette_1, edgecolor='black')
    axes[0].set_title(f'Tần suất Xuất hiện Nhãn - {ds_name}', fontweight='bold')
    axes[0].set_xlabel('Nhãn')
    axes[0].set_ylabel('Số lượng mẫu')
    axes[0].tick_params(axis='x', rotation=30)

    axes[1].bar(label_names, label_ratio * 100, color=palette_2, edgecolor='black')
    axes[1].set_title(f'Tỷ lệ Nhãn Dương (%) - {ds_name}', fontweight='bold')
    axes[1].set_xlabel('Nhãn')
    axes[1].set_ylabel('Tỷ lệ (%)')
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(out_dir / 'eda_label_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Heatmap ma trận tương quan nhãn
    fig, ax = plt.subplots(figsize=(max(6, num_labels * 0.7), max(5, num_labels * 0.6)))
    corr_matrix = np.corrcoef(Y.T)
    if np.isnan(corr_matrix).any():
        corr_matrix = np.nan_to_num(corr_matrix)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=label_names, yticklabels=label_names, ax=ax,
                center=0, vmin=-1, vmax=1)
    ax.set_title(f'Ma trận Tương quan Nhãn - {ds_name}', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / 'eda_label_correlation.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_grouped_bar_chart(df_mean, df_std, ds_name, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(METRICS_NAMES))
    width = 0.25
    colors = ['#1f77b4', '#ff7f0e', '#d62728']

    for idx_m, (m_name, color) in enumerate(zip(['BR', 'CC', 'EDL-ECC'], colors)):
        scores = [df_mean.loc[m_name, met] for met in METRICS_NAMES]
        stds   = [df_std.loc[m_name, met]  for met in METRICS_NAMES]
        ax.bar(x + idx_m * width, scores, width, label=m_name,
               color=color, edgecolor='black', alpha=0.85, yerr=stds, capsize=4)

    ax.set_xlabel('Chỉ số Đánh giá (5-Fold CV Mean ± Std)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Điểm số', fontsize=12, fontweight='bold')
    ax.set_title(f'So sánh BR vs CC vs EDL-ECC (5-Fold CV) — {ds_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(METRICS_NAMES, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / 'metrics_bar_chart.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_radar_chart(df_mean, ds_name, out_dir):
    angles = np.linspace(0, 2 * np.pi, len(METRICS_NAMES), endpoint=False).tolist() + [0]
    colors_radar = {'BR': '#1f77b4', 'CC': '#ff7f0e', 'EDL-ECC': '#d62728'}

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for m_name, color in colors_radar.items():
        scores = [df_mean.loc[m_name, met] for met in METRICS_NAMES] + [df_mean.loc[m_name, METRICS_NAMES[0]]]
        ax.plot(angles, scores, label=m_name, linewidth=2.5, color=color)
        ax.fill(angles, scores, alpha=0.12, color=color)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), METRICS_NAMES, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f'Radar Chart (5-Fold CV) — {ds_name}', size=14, y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / 'radar_chart.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_metrics_table(df_mean, ds_name, out_dir):
    df_table = df_mean.round(4).copy()
    df_table.insert(0, 'Model', df_table.index)
    df_table = df_table.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis('off')
    tbl = ax.table(cellText=df_table.values, colLabels=df_table.columns,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)

    # Highlight hàng EDL-ECC
    for j in range(len(df_table.columns)):
        tbl[(3, j)].set_facecolor('#FFE0E0')
        tbl[(3, j)].set_text_props(fontweight='bold')

    plt.title(f'Kết quả 5-Fold CV Mean — {ds_name}', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(out_dir / 'metrics_table.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Xuất CSV
    df_mean.round(4).to_csv(out_dir / 'dataset_metrics_table.csv')


def plot_uncertainty_distribution(fold_unc_correct, fold_unc_wrong, ds_name, out_dir):
    if not fold_unc_correct or not fold_unc_wrong:
        return
    mean_u_correct = float(np.mean(fold_unc_correct))
    mean_u_wrong   = float(np.mean(fold_unc_wrong))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(fold_unc_correct, bins=30, alpha=0.7, color='#2ca02c',
            label=f'Dự đoán ĐÚNG (n={len(fold_unc_correct)})', density=True)
    ax.hist(fold_unc_wrong,   bins=30, alpha=0.7, color='#d62728',
            label=f'Dự đoán SAI  (n={len(fold_unc_wrong)})', density=True)
    ax.axvline(mean_u_correct, color='#2ca02c', linestyle='--', linewidth=2,
               label=f'Mean u đúng = {mean_u_correct:.3f}')
    ax.axvline(mean_u_wrong,   color='#d62728', linestyle='--', linewidth=2,
               label=f'Mean u sai  = {mean_u_wrong:.3f}')
    ax.set_xlabel('Độ bất định Evidential (u)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mật độ', fontsize=12, fontweight='bold')
    ax.set_title(f'Phân bố Độ bất định EDL-ECC — {ds_name}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'uncertainty_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_dataset_visualizations(df_mean, df_std, fold_unc_correct, fold_unc_wrong, ds_name, out_dir):
    plot_grouped_bar_chart(df_mean, df_std, ds_name, out_dir)
    plot_radar_chart(df_mean, ds_name, out_dir)
    plot_metrics_table(df_mean, ds_name, out_dir)
    plot_uncertainty_distribution(fold_unc_correct, fold_unc_wrong, ds_name, out_dir)


def plot_overall_summary(all_results, out_dir):
    datasets = list(all_results.keys())
    if not datasets:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(datasets))
    width = 0.25

    br_f1  = [all_results[d]['BR']['Micro-F1'] for d in datasets]
    cc_f1  = [all_results[d]['CC']['Micro-F1'] for d in datasets]
    edl_f1 = [all_results[d]['EDL-ECC']['Micro-F1'] for d in datasets]

    ax.bar(x - width, br_f1,  width, label='BR', color='#1f77b4', edgecolor='black', alpha=0.85)
    ax.bar(x,         cc_f1,  width, label='CC', color='#ff7f0e', edgecolor='black', alpha=0.85)
    ax.bar(x + width, edl_f1, width, label='EDL-ECC', color='#d62728', edgecolor='black', alpha=0.85)

    ax.set_ylabel('Micro-F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Tổng hợp Đánh giá Toàn diện: So sánh Micro-F1 trên 9 Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha='right', fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / 'overall_micro_f1_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
