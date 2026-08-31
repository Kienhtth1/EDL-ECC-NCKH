import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

from config import (
    DATA_DIR, OUTPUTS_DIR, DATASET_CONFIGS,
    HIDDEN, DROPOUT, N_CHAINS, EPOCHS, LR, SEED,
    METRICS_NAMES, DEVICE
)
from dataset import load_arff
from evaluate import run_kfold_cv, print_dataset_report, generate_and_save_summary
from plots import plot_eda_charts, plot_dataset_visualizations, plot_overall_summary


def process_single_dataset(ds_name, cfg, dev=DEVICE):
    arff_path = DATA_DIR / cfg['file']
    if not arff_path.exists():
        print(f'\n[SKIP] {ds_name}: {arff_path} không tồn tại.')
        return None, None
    ds_out_dir = OUTPUTS_DIR / ds_name
    ds_out_dir.mkdir(parents=True, exist_ok=True)

    num_labels = cfg['num_labels']
    X, Y, label_names = load_arff(arff_path, num_labels)
    plot_eda_charts(Y, label_names, ds_name, ds_out_dir)
    X = StandardScaler().fit_transform(X).astype('float32')
    df_mean, df_std, unc_corr, unc_wrg, total_time = run_kfold_cv(
        X, Y, num_labels, ds_name, dev=dev, n_splits=5, seed=SEED
    )
    res = {
        mn: {m: round(float(df_mean.loc[mn, m]), 4) for m in METRICS_NAMES}
        for mn in ['BR', 'CC', 'EDL-ECC']
    }
    res_std = {
        mn: {m: round(float(df_std.loc[mn, m]), 4) for m in METRICS_NAMES}
        for mn in ['BR', 'CC', 'EDL-ECC']
    }
    plot_dataset_visualizations(df_mean, df_std, unc_corr, unc_wrg, ds_name, ds_out_dir)
    print_dataset_report(res, total_time, ds_name, ds_out_dir)

    return res, res_std


def run():
    all_results = {}
    all_std_results = {}

    for ds_name, cfg in DATASET_CONFIGS.items():
        res, res_std = process_single_dataset(ds_name, cfg, dev=DEVICE)
        if res is not None:
            all_results[ds_name] = res
            all_std_results[ds_name] = res_std

    if all_results:
        plot_overall_summary(all_results, OUTPUTS_DIR)
        generate_and_save_summary(all_results, all_std_results, OUTPUTS_DIR)


if __name__ == '__main__':
    run()
