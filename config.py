import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        plt.style.use('default')


DATA_DIR = Path('./data')
OUTPUTS_DIR = Path('./outputs')
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
DATASET_CONFIGS = {
    'Scene':           {'file': 'Scene.arff',           'num_labels': 6},
    'Yeast':           {'file': 'Yeast.arff',           'num_labels': 14},
    'emotions':        {'file': 'emotions.arff',         'num_labels': 6},
    'HumanPseAAC':     {'file': 'HumanPseAAC.arff',     'num_labels': 14},
    'PlantPseAAC':     {'file': 'PlantPseAAC.arff',     'num_labels': 12},
    'GpositivePseAAC': {'file': 'GpositivePseAAC.arff', 'num_labels': 4},
    'VirusPseAAC':     {'file': 'VirusPseAAC.arff',     'num_labels': 6},
    'Water-quality':   {'file': 'Water-quality.arff',   'num_labels': 14},
    'CHD_49':          {'file': 'CHD_49.arff',          'num_labels': 6},
}

# ── Siêu tham số Huấn luyện & Đánh giá 
N_CHAINS      = 3
HIDDEN        = 256
DROPOUT       = 0.3
EPOCHS        = 100
BATCH_SIZE    = 32
LR            = 1e-3
SEED          = 42
METRICS_NAMES = ['1-HL', 'SubAcc', 'Micro-F1', 'Macro-F1', 'Jaccard']


def get_device():
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print(f"Device: {dev} (NVIDIA CUDA GPU)\n")
    else:
        try:
            import torch_directml
            dev = torch_directml.device()
            print(f"Device: {dev} (DirectML AMD/Intel GPU)\n")
        except ImportError:
            dev = torch.device('cpu')
            print("Device: cpu \n")
    return dev

DEVICE = get_device()
