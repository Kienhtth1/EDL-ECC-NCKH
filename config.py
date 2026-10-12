import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


# ── Kaggle / Local detection
IS_KAGGLE   = Path('/kaggle/working').exists()
OUTPUTS_DIR = Path('/kaggle/working/outputs') if IS_KAGGLE else Path('./outputs')
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR_CANDIDATES = [
    Path('/kaggle/input'),
    Path('./data'),
    Path('.'),
]


try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        plt.style.use('default')


DATA_DIR = Path('./data')
    
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
N_CHAINS      = 20
HIDDEN        = 256
DROPOUT       = 0.2
EPOCHS        = 100
BATCH_SIZE    = 64
LR            = 1e-3
SEED          = 42
THRESHOLD     = 0.5
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


def find_file(filename):
    """Tìm file data theo tên, duyệt đệ quy qua DATA_DIR_CANDIDATES."""
    target_lower = filename.lower()
    for base in DATA_DIR_CANDIDATES:
        if not base.exists():
            continue
        for p in base.rglob('*'):
            if p.is_file() and p.name.lower() == target_lower:
                return p
    return None
