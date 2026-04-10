# preset.py
import os

def _sfuda_paper_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _default_wimans_data_root():
    return os.environ.get(
        "WIMANS_DATA_ROOT",
        os.path.join(os.path.dirname(_sfuda_paper_root()), "wimans_dataset"),
    )


_WIMANS_DATA = _default_wimans_data_root()

preset = {
    # Define task
    "source_task": "activity",  # "identity", "activity", "location"
    "target_task": "activity",
    "repeat":3,
    # Paths of data
    "path": {
        "data_x": os.path.join(_WIMANS_DATA, "wifi_csi", "amp"),
        "data_y": os.path.join(_WIMANS_DATA, "annotation.csv"),
        "save_dir": "SHOT_WiMANS_results"  # Path to save results
    },

    # Data selection for experiments
    "source_data": {
        "num_users": ["0","1"],  # Selected number(s) of users
        "wifi_band": ["2.4"],  # Selected WiFi band(s)
        "environment": ["classroom"],  # Selected environment(s)
        "length": 3000,  # Default length of CSI
    },
    "target_data": {
        "num_users": ["4","5"],  # Selected number(s) of users
        "wifi_band": ["2.4"],  # Selected WiFi band(s)
        "environment": ["classroom"],  # Selected environment(s)
        "length": 3000,  # Default length of CSI
    },

    # Training configuration
    "training": {
        "gpu_id": "0,1,2",              # GPU device ID
        "max_epoch": 200,            # Maximum training epochs
        "batch_size": 128,           # Batch size for training
        "lr": 1e-3,                 # Learning rate
        "seed": 42,               # Random seed for reproducibility
        "bottleneck": 128,          # Bottleneck layer dimension
        "ent": True,                # Enable entropy minimization
        "ent_par": 1.0,             # Entropy loss weight
        "cls_par": 0.3,             # Classification loss weight
        "save_models": True,        # Save adapted models

    },
    
    # Encoding of activities and locations
    "encoding": {
        "activity": {  # Encoding of different activities
            "nan":      [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "nothing":  [1, 0, 0, 0, 0, 0, 0, 0, 0],
            "walk":     [0, 1, 0, 0, 0, 0, 0, 0, 0],
            "rotation": [0, 0, 1, 0, 0, 0, 0, 0, 0],
            "jump":     [0, 0, 0, 1, 0, 0, 0, 0, 0],
            "wave":     [0, 0, 0, 0, 1, 0, 0, 0, 0],
            "lie_down": [0, 0, 0, 0, 0, 1, 0, 0, 0],
            "pick_up":  [0, 0, 0, 0, 0, 0, 1, 0, 0],
            "sit_down": [0, 0, 0, 0, 0, 0, 0, 1, 0],
            "stand_up": [0, 0, 0, 0, 0, 0, 0, 0, 1],
        },
        "location": {  # Encoding of different locations
            "nan":  [0, 0, 0, 0, 0],
            "a":    [1, 0, 0, 0, 0],
            "b":    [0, 1, 0, 0, 0],
            "c":    [0, 0, 1, 0, 0],
            "d":    [0, 0, 0, 1, 0],
            "e":    [0, 0, 0, 0, 1],
        },
    },
}