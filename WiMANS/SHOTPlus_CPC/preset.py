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
    "repeat": 3,

    # Paths of data
    "path": {
        "data_x": os.path.join(_WIMANS_DATA, "wifi_csi", "amp"),
        "data_y": os.path.join(_WIMANS_DATA, "annotation.csv"),
        "save_dir": "SHOTPlus_WiMANS_Results"  # Path to save results
    },

    # Data selection for experiments
    "source_data": {
        "num_users": ["0","1","2","3","4","5"],  # Selected number(s) of users
        "wifi_band": ["2.4"],  # Selected WiFi band(s)
        "environment": ["classroom"],  # Selected environment(s)
        "length": 3000,  # Default length of CSI
        "use_augmix": False,              # Enable/disable AugMix for source
        "augmix_severity": 5,            # Augmentation severity (1-10)
        "augmix_width": 3,               # Number of augmentation chains
        "augmix_depth": -1,              # Depth of chains (-1 for random 1-3)
        "augmix_alpha": 1.0,             # Beta distribution parameter for mixing
        "augmix_ops": [                  # CSI-specific augmentation operations
            "multipath_fading",
            "path_loss",
            "frequency_selective_fading",
            "environmental_interference",
        ],
    },
    "target_data": {
        "num_users": ["0","1","2","3","4","5"],  # Selected number(s) of users
        "wifi_band": ["5"],  # Selected WiFi band(s)
        "environment": ["meeting_room"],  # Selected environment(s)
        "length": 3000,  # Default length of CSI
        "use_augmix": False,              # Enable/disable AugMix for source
        "augmix_severity": 1,            # Augmentation severity (1-10)
        "augmix_width": 2,               # Number of augmentation chains
        "augmix_depth": -1,              # Depth of chains (-1 for random 1-3)
        "augmix_alpha": 0.3,             # Beta distribution parameter for mixing
        "augmix_ops": [                  # CSI-specific augmentation operations
            "multipath_fading",
            "environmental_interference",
        ],
    },
    
    # Training configuration
    "training": {
        "gpu_id": "0,1",              # GPU device ID
        "max_epoch": 50,               # Maximum training epochs
        "batch_size": 64,              # Batch size for training
        "lr": 1e-3,                     # Learning rate for source training
        "target_lr": 1e-4,              # Learning rate for target adaptation
        "seed": 42,                     # Random seed for reproducibility
        "bottleneck": 128,              # Bottleneck layer dimension

        # Label Smoothing
        "smooth": 0.2,                  # Label smoothing epsilon (0.2 = 20% smoothing)

        # Entropy Minimization & Diversity Regularization
        "ent": True,                    # Enable entropy minimization
        "gent": True,                   # Enable weighted diversity regularization (Gent)
        "use_occupancy_weighted_gent": True,
        "ent_par": 1.0,                 # Entropy loss weight
        "cls_par": 0.0,                 # Classification loss weight (for pseudo-labeling, if used)

        # Rotation Self-supervised learning (SSL)
        "ssl": 0.5,                     # SSL rotation loss weight (0 to disable, 0.6 recommended)
        "ssl_max_epoch": 70,            # Epochs for SSL rotation pre-training
        
        # ===== CPC Self-supervised learning (Per-Slot) =====
        "cpc_weight": 10.0,              # Weight for CPC loss (REDUCED - try 0.1, 0.05, 0.01)
        "old_cpc": False,
        "cpc_pretrain": True,           # Pre-train CPC on target domain
        "cpc_pretrain_epochs": 20,      # Epochs for CPC pre-training
        "cpc_pretrain_lr": 1e-3,        # Learning rate for CPC pre-training
        "cpc_temperature": 0.07,        # Temperature for InfoNCE
        


        # CPC Windowing parameters (WiMANS: 3000 timesteps / 30 = 100 windows)
        "cpc_window_size": 30,          # Window size in timesteps (3000/30 = 100 windows)
        "cpc_prediction_steps": 9,      # Number of future windows to predict (K)

        # CPC architecture parameters
        "cpc_embedding_dim": 256,       # Encoder output dimension
        "cpc_hidden_dim": 512,          # GRU hidden dimension
        "cpc_projection_dim": 256,      # Projection head output
        "cpc_num_gru_layers": 1,        # Number of GRU layers

        # CPC data augmentation (random masking)
        "cpc_use_masking": True,        # Enable random masking augmentation
        "cpc_mask_prob": 0.5,           # Probability of applying masking
        "cpc_mask_ratio": 0.15,         # Ratio of data to mask (15%)
        # ===== NEW: Per-Slot CPC Parameters =====
        "cpc_num_slots": 6,             # Number of slots (3 antennas)
                                        # input_freq auto-calculated: 270/3 
        "cpc_negative_mode": "both",  # Options: "cross_batch", "cross_slot", "both"
        "cpc_return_per_slot_loss": False,   # True for per-slot losses (advanced)
        

        "save_models": True             # Save adapted models
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
