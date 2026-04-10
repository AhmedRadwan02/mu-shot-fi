import os

def _sfuda_paper_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _default_widar_data_root():
    return os.environ.get(
        "WIDAR_DATA_ROOT",
        os.path.join(os.path.dirname(_sfuda_paper_root()), "widar_dataset", "organized_dataset"),
    )


_WIDAR_BASE = _default_widar_data_root()

preset = {
    # Define task and dataset type
    "dataset_type": "widar",
    "source_task": "gesture",
    "target_task": "gesture",
    "repeat": 3,

    # Paths
    "path": {
        "base_path": os.path.join(_WIDAR_BASE, ""),
        "save_dir": "WiDAR_Domain_Adaptation_results"
    },
    "scenario_name": "crossFace_Phase_CPC_Rotation",

    # Source domain data configuration
    "source_data": {
       "rooms": ["Room1_Classroom"],
       "users": ["user1", "user2", "user3","user4", "user5","user10","user11","user12","user13"],
       "receivers": [1, 2, 3, 4, 5, 6],
       "gestures": [
           "Push&Pull", "Sweep", "Clap", "Slide",
           "Draw-O(Horizontal)", "Draw-Zigzag(Horizontal)"
       ],
       "torso_locations": [1,2,3,4,5],
       "face_orientations":  [2,3,4,5],
       "target_len": 1200,
       "csi_components": ["phase"],
       "exclude_digits": True,
       "min_time_steps": 50,
       "max_samples_per_class": None,
       "use_csi_ratio": True,
       "csi_ratio_antennas": (1, 2),
    },

    # Target domain data configuration
    "target_data": {
       "rooms": ["Room1_Classroom"],
       "users": ["user1", "user2", "user3","user4", "user5","user10","user11","user12","user13"],
       "receivers": [1, 2, 3, 4, 5, 6],
       "gestures": [
           "Push&Pull", "Sweep", "Clap", "Slide",
           "Draw-O(Horizontal)", "Draw-Zigzag(Horizontal)"
       ],
       "torso_locations": [1,2,3,4,5],
       "face_orientations":  [1],
       "target_len": 1200,
       "csi_components": ["phase"],
       "exclude_digits": True,
       "min_time_steps": 50,
       "max_samples_per_class": None,
       "use_csi_ratio": True,
       "csi_ratio_antennas": (1, 2),
    },

    "data_processing": {
        "normalization": "per_sample",
        "norm_eps": 1e-5,

        "use_augmentation": False,
        "p_timeshift": 0.6,
        "max_shift": 120,
        "p_mask": 0.5,
        "mask_ratio": 0.15,
        "n_blocks": 1,
        "min_ar": 0.33,
        "max_ar": 3.0,
        "mask_value": 0.0,

        # Train/test split
        "test_size": 0.3,
        "pad_value": 0.0,
        "output_dtype": "float32",
    },

    # Training configuration
    "training": {
        # Hardware
        "gpu_id": "all",
        "num_workers": 16,

        # Source training
        "max_epoch": 30,
        "batch_size": 32,
        "lr": 1e-1,
        "seed": 42,
        "bottleneck": 512,
        "save_models": True,
        "smooth": 0.1,  # Label smoothing for source training

        # Target adaptation (SHOT parameters)
        "adaptation_lr": 1e-4,
        "adaptation_max_epoch": 70,  # Epochs for target adaptation

        # Entropy minimization
        "ent": True,              # Enable entropy minimization
        "ent_par": 1.0,           # Weight for entropy loss
        "gent": True,             # Enable gentleness (diversity) term

        # Pseudo-labeling
        "cls_par": 0.1,           # Weight for pseudo-labeling loss (0 to disable)

        # CPC Self-supervised learning
        "cpc_weight": 0.3,              # Weight for CPC loss (0 to disable)
        "cpc_pretrain": True,           # Pre-train CPC on target domain
        "cpc_pretrain_epochs": 70,      # Epochs for CPC pre-training
        "cpc_pretrain_lr": 1e-3,        # Learning rate for CPC pre-training
        "cpc_temperature": 0.07,        # Temperature for InfoNCE

        # CPC Windowing parameters
        "cpc_window_size": 10,          # Window size in timesteps (1200/10 = 120 windows)
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

        # Rotation Self-supervised learning (SSL)
        "ssl": 0.3,                     # Weight for rotation SSL loss (0 to disable)
        "ssl_max_epoch": 70,            # Epochs for SSL rotation pre-training
    },
}
