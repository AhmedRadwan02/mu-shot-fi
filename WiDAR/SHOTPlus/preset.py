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
    "scenario_name": "crossRoom_Phase",

    # Source domain data configuration
    "source_data": {
       "rooms": ["Room1_Classroom"],
       "users": ["user1", "user2", "user3","user4", "user5"],
       "receivers": [1, 2, 3, 4, 5, 6],
       "gestures": [
           "Push&Pull", "Sweep", "Clap", "Slide",
           "Draw-O(Horizontal)", "Draw-Zigzag(Horizontal)"
       ],
       "torso_locations": [1,2,3,4,5],
       "face_orientations":  [1,2,3,4,5],
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
       "rooms": ["Room2_Hall"],
       "users": ["user1", "user2", "user3","user4", "user5"],
       "receivers": [1, 2, 3, 4, 5, 6],
       "gestures": [
           "Push&Pull", "Sweep", "Clap", "Slide",
           "Draw-O(Horizontal)", "Draw-Zigzag(Horizontal)"
       ],
       "torso_locations": [1,2,3,4,5],
       "face_orientations":  [1,2,3,4,5],
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
        
        # Target adaptation (SHOT parameters)
        "adaptation_lr": 1e-4,
        "adaptation_max_epoch": 70,  # Epochs for target adaptation
        
        # Entropy minimization
        "ent": True,              # Enable entropy minimization
        "ent_par": 1.0,           # Weight for entropy loss
        "gent": True,             # Enable gentleness (diversity) term
        
        # Pseudo-labeling
        "cls_par": 0.1,           # Weight for pseudo-labeling loss (0 to disable)
        
        # Self-supervised learning (rotation)
        "ssl": 0.0,               # Weight for SSL rotation loss (0 to disable, e.g., 0.6 to enable)
        "ssl_max_epoch": 70,      # Epochs for SSL rotation pre-training

        # Temporal-Shift Discrimination (TSD)
        "tsd": 0.0,               # Weight for TSD loss (0 to disable, e.g., 0.2 to enable)
        "tsd_max_epoch": 50,      # Epochs for TSD pre-training
        "tsd_shifts": [-200, -100, 0, 100, 200],  # Temporal shift values (5 classes)
    },
}