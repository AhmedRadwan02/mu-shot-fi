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
    "dataset_type": "widar",  # "widar" for WiDAR dataset
    "source_task": "gesture",  # gesture recognition task
    "target_task": "gesture",
    "repeat": 3,
    
    # Paths of data
    "path": {
        "base_path": os.path.join(_WIDAR_BASE, ""),
        "save_dir": "WiDAR_Domain_Adaptation_results"
    },
    "scenario_name": "crossRoom_Room12_to_Room3_AllUsers_noAugs_PerSampleNorm_onlyAmp_CSIRatio",
    "model": {
        "name": "resnet18"
    },
    
    "adaptation": {
        "method": "SHOT" 
    },
        
    "source_data": {
       "rooms": ["Room1_Classroom","Room2_Hall"],
       "users": ["user1", "user2", "user3", "user4", "user5", "user6", "user10", "user11", "user12", "user13", "user14", "user15", "user16", "user17"],
       "receivers": [1, 2, 3, 4, 5, 6],
       "gestures": [
           "Push&Pull", "Sweep", "Clap", "Slide",
           "Draw-O(Horizontal)", "Draw-Zigzag(Horizontal)"
       ],
       "torso_locations": [1, 2, 3, 4, 5],
       "face_orientations": [1, 2, 3, 4, 5],
       "target_len": 1200,
       "include_phase": False,
       "exclude_digits": True,
       "min_time_steps": 50,
       "max_samples_per_class": None,
       "use_csi_ratio": True,
       "csi_ratio_antennas": (1, 2),
    },
    
    "target_data": {
       "rooms": ["Room3_Office"],
       "users": ["user3", "user7", "user8", "user9"],
       "receivers": [1, 2, 3, 4, 5, 6],
       "gestures": [
           "Push&Pull", "Sweep", "Clap", "Slide",
           "Draw-O(Horizontal)", "Draw-Zigzag(Horizontal)"
       ],
       "torso_locations": [1, 2, 3, 4, 5],
       "face_orientations": [1, 2, 3, 4, 5],
       "target_len": 1200,
       "include_phase": False,
       "exclude_digits": True,
       "min_time_steps": 50,
       "max_samples_per_class": None,
       "use_csi_ratio": True,
       "csi_ratio_antennas": (1, 2),
    },
    # Data processing configuration
    "data_processing": {
        # Normalization options: "dataset", "per_sample", "none"
        "normalization": "per_sample",
        "norm_eps": 1e-5,
        
        # Data augmentation toggle and parameters
        "use_augmentation": False,
        
        # Time shift augmentation
        "p_timeshift": 0.6,
        "max_shift": 120,
        
        # Rectangular masking augmentation  
        "p_mask": 0.5,
        "mask_ratio": 0.10,
        "n_blocks": 1,
        "min_ar": 0.33,
        "max_ar": 3.0,
        "mask_value": 0.0,
        
        # Train/test split
        "test_size": 0.2,
        "pad_value": 0.0,
        "output_dtype": "float32",
    },


    # Training configuration
    "training": {
        "gpu_id": "all",
        "num_workers": 3,
        "max_epoch": 50,            # Maximum training epochs
        "batch_size": 96,           # Batch size for training
        "lr": 1e-3,                 # Learning rate
        "seed": 42,               # Random seed for reproducibility
        "bottleneck": 256,          # Bottleneck layer dimension
        "ent": True,                # Enable entropy minimization
        "ent_par": 1.0,             # Entropy loss weight
        "cls_par": 0.3,             # Classification loss weight
        "save_models": True,        # Save adapted models

    },
}