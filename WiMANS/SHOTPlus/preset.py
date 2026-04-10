# preset.py
import os

def _sfuda_paper_root():
    """Project root: parent of the ``WiMANS`` package (three levels up from this file)."""
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
        "save_dir": "SHOTPlus_Sensitivity_WiMANS_Results"  # Path to save results
    },
    
    # Data selection for experiments
    "source_data": {
        "num_users": ["0","1","2","3","4","5"],  # Selected number(s) of users
        "wifi_band": ["2.4"],  # Selected WiFi band(s)
        "environment": ["classroom"],  # Selected environment(s)
        "length": 3000,  # Default length of CSI
        "data_type": "amp",
        "normalize": None,
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
        "wifi_band": ["2.4"],  # Selected WiFi band(s)
        "environment": ["meeting_room"],  # Selected environment(s)
        "length": 3000,  # Default length of CSI
        "data_type": "amp",
        "normalize": None,
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
        "gpu_id": "0",              # GPU device ID
        "max_epoch": 50,               # Maximum training epochs
        "batch_size": 32,              # Batch size for training
        "lr": 1e-3,                     # Learning rate for source training
        "target_lr": 1e-4,              # Learning rate for target adaptation
        "seed": 42,                     # Random seed for reproducibility
        "bottleneck": 128,              # Bottleneck layer dimension
        "distill_weight": 1.0,
        # ============================================================
        # TRANSFORMER ARCHITECTURE (NEW)
        # ============================================================
        "use_transformer": False,         # Enable transformer decoder with learnable queries
        "num_decoder_layers": 4,         # Number of transformer decoder layers (try 2-6)
        "nhead": 2,                      # Number of attention heads (must divide bottleneck evenly)
        "dim_feedforward": 256,          # FFN hidden dimension (typically 2x bottleneck)
        "transformer_dropout": 0.1,      # Dropout in transformer layers
        # Note: If bottleneck=128 and nhead=4, each head has dimension 128/4=32
        # If bottleneck=256 and nhead=8, each head has dimension 256/8=32
        # ============================================================     
        # Label Smoothing
        "smooth": 0.2,                  # Label smoothing epsilon (0.1 = 10% smoothing)
        # Entropy Minimization & Diversity Regularization
        "ent": True,                    # Enable entropy minimization
        
        #----------Tests-----------------------
        "unfreeze_classifier":False,   # Keep classifier frozen during adaptation (recommended for transformer)
        "feature_coverage": False,      # Enable feature space coverage
        "coverage_weight": 0.3,         # Weight for coverage loss
        
        # New: Slot-level regularization (multi-user specific)
        "slot_diversity": False,        # Enable slot diversity
        "slot_div_weight": 0.3,         # Weight for slot diversity
        
        "use_occupancy_conditioned": False,
        "occ_count_weight": 0.3,
        #---------------------------------------
        
        # ============================================================
        # GENT SETTINGS (DISABLED - replaced by decoupled IM)
        # ============================================================
        "gent": True,                              # Disabled (using decoupled IM instead)
        "use_hierarchical_gent": False,             # Disabled
        "use_class_balanced_gent": False,           # Disabled
        "use_occupancy_weighted_gent": True,      # True = occ-weighted GENT; False = original GENT
        "use_slot_diversity_gent": False,
        
        "ent_par": 1.0,                 # Entropy loss weight (multiplies decoupled IM)
        "cls_par": 0.0,                 # Classification loss weight (for pseudo-labeling, if used)
        
        # Self-supervised learning (SSL) rotation
        "ssl": 0.005,                     # SSL rotation loss weight (0 to disable, 0.6 recommended)
        "ssl_max_epoch": 70,            # Epochs for SSL rotation pre-training
        
        #----------------UDA--------------------
        "use_uda": False,
        "use_source_supervision": False,
        "source_weight": 1.0,
        "use_mmd": False,
        "mmd_weight": 0.3,
        "use_adversarial": False,
        "adv_weight": 0.3,
        "use_coral": False,
        "coral_weight": 0.3,
        
        #----------Temporal Consistency----------------
        "temporal_consistency": False,
        "temporal_weight": 0.3,         # Start here, try 0.1-0.5
        "save_models": True,             # Save adapted models
        #---------- Dual---------------------
        "dual_branch": False,
        "branch1_epochs": 25,
        "branch1_ent_par": 1.0,
        "branch1_gent": False,
        "feature_distill_weight":0.5,
        "logit_distill_weight": 0.5,
        "distill_temperature": 4.0
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