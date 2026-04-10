import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import random
from typing import Tuple, Optional, List, Dict, Any, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import mmap
import csiread
from pathlib import Path
# Import preset configuration
from preset import preset
# Pre-compiled regex for filename parsing
STEM_RE = re.compile(
    r'^(?P<user>user\d+)-(?P<gesture>\d+)-(?P<torso>\d+)-(?P<face>\d+)-(?P<rep>\d+)-r(?P<rx>\d+)\.dat$'
)
class WidarDatasetReader:
    def __init__(self, base_path: str, max_receivers: int = 6, io_workers: Optional[int] = None):
        self.base_path = Path(base_path)
        self.max_receivers = max_receivers
        self.io_workers = io_workers or 4

        # Known empty files to skip
        self.empty_files = frozenset({
            "user2-6-4-4-2-r1.dat", "user3-1-3-1-8-r5.dat", "user2-3-5-3-4-r4.dat",
            "user6-3-1-1-5-r5.dat", "user8-1-1-1-1-r5.dat", "user8-3-3-3-5-r2.dat", 
            "user9-1-1-1-1-r1.dat",
        })

        # Pre-compute digits mapping
        digits_map = {}
        for i in range(1, 11):
            digits_map[i] = f'Draw-{i if i != 10 else 0}'
        digits_map[0] = 'Draw-0'

        # Complete gesture mappings
        self.per_csi_mapping: Dict[str, Dict[int, str]] = {
            # Room 1 files
            'CSI_20181109': {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Slide', 5: 'Draw-Zigzag(Vertical)', 6: 'Draw-N(Vertical)'},
            'CSI_20181112': digits_map,
            'CSI_20181115': {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Draw-O(Vertical)', 5: 'Draw-Zigzag(Vertical)', 6: 'Draw-N(Vertical)'},
            'CSI_20181116': digits_map,
            'CSI_20181121': {1: 'Slide', 2: 'Draw-O(Horizontal)', 3: 'Draw-Zigzag(Horizontal)', 4: 'Draw-N(Horizontal)', 5: 'Draw-Triangle(Horizontal)', 6: 'Draw-Rectangle(Horizontal)'},
            'CSI_20181130': {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Slide', 5: 'Draw-O(Horizontal)', 6: 'Draw-Zigzag(Horizontal)', 7: 'Draw-N(Horizontal)', 8: 'Draw-Triangle(Horizontal)', 9: 'Draw-Rectangle(Horizontal)'},

            # Room 2 files
            'CSI_20181117': {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Draw-O(Vertical)', 5: 'Draw-Zigzag(Vertical)', 6: 'Draw-N(Vertical)'},
            'CSI_20181118': {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Draw-O(Vertical)', 5: 'Draw-Zigzag(Vertical)', 6: 'Draw-N(Vertical)'},
            'CSI_20181127': {1: 'Slide', 2: 'Draw-O(Horizontal)', 3: 'Draw-Zigzag(Horizontal)', 4: 'Draw-N(Horizontal)', 5: 'Draw-Triangle(Horizontal)', 6: 'Draw-Rectangle(Horizontal)'},
            'CSI_20181128': {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Draw-O(Horizontal)', 5: 'Draw-Zigzag(Horizontal)', 6: 'Draw-N(Horizontal)'},
            'CSI_20181204': {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Slide', 5: 'Draw-O(Horizontal)', 6: 'Draw-Zigzag(Horizontal)', 7: 'Draw-N(Horizontal)', 8: 'Draw-Triangle(Horizontal)', 9: 'Draw-Rectangle(Horizontal)'},
            'CSI_20181205': {1: 'Draw-O(Horizontal)', 2: 'Draw-Zigzag(Horizontal)', 3: 'Draw-N(Horizontal)', 4: 'Draw-Triangle(Horizontal)', 5: 'Draw-Rectangle(Horizontal)', 6: 'Draw-Rectangle(Horizontal)'},
            'CSI_20181208': {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Slide'},
            'CSI_20181209': {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Slide', 5: 'Draw-O(Horizontal)', 6: 'Draw-Zigzag(Horizontal)'},

            # Room 3 files
            'CSI_20181211': {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Slide', 5: 'Draw-O(Horizontal)', 6: 'Draw-Zigzag(Horizontal)'},
        }

        # Room mapping
        self.date_to_room = {
            'CSI_20181109': 'Room1_Classroom', 'CSI_20181112': 'Room1_Classroom', 
            'CSI_20181115': 'Room1_Classroom', 'CSI_20181116': 'Room1_Classroom',
            'CSI_20181121': 'Room1_Classroom', 'CSI_20181130': 'Room1_Classroom',
            'CSI_20181117': 'Room2_Hall', 'CSI_20181118': 'Room2_Hall',
            'CSI_20181127': 'Room2_Hall', 'CSI_20181128': 'Room2_Hall',
            'CSI_20181204': 'Room2_Hall', 'CSI_20181205': 'Room2_Hall',
            'CSI_20181208': 'Room2_Hall', 'CSI_20181209': 'Room2_Hall',
            'CSI_20181211': 'Room3_Office',
        }
    def compute_csi_ratio_global(self, csi_combined: np.ndarray, reference_antenna: int = 1, target_antenna: int = 2) -> np.ndarray:
        """
        Compute CSI ratio globally across all receivers.

        Args:
            csi_combined: Shape (time, 90, num_receivers)
            reference_antenna: Reference antenna (1, 2, or 3)  
            target_antenna: Target antenna (1, 2, or 3)

        Returns:
            CSI ratio with shape (time, 30 * num_receivers)
        """
        T, _, R = csi_combined.shape

        # Reshape to (time, 30_subcarriers, 3_antennas, num_receivers)
        csi_reshaped = csi_combined.reshape(T, 30, 3, R)

        # Extract reference and target antennas across ALL receivers
        ref_data = csi_reshaped[:, :, reference_antenna - 1, :]  # (time, 30, num_receivers)
        target_data = csi_reshaped[:, :, target_antenna - 1, :]  # (time, 30, num_receivers)

        # Compute ratio
        eps = 1e-8
        csi_ratio = target_data / (ref_data + eps)  # (time, 30, num_receivers)

        # Flatten to (time, 30 * num_receivers)
        return csi_ratio.reshape(T, -1)
    def read_widar_csi(self,filename: str) -> np.ndarray:
        """Read Widar CSI with csiread, return shape (n_packets, 90) complex64."""
        try:
            file_path = Path(filename)
            if not file_path.exists() or file_path.stat().st_size == 0:
                return np.empty((0, 90), dtype=np.complex64)

            # Intel 5300 config: 3 Rx antennas × 1 Tx × 30 subcarriers
            csidata = csiread.Intel(filename, nrxnum=3, ntxnum=1, pl_size=0)
            csidata.read()

            # Shape: (packets, 30, nrx=3, ntx=1)
            csi = csidata.csi  # complex128 array

            # Reshape to (packets, 90) to match your old function
            n_packets = csi.shape[0]
            csi_flat = csi.reshape(n_packets, 90)

            return csi_flat

        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return np.empty((0, 90), dtype=np.complex64)
    @staticmethod
    @lru_cache(maxsize=512)
    def _csi_date_from_path(p: Path) -> Optional[str]:
        """Cached CSI date extraction from path."""
        for part in p.parts:
            if part.startswith('CSI_') and len(part) == 12:
                return part
        return None
    def parse_filename(self, file: Path, csi_date_hint: Optional[str] = None) -> Optional[Dict]:
        """Optimized filename parsing."""
        name = file.name

        if name in self.empty_files:
            return None

        match = STEM_RE.match(name)
        if not match:
            print(f"Warning: unexpected filename format: {name}")
            return None

        try:
            groups = match.groupdict()
            gesture_num = int(groups['gesture'])

            csi_date = csi_date_hint or self._csi_date_from_path(file) or "UNKNOWN_DATE"
            action_name = self.map_gesture_num_to_action(gesture_num, csi_date)
            room = self.date_to_room.get(csi_date, "UNKNOWN_ROOM")

            return {
                "user": groups['user'], 
                "gesture_num": gesture_num, 
                "action_name": action_name,
                "torso_location": int(groups['torso']), 
                "face_orientation": int(groups['face']), 
                "repetition": int(groups['rep']),
                "receiver": int(groups['rx']), 
                "csi_date": csi_date, 
                "room": room,
            }
        except (ValueError, KeyError) as e:
            print(f"Warning: cannot parse {name}: {e}")
            return None
    def map_gesture_num_to_action(self, gesture_num: int, csi_date: str) -> str:
        """Gesture number to action name mapping."""
        mapping = self.per_csi_mapping.get(csi_date)
        if mapping and gesture_num in mapping:
            return mapping[gesture_num]

        if gesture_num == 10 and csi_date in ('CSI_20181112', 'CSI_20181116'):
            return 'Draw-0'

        print(f"Warning: unknown gesture {gesture_num} for {csi_date}")
        return f"Unknown_{csi_date}_{gesture_num}"
    def _segment_mean_downsample(self, X: np.ndarray, target_len: int) -> np.ndarray:
        """
        Area-preserving mean downsampling along time.
        Given X with shape (T, D) and target_len <= T, returns (target_len, D)
        where each output row is the average over a fractional bin of the input.
        """
        T, D = X.shape
        if target_len <= 0:
            raise ValueError("target_len must be positive")
        if target_len == T:
            return X
        if target_len > T:
            raise ValueError("target_len > T: this function only downsamples")

        Xf = X.astype(np.float64, copy=False)

        # Cumulative sum with a leading zero for integral lookups: (T+1, D)
        cs = np.vstack([np.zeros((1, D), dtype=Xf.dtype), np.cumsum(Xf, axis=0)])

        # Fractional bin edges in the original index space [0, T]
        edges = np.linspace(0.0, T, num=target_len + 1)

        # Linear interpolation of cumulative sums at fractional edges
        idx0 = np.floor(edges).astype(int)               # left integer index
        idx1 = np.minimum(idx0 + 1, T)                   # right index (clipped)
        frac = (edges - idx0)[:, None]                   # (target_len+1, 1)

        cs_edges = (1.0 - frac) * cs[idx0] + frac * cs[idx1]  # (target_len+1, D)

        # Segment sums and exact widths → means
        seg_sums = cs_edges[1:] - cs_edges[:-1]          # (target_len, D)
        widths = (edges[1:] - edges[:-1])[:, None]       # (target_len, 1)
        out = seg_sums / widths

        return out.astype(X.dtype, copy=False)
    def _fit_length(self, X: np.ndarray, target_len: int, pad_value: float = 0.0) -> np.ndarray:
        """Fit array to target length via downsampling or padding."""
        T, D = X.shape
        if T == target_len:
            return X

        if T > target_len:
            return self._segment_mean_downsample(X, target_len)

        out = np.full((target_len, D), pad_value, dtype=X.dtype)
        out[:T] = X
        return out
    def _index_patterns(self, rooms: List[str], users: List[str], receivers: List[int]) -> Dict[str, Dict]:
        """Index file patterns for efficient loading."""
        users_set = frozenset(users)
        rooms_set = frozenset(rooms)
        receivers_set = frozenset(receivers)
        patterns: Dict[str, Dict] = {}

        print(f"Searching for files in: {self.base_path}")
        print(f"Looking for rooms: {rooms}")
        print(f"Looking for users: {users}")
        print(f"Looking for receivers: {receivers}")

        dat_files = list(self.base_path.rglob("*.dat"))
        print(f"Found {len(dat_files)} total .dat files")

        valid_files = []
        for f in dat_files:
            if f.name in self.empty_files:
                continue

            parts = f.parts
            room_found = any(part in rooms_set for part in parts)
            user_found = any(part in users_set for part in parts)

            if room_found and user_found:
                valid_files.append(f)

        print(f"After filtering: {len(valid_files)} valid files")

        for f in valid_files:
            match = STEM_RE.match(f.name)
            if not match:
                continue

            groups = match.groupdict()
            user = groups['user']
            rx = int(groups['rx'])

            if user not in users_set or rx not in receivers_set:
                continue

            csi_date = None
            room_name = None
            for part in f.parts:
                if part.startswith('CSI_'):
                    csi_date = part
                elif part in rooms_set:
                    room_name = part

            if not csi_date or not room_name:
                continue

            stem_wo_rx = f.stem.rsplit('-', 1)[0]
            pattern = str(f.parent / stem_wo_rx)

            if pattern not in patterns:
                patterns[pattern] = {
                    'csi_date': csi_date,
                    'room': room_name,
                    'user': user,
                    'receivers': {}
                }
            patterns[pattern]['receivers'][rx] = f

        print(f"Found {len(patterns)} unique patterns")
        return patterns
    def _load_one_pattern(self, pattern: str, entry: Dict, min_time_steps: int, receivers: List[int]) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Load single pattern with parallel I/O."""
        available_receivers: Dict[int, Path] = entry['receivers']
        selected_receivers = {rx: path for rx, path in available_receivers.items() if rx in receivers}

        if not selected_receivers:
            return None, None

        max_rx = max(receivers)
        results = [None] * (max_rx + 1)

        max_workers = min(self.io_workers, len(selected_receivers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_rx = {
                executor.submit(self.read_widar_csi, str(path)): (rx, path) 
                for rx, path in selected_receivers.items()
            }

            for future in as_completed(future_to_rx):
                rx, _ = future_to_rx[future]
                try:
                    arr = future.result()
                    if arr.shape[0] > 0:
                        results[rx] = arr
                except Exception as e:
                    print(f"Error loading receiver {rx}: {e}")

        non_empty = [arr for arr in results if arr is not None]
        if not non_empty:
            return None, None

        min_T = min(arr.shape[0] for arr in non_empty)
        if min_T < min_time_steps:
            return None, None

        R = len(receivers)
        out = np.zeros((min_T, 90, R), dtype=np.complex64)

        rx_to_idx = {rx: idx for idx, rx in enumerate(receivers)}
        for rx in selected_receivers.keys():
            if results[rx] is not None and results[rx].shape[0] >= min_T:
                out_idx = rx_to_idx[rx]
                out[:, :, out_idx] = results[rx][:min_T]

        any_rx = next(iter(selected_receivers.values()))
        meta = self.parse_filename(any_rx, csi_date_hint=entry['csi_date'])
        if meta is None:
            return None, None

        meta = dict(meta)
        meta["room"] = entry['room']
        meta["file_pattern"] = pattern
        meta["selected_receivers"] = receivers

        return out, meta
    def load_dataset(self, rooms: List[str], users: List[str],
                     receivers: List[int] = None,
                     torso_locations: Optional[List[int]] = None,
                     gestures: Optional[List[str]] = None,
                     face_orientations: Optional[List[int]] = None,
                     exclude_digits: bool = False,
                     min_time_steps: int = 50,
                     max_samples_per_class: Optional[int] = None,
                     target_len: int = 1200,
                     pad_value: float = 0.0,
                     output_dtype: str = "float32",
                     csi_components: List[str] = ["amplitude"],  # NEW: List of components to include
                     use_csi_ratio: bool = False,
                     csi_ratio_antennas: tuple = (1, 2)
                     ) -> Tuple[np.ndarray, np.ndarray, List[Dict], LabelEncoder]:
        """
        Load WiDAR dataset with flexible component selection.

        Args:
            csi_components: List of components to include. Options: ["amplitude"], ["phase"], 
                           ["amplitude", "phase"], or ["phase", "amplitude"]
            All other parameters remain the same...
        """
        if receivers is None:
            receivers = list(range(1, self.max_receivers + 1))

        # Validate components
        valid_components = {"amplitude", "phase"}
        if not csi_components:
            raise ValueError("csi_components cannot be empty")

        for comp in csi_components:
            if comp not in valid_components:
                raise ValueError(f"Invalid component '{comp}'. Must be one of {valid_components}")

        patterns = self._index_patterns(rooms, users, receivers)
        if not patterns:
            return np.array([]), np.array([]), [], LabelEncoder()

        torso_set = frozenset(torso_locations) if torso_locations else frozenset(range(1, 9))
        face_set = frozenset(face_orientations) if face_orientations else frozenset(range(1, 6)) 
        gesture_filter_set = frozenset(gestures) if gestures else None

        samples, metas, labels = [], [], []
        class_counts: Dict[str, int] = {}

        print(f"Processing CSI with components: {csi_components}")

        for pat, entry in patterns.items():
            csi, meta = self._load_one_pattern(pat, entry, min_time_steps, receivers)
            if csi is None or meta is None:
                continue

            # Apply filters
            if meta["torso_location"] not in torso_set:
                continue
            if meta["face_orientation"] not in face_set:  
                continue
            action = meta["action_name"]
            if action.startswith("Unknown_"):
                continue
            if exclude_digits and any(action.startswith(f"Draw-{i}") for i in range(10)):
                continue
            if gesture_filter_set and action not in gesture_filter_set:
                continue
            if max_samples_per_class and class_counts.get(action, 0) >= max_samples_per_class:
                continue
            class_counts[action] = class_counts.get(action, 0) + 1

            # NEW: Process CSI with flexible component selection
            if use_csi_ratio:
                # Apply CSI ratio GLOBALLY after combining all receivers
                csi_ratio = self.compute_csi_ratio_global(
                    csi,  # Shape: (time, 90, num_receivers)
                    reference_antenna=csi_ratio_antennas[0],
                    target_antenna=csi_ratio_antennas[1]
                )  # Shape: (time, 30 * num_receivers)

                # Extract requested components from the ratio
                component_arrays = []

                if "amplitude" in csi_components:
                    amp = np.abs(csi_ratio, dtype=np.float32)
                    amp = self._fit_length(amp, target_len, pad_value=pad_value)
                    component_arrays.append(amp)

                if "phase" in csi_components:
                    phase = np.angle(csi_ratio)
                    phase = np.unwrap(phase, axis=0)  # Unwrap along time axis (first dimension)
                    phase = self._fit_length(phase, target_len, pad_value=pad_value)
                    component_arrays.append(phase)

                # Stack components in the order they were requested
                if len(component_arrays) == 1:
                    sample = component_arrays[0][:, :, np.newaxis]  # (T,F,1)
                else:
                    sample = np.stack(component_arrays, axis=2)     # (T,F,num_components)

            else:
                # Original processing with flexible component selection
                component_arrays = []

                if "amplitude" in csi_components:
                    amp = np.abs(csi, dtype=np.float32).reshape(csi.shape[0], -1)
                    amp = self._fit_length(amp, target_len, pad_value=pad_value)
                    component_arrays.append(amp)

                if "phase" in csi_components:
                    phase = np.angle(csi).reshape(csi.shape[0], -1)
                    phase = np.unwrap(phase, axis=0)  # Unwrap along time axis (first dimension)
                    phase = self._fit_length(phase, target_len, pad_value=pad_value)
                    component_arrays.append(phase)

                # Stack components in the order they were requested
                if len(component_arrays) == 1:
                    sample = component_arrays[0][:, :, np.newaxis]  # (T,F,1)
                else:
                    sample = np.stack(component_arrays, axis=2)     # (T,F,num_components)

            samples.append(sample.astype(output_dtype))
            metas.append(meta)
            labels.append(action)

        if not samples:
            return np.array([]), np.array([]), [], LabelEncoder()

        X = np.stack(samples, axis=0).astype(output_dtype, copy=False)
        encoder = LabelEncoder()
        y = encoder.fit_transform(labels).astype(np.int32)

        print(f"Final dataset: X={X.shape}, y={y.shape}, classes={list(encoder.classes_)}")
        print(f"Components included: {csi_components} (channels={X.shape[-1]})")

        return X, y, metas, encoder

def load_widar_data(config_key: str = "source_data", 
                    random_state: int = 42) -> Dict[str, Any]:
    """
    Load WiDAR dataset with flexible component selection using preset configuration.

    Args:
        config_key: "source_data" or "target_data"
        random_state: Random seed for reproducible splits

    Returns:
        Dictionary containing properly normalized data and loaders
    """
    # Get configuration from preset
    config = preset[config_key]
    base_path = preset["path"]["base_path"]

    print(f"Loading {config_key} with flexible component selection using preset config...")

    # Initialize WiDAR dataset reader
    reader = WidarDatasetReader(base_path)

    # NEW: Get components from config, with backward compatibility
    if "csi_components" in config:
        csi_components = config["csi_components"]
    else:
        # Backward compatibility: convert include_phase to component list
        include_phase = config.get("include_phase", True)
        if include_phase:
            csi_components = ["amplitude", "phase"]
        else:
            csi_components = ["amplitude"]

    # Load raw dataset with flexible component selection
    X, y, metas, encoder = reader.load_dataset(
        rooms=config["rooms"],
        users=config["users"],
        receivers=config["receivers"],
        torso_locations=config["torso_locations"],
        face_orientations=config["face_orientations"],
        gestures=config["gestures"],
        exclude_digits=config["exclude_digits"],
        min_time_steps=config["min_time_steps"],
        max_samples_per_class=config["max_samples_per_class"],
        target_len=config["target_len"],
        csi_components=csi_components,  # NEW: Use component list
        use_csi_ratio=config.get("use_csi_ratio", False),
        csi_ratio_antennas=config.get("csi_ratio_antennas", (1, 2)),
        pad_value=preset["data_processing"]["pad_value"],
        output_dtype=preset["data_processing"]["output_dtype"]
    )

    if len(X) == 0:
        raise ValueError(f"No data loaded for {config_key}! Check your configuration.")

    print(f"Loaded {len(X)} samples with {len(encoder.classes_)} classes")
    print(f"Raw data shape: {X.shape}")
    print(f"Classes: {list(encoder.classes_)}")
    print(f"CSI components: {csi_components}")

    # Report CSI ratio usage
    if config.get("use_csi_ratio", False):
        ref_ant, target_ant = config.get("csi_ratio_antennas", (1, 2))
        print(f"GLOBAL CSI ratio applied: antenna {target_ant} / antenna {ref_ant}")
        print(f"Feature reduction: 90 -> 30 per receiver (noise reduction)")
    else:
        print("Using original CSI data (all antennas)")

    # Check data statistics before normalization
    print(f"Raw data statistics:")
    print(f"  Mean: {X.mean():.6f}, Std: {X.std():.6f}")
    print(f"  Min: {X.min():.6f}, Max: {X.max():.6f}")

    # Report component-specific statistics
    if X.ndim == 4:  # Has channel dimension
        for i, component in enumerate(csi_components):
            if i < X.shape[-1]:
                comp_data = X[..., i]
                print(f"  {component.capitalize()} channel - Mean: {comp_data.mean():.6f}, Std: {comp_data.std():.6f}")
    else:
        print(f"  Single component ({csi_components[0]})")

    # Rest of the function remains the same...
    # [Continue with existing normalization pipeline and data loader creation]

    # Train-test split
    test_size = preset["data_processing"]["test_size"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")

    # STANDARDIZED NORMALIZATION PIPELINE
    print("\n" + "="*50)
    print("APPLYING STANDARDIZED NORMALIZATION PIPELINE")
    print("="*50)

    normalization_type = preset["data_processing"]["normalization"]
    print(f"Normalization type: {normalization_type}")
    print(f"Normalization eps: {preset['data_processing']['norm_eps']}")

    X_train_norm = X_train
    X_test_norm = X_test
    dataset_normalizer = None

    if normalization_type == "dataset":
        # Dataset-level normalization: fit on training data, apply to all
        print("Applying dataset-level normalization...")
        dataset_normalizer = DatasetNormalizer()
        X_train_norm = dataset_normalizer.fit_transform(X_train)
        X_test_norm = dataset_normalizer.transform(X_test)

        print(f"After dataset normalization:")
        print(f"  Train - Mean: {X_train_norm.mean():.6f}, Std: {X_train_norm.std():.6f}")
        print(f"  Test - Mean: {X_test_norm.mean():.6f}, Std: {X_test_norm.std():.6f}")

    elif normalization_type == "per_sample":
        # Only per-sample normalization will be applied during data loading
        print("Dataset-level normalization skipped. Using only per-sample normalization.")

    elif normalization_type == "none":
        # No normalization at all
        print("All normalization disabled.")

    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}. Use 'dataset', 'per_sample', or 'none'")

    # STANDARDIZED TRANSFORM SETUP
    print("\nSetting up standardized transforms...")
    per_sample_norm = None
    if normalization_type in ["dataset", "per_sample"]:
        per_sample_norm = CSIPerSampleTimeNorm()
        print(f"Per-sample normalization enabled (eps={preset['data_processing']['norm_eps']})")

    augmenter = None
    if preset["data_processing"]["use_augmentation"]:
        augmenter = CSIAugment()
        print(f"Data augmentation enabled with preset parameters:")
        print(f"  Time shift: p={preset['data_processing']['p_timeshift']}, max_shift={preset['data_processing']['max_shift']}")
        print(f"  Masking: p={preset['data_processing']['p_mask']}, ratio={preset['data_processing']['mask_ratio']}")

    # Create combined transforms
    def create_transform(use_augment=False):
        transforms = []
        if use_augment and augmenter is not None:
            transforms.append(augmenter)
        if per_sample_norm is not None:
            transforms.append(per_sample_norm)

        def combined_transform(x):
            for transform in transforms:
                x = transform(x)
            return x
        return combined_transform if transforms else None

    # Training transform: augmentation + per-sample normalization (if enabled)
    train_transform = create_transform(use_augment=True)

    # Test transform: only per-sample normalization (if enabled)
    test_transform = create_transform(use_augment=False)

    # Create datasets with transforms
    train_dataset = CSIDataset(X_train_norm, y_train, transform=train_transform)
    test_dataset = CSIDataset(X_test_norm, y_test, transform=test_transform)

    # Split test set into validation and test sets (50/50 split)
    test_size = len(test_dataset)
    val_size = test_size // 2
    test_size = test_size - val_size

    # Create indices for splitting
    indices = list(range(len(test_dataset)))
    val_indices = indices[:val_size]
    test_indices = indices[val_size:]

    # Create validation and test subsets
    val_dataset = torch.utils.data.Subset(test_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    print(f"Created datasets with standardized transforms:")
    print(f"  Train dataset: {len(train_dataset)} samples")
    print(f"  Validation dataset: {len(val_dataset)} samples")
    print(f"  Test dataset: {len(test_dataset)} samples")

    # Create data loaders
    batch_size = preset["training"]["batch_size"]
    num_workers = preset["training"]["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    train_loader_no_shuffle = DataLoader(
        train_dataset,  # SAME dataset as train_loader
        batch_size=batch_size,
        shuffle=False,  # NOT shuffled - for consistent indexing during pseudo-labeling
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False  # Don't drop last to get all samples
    )
    

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    print(f"Created data loaders:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Determine input characteristics
    sample_shape = X_train_norm[0].shape  # Shape of single sample
    num_classes = len(encoder.classes_)
    input_channels = len(csi_components)

    feature_dim = sample_shape[-2] if len(sample_shape) > 2 else sample_shape[-1]

    # Calculate expected feature dimension based on CSI ratio usage
    num_receivers = len(config["receivers"])
    if config.get("use_csi_ratio", False):
        expected_features = 30 * num_receivers  # 30 subcarriers per receiver
    else:
        expected_features = 90 * num_receivers  # 90 features per receiver

    print(f"\nDataset characteristics:")
    print(f"  Single sample shape: {sample_shape}")
    print(f"  Input channels: {input_channels}")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Expected features: {expected_features}")
    print(f"  Number of classes: {num_classes}")
    print(f"  CSI components: {csi_components}")
    if config.get("use_csi_ratio", False):
        ref_ant, target_ant = config.get("csi_ratio_antennas", (1, 2))
        print(f"  CSI ratio: antenna {target_ant} / antenna {ref_ant}")
    print(f"  Normalization applied: {normalization_type}")

    return {
        "train_loader": train_loader,
        "train_loader_no_shuffle": train_loader_no_shuffle,
        "val_loader": val_loader,  # Added validation loader
        "test_loader": test_loader,
        "X_train": X_train_norm,
        "X_test": X_test_norm,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": encoder,
        "metadata": metas,
        "dataset_normalizer": dataset_normalizer,
        "num_classes": num_classes,
        "input_channels": input_channels,
        "feature_dim": feature_dim,
        "sample_shape": sample_shape,
        "csi_components": csi_components,
        "use_csi_ratio": config.get("use_csi_ratio", False),
        "csi_ratio_antennas": config.get("csi_ratio_antennas", (1, 2)) if config.get("use_csi_ratio", False) else None,
        "config_used": config,
        "normalization_type": normalization_type
    }
# ----------------- STANDARDIZED Data Normalization Classes (unchanged) -----------------
class DatasetNormalizer:
    """
    Dataset-level normalization across (N, T) dimensions.
    CRITICAL: This must be fitted ONLY on training data, then applied to all splits.
    """
    def __init__(self, eps: Optional[float] = None):
        self.eps = eps if eps is not None else preset["data_processing"]["norm_eps"]
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, X: np.ndarray):
        """Compute normalization statistics from training data ONLY."""
        print(f"Computing dataset normalization statistics for shape {X.shape}")

        # Normalize across (N, T) dimensions, keeping (F, C) separate
        self.mean = X.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self.std = X.std(axis=(0, 1), keepdims=True).astype(np.float32)

        # Prevent division by zero
        self.std[self.std < self.eps] = 1.0

        self.fitted = True
        print(f"Dataset normalization stats:")
        print(f"  Mean shape: {self.mean.shape}, range: [{self.mean.min():.6f}, {self.mean.max():.6f}]")
        print(f"  Std shape: {self.std.shape}, range: [{self.std.min():.6f}, {self.std.max():.6f}]")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization to data."""
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")

        X_norm = ((X - self.mean) / self.std).astype(np.float32)
        print(f"After dataset normalization: mean={X_norm.mean():.6f}, std={X_norm.std():.6f}")
        return X_norm

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
class CSIPerSampleTimeNorm:
    """
    Per-sample normalization across time dimension.
    CRITICAL: This is applied during training to each sample individually.
    """
    def __init__(self, eps: Optional[float] = None):
        self.eps = eps if eps is not None else preset["data_processing"]["norm_eps"]
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize each sample across time dimension.
        x: [T, F] or [T, F, C]
        """
        if x.ndim == 2:  # [T, F]
            mu = x.mean(axis=0, keepdims=True)  # [1, F]
            sd = x.std(axis=0, keepdims=True)   # [1, F]
        elif x.ndim == 3:  # [T, F, C]
            mu = x.mean(axis=0, keepdims=True)  # [1, F, C]
            sd = x.std(axis=0, keepdims=True)   # [1, F, C]
        else:
            raise ValueError(f"Expected 2D or 3D array per sample, got {x.shape}")

        # Prevent division by zero
        sd = np.where(sd < self.eps, 1.0, sd)

        return ((x - mu) / sd).astype(np.float32)
class CSIPerSampleMinMaxNorm:
    """
    Per-sample Min-Max normalization to [0, 1] range.
    Applied during training to each sample individually.
    """
    def __init__(self, eps: Optional[float] = None):
        self.eps = eps if eps is not None else preset["data_processing"]["norm_eps"]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize each sample to [0, 1] range.
        x: [T, F] or [T, F, C]
        """
        if x.ndim == 2:  # [T, F]
            x_min = x.min(axis=0, keepdims=True)  # [1, F]
            x_max = x.max(axis=0, keepdims=True)  # [1, F]
        elif x.ndim == 3:  # [T, F, C]
            x_min = x.min(axis=0, keepdims=True)  # [1, F, C]
            x_max = x.max(axis=0, keepdims=True)  # [1, F, C]
        else:
            raise ValueError(f"Expected 2D or 3D array per sample, got {x.shape}")

        # Compute range
        x_range = x_max - x_min

        # Prevent division by zero (if range is too small, keep original values)
        x_range = np.where(x_range < self.eps, 1.0, x_range)

        # Min-max scaling to [0, 1]
        return ((x - x_min) / x_range).astype(np.float32)
# ----------------- STANDARDIZED Data Augmentation (unchanged) -----------------
class CSIAugment:
    """
    CSI data augmentation with time shift and rectangular masking.
    All parameters are read from preset configuration.
    """
    def __init__(
        self,
        p_timeshift: Optional[float] = None,
        max_shift: Optional[int] = None,
        p_mask: Optional[float] = None,
        mask_ratio: Optional[float] = None,
        n_blocks: Optional[int] = None,
        min_ar: Optional[float] = None,
        max_ar: Optional[float] = None,
        mask_value: Optional[float] = None,
    ):
        # Use preset values if not provided
        self.p_timeshift = p_timeshift if p_timeshift is not None else preset["data_processing"]["p_timeshift"]
        self.max_shift = max_shift if max_shift is not None else preset["data_processing"]["max_shift"]
        self.p_mask = p_mask if p_mask is not None else preset["data_processing"]["p_mask"]
        self.mask_ratio = mask_ratio if mask_ratio is not None else preset["data_processing"]["mask_ratio"]
        self.n_blocks = n_blocks if n_blocks is not None else max(1, int(preset["data_processing"]["n_blocks"]))
        self.min_ar = min_ar if min_ar is not None else float(preset["data_processing"]["min_ar"])
        self.max_ar = max_ar if max_ar is not None else float(preset["data_processing"]["max_ar"])
        self.mask_value = mask_value if mask_value is not None else preset["data_processing"]["mask_value"]
    def _ensure_3d(self, x: np.ndarray):
        """Ensure input is 3D: [T,F,C]"""
        if x.ndim == 2:
            return x[..., None]
        return x
    def _time_shift_zero_pad(self, x: np.ndarray, max_shift: int):
        """Apply time shift with zero padding."""
        if max_shift <= 0:
            return x
        T, F, C = x.shape
        s = np.random.randint(-max_shift, max_shift + 1)
        if s == 0:
            return x
        out = np.empty_like(x)
        if s > 0:
            out[:s, :, :] = 0.0
            out[s:, :, :] = x[:T - s, :, :]
        else:
            s = -s
            out[T - s:, :, :] = 0.0
        return out
    def _mask_by_percentage(self, x: np.ndarray, ratio: float, n_blocks: int, min_ar: float, max_ar: float):
        """Apply rectangular masking by percentage."""
        T, F, C = x.shape
        total = T * F
        if total <= 0 or ratio <= 0:
            return x
        area_per = max(1, int(round((ratio * total) / n_blocks)))
        for _ in range(n_blocks):
            ar = float(np.random.uniform(min_ar, max_ar))
            h = int(max(1, round(np.sqrt(area_per / ar))))
            w = int(max(1, round(h * ar)))
            h = min(h, T)
            w = min(w, F)
            t0 = 0 if T == h else np.random.randint(0, T - h + 1)
            f0 = 0 if F == w else np.random.randint(0, F - w + 1)
            x[t0:t0+h, f0:f0+w, :] = self.mask_value
        return x
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply augmentations to input data."""
        x = x.astype(np.float32, copy=True)
        x3 = self._ensure_3d(x)

        # Time shift
        if np.random.rand() < self.p_timeshift:
            x3 = self._time_shift_zero_pad(x3, self.max_shift)

        # Percentage masking
        if np.random.rand() < self.p_mask:
            x3 = self._mask_by_percentage(
                x3, ratio=self.mask_ratio, n_blocks=self.n_blocks,
                min_ar=self.min_ar, max_ar=self.max_ar
            )

        # Return with original dimensions
        return x3 if x.ndim == 3 else x3[..., 0]
# ----------------- STANDARDIZED PyTorch Dataset (unchanged) -----------------
class CSIDataset(Dataset):
    """PyTorch Dataset for CSI data with optional transforms."""
    def __init__(self, X_np: np.ndarray, y_np: np.ndarray, transform=None):
        self.X = X_np
        self.y = y_np
        self.transform = transform
        assert self.X.ndim in (3, 4), f"Expected [N,T,F] or [N,T,F,C], got {self.X.shape}"
        assert self.X.shape[0] == self.y.shape[0]
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        x = self.X[idx]  # [T,F] or [T,F,C]
        if x.ndim == 2:   # (T,F)
            x = np.expand_dims(x, axis=2)  # (T,F,1)
        if self.transform is not None:
            x = self.transform(x)
        return x.astype(np.float32, copy=False), self.y[idx]
# ----------------- Testing Functions for Component Selection -----------------
def test_component_selection():
    """Test the new component selection functionality."""
    print("="*60)
    print("TESTING FLEXIBLE CSI COMPONENT SELECTION")
    print("="*60)

    # Test different component combinations
    test_configs = [
        ["amplitude"],
        ["phase"], 
        ["amplitude", "phase"],
        ["phase", "amplitude"]  # Different order
    ]

    try:
        for components in test_configs:
            print(f"\n{'='*40}")
            print(f"TESTING COMPONENTS: {components}")
            print(f"{'='*40}")

            # Temporarily modify preset for testing
            original_components = preset.get("source_data", {}).get("csi_components", ["amplitude", "phase"])
            if "source_data" not in preset:
                preset["source_data"] = {}
            preset["source_data"]["csi_components"] = components

            try:
                # Load data with specified components
                data_info = load_widar_data("source_data", random_state=42)

                print("✓ Data loading successful")
                print(f"  Input channels: {data_info['input_channels']}")
                print(f"  Expected channels: {len(components)}")
                print(f"  Components: {data_info['csi_components']}")
                print(f"  Sample shape: {data_info['sample_shape']}")

                # Test a batch
                train_batch = next(iter(data_info['train_loader']))
                x_batch, y_batch = train_batch

                print(f"✓ Batch loading successful")
                print(f"  Batch shape: {x_batch.shape}")
                print(f"  Expected final dimension: {len(components)}")

                # Verify channel dimension matches expected
                if x_batch.shape[-1] != len(components):
                    print(f"  ⚠️  Warning: Expected {len(components)} channels, got {x_batch.shape[-1]}")
                else:
                    print(f"  ✓ Channel dimension correct: {x_batch.shape[-1]}")

            except Exception as e:
                print(f"✗ Error with components {components}: {e}")
            finally:
                # Restore original setting
                preset["source_data"]["csi_components"] = original_components

        return True

    except Exception as e:
        print(f"✗ Error during component testing: {e}")
        import traceback
        traceback.print_exc()
        return False
# Example usage with different component configurations
def example_usage():
    """Example of how to use the new component selection."""
    print("="*60)
    print("EXAMPLE USAGE OF COMPONENT SELECTION")
    print("="*60)

    examples = [
        {
            "name": "Amplitude only",
            "components": ["amplitude"],
            "description": "Use only amplitude information (magnitude of CSI)"
        },
        {
            "name": "Phase only", 
            "components": ["phase"],
            "description": "Use only phase information (angle of CSI)"
        },
        {
            "name": "Both amplitude and phase",
            "components": ["amplitude", "phase"], 
            "description": "Use both amplitude and phase (full complex CSI)"
        },
        {
            "name": "Phase then amplitude (different order)",
            "components": ["phase", "amplitude"],
            "description": "Same as above but with phase as first channel"
        }
    ]

    for example in examples:
        print(f"\n{example['name']}:")
        print(f"  Components: {example['components']}")
        print(f"  Description: {example['description']}")
        print(f"  Expected channels: {len(example['components'])}")
        print(f"  Usage: preset['source_data']['csi_components'] = {example['components']}")
def apply_windowing(data_x, windowing=None):
    """
    Apply windowing transformation to split time dimension into windows

    Args:
        data_x: Input data with shape (batch, time, features) or (batch, time, features, channels)
        windowing: Number of windows to create (uses preset if None)

    Returns:
        Windowed data with shape (batch, window, time_per_window, features) or 
        (batch, window, time_per_window, features, channels)
    """
    if windowing is None:
        windowing = preset["target_data"].get("windowing", 1)

    if windowing <= 1:
        return data_x

    if data_x.ndim == 3:  # (batch, time, features)
        batch_size, time_steps, features = data_x.shape
        channels = None
    elif data_x.ndim == 4:  # (batch, time, features, channels)
        batch_size, time_steps, features, channels = data_x.shape
    else:
        raise ValueError(f"Expected 3D or 4D input, got shape {data_x.shape}")

    # Calculate time steps per window
    time_per_window = time_steps // windowing

    if time_per_window == 0:
        raise ValueError(f"Cannot create {windowing} windows from {time_steps} time steps. "
                        f"Reduce windowing parameter or increase time dimension.")

    # Trim data to fit evenly into windows
    trimmed_time = time_per_window * windowing
    if trimmed_time < time_steps:
        print(f"Warning: Trimming time dimension from {time_steps} to {trimmed_time} "
              f"to fit {windowing} windows evenly")
        data_x = data_x[:, :trimmed_time]

    # Reshape based on input dimensions
    if data_x.ndim == 3:
        # (batch, time, features) -> (batch, windowing, time_per_window, features)
        windowed_data = data_x.reshape(batch_size, windowing, time_per_window, features)
    else:
        # (batch, time, features, channels) -> (batch, windowing, time_per_window, features, channels)
        windowed_data = data_x.reshape(batch_size, windowing, time_per_window, features, channels)

    return windowed_data
def test_normalization_pipeline():
    """Test the standardized normalization pipeline with debug output."""
    print("="*60)
    print("TESTING STANDARDIZED NORMALIZATION PIPELINE")
    print("="*60)

    try:
        # Test different normalization types
        for norm_type in ["dataset", "per_sample", "none"]:
            print(f"\n{'='*40}")
            print(f"TESTING NORMALIZATION TYPE: {norm_type}")
            print(f"{'='*40}")

            # Temporarily modify preset for testing
            original_norm = preset["data_processing"]["normalization"]
            preset["data_processing"]["normalization"] = norm_type

            try:
                # Load source data with normalization
                data_info = load_widar_data("source_data", random_state=42)

                print("✓ Data loading successful")
                print(f"  Train loader: {len(data_info['train_loader'])} batches")
                print(f"  Test loader: {len(data_info['test_loader'])} batches")
                print(f"  Input channels: {data_info['input_channels']}")
                print(f"  Feature dim: {data_info['feature_dim']}")
                print(f"  Classes: {data_info['num_classes']}")
                print(f"  Normalization type: {data_info['normalization_type']}")

                # Test a batch
                train_batch = next(iter(data_info['train_loader']))
                x_batch, y_batch = train_batch

                print(f"✓ Batch loading successful")
                print(f"  Batch shape: {x_batch.shape}")
                print(f"  Batch dtype: {x_batch.dtype}")
                print(f"  Batch stats: mean={x_batch.mean():.6f}, std={x_batch.std():.6f}")
                print(f"  Label shape: {y_batch.shape}")

            except Exception as e:
                print(f"✗ Error with {norm_type} normalization: {e}")
            finally:
                # Restore original setting
                preset["data_processing"]["normalization"] = original_norm

        return True

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
def debug_preset_usage():
    """Debug which preset parameters are being used."""
    print("="*60)
    print("DEBUGGING PRESET PARAMETER USAGE")
    print("="*60)

    print("Data processing parameters from preset:")
    for key, value in preset["data_processing"].items():
        print(f"  {key}: {value}")

    print("\nTesting parameter usage...")

    # Test DatasetNormalizer
    normalizer = DatasetNormalizer()
    print(f"DatasetNormalizer eps: {normalizer.eps} (should be {preset['data_processing']['norm_eps']})")

    # Test CSIPerSampleTimeNorm
    per_sample_norm = CSIPerSampleTimeNorm()
    print(f"CSIPerSampleTimeNorm eps: {per_sample_norm.eps} (should be {preset['data_processing']['norm_eps']})")

    # Test CSIAugment
    augmenter = CSIAugment()
    print(f"CSIAugment parameters:")
    print(f"  p_timeshift: {augmenter.p_timeshift} (should be {preset['data_processing']['p_timeshift']})")
    print(f"  max_shift: {augmenter.max_shift} (should be {preset['data_processing']['max_shift']})")
    print(f"  p_mask: {augmenter.p_mask} (should be {preset['data_processing']['p_mask']})")
    print(f"  mask_ratio: {augmenter.mask_ratio} (should be {preset['data_processing']['mask_ratio']}")
    print(f"  mask_value: {augmenter.mask_value} (should be {preset['data_processing']['mask_value']})")