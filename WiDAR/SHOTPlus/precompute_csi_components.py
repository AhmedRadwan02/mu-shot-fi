"""
[file]          precompute_csi_components.py
[description]   Pre-compute and save amplitude/phase .npy files for WiDAR dataset
                Applies CSI ratio and phase unwrapping before saving
"""
import os
import re
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List
import csiread
from tqdm import tqdm

# Pre-compiled regex for filename parsing
STEM_RE = re.compile(
    r'^(?P<user>user\d+)-(?P<gesture>\d+)-(?P<torso>\d+)-(?P<face>\d+)-(?P<rep>\d+)-r(?P<rx>\d+)\.dat$'
)

class CSIPreprocessor:
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
    
    def read_widar_csi(self, filename: str) -> np.ndarray:
        """Read Widar CSI with csiread, return shape (n_packets, 90) complex64."""
        try:
            file_path = Path(filename)
            if not file_path.exists() or file_path.stat().st_size == 0:
                return np.empty((0, 90), dtype=np.complex64)
    
            # Intel 5300 config: 3 Rx antennas × 1 Tx × 30 subcarriers
            csidata = csiread.Intel(filename, nrxnum=3, ntxnum=1, pl_size=0)
            csidata.read()
    
            # Shape: (packets, 30, nrx=3, ntx=1)
            csi = csidata.csi
    
            # Reshape to (packets, 90) to match your format
            n_packets = csi.shape[0]
            csi_flat = csi.reshape(n_packets, 90)
    
            return csi_flat.astype(np.complex64)
    
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return np.empty((0, 90), dtype=np.complex64)
    
    def compute_csi_ratio_global(self, csi_combined: np.ndarray, 
                                  reference_antenna: int = 1, 
                                  target_antenna: int = 2) -> np.ndarray:
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
    
    def find_file_patterns(self) -> Dict[str, Dict]:
        """Find all .dat file patterns grouped by base name (without receiver suffix)."""
        patterns: Dict[str, Dict] = {}
        
        print(f"Searching for .dat files in: {self.base_path}")
        dat_files = list(self.base_path.rglob("*.dat"))
        print(f"Found {len(dat_files)} total .dat files")
        
        for f in dat_files:
            if f.name in self.empty_files:
                continue
            
            match = STEM_RE.match(f.name)
            if not match:
                continue
            
            groups = match.groupdict()
            rx = int(groups['rx'])
            
            # Pattern key without receiver suffix
            stem_wo_rx = f.stem.rsplit('-', 1)[0]
            pattern = str(f.parent / stem_wo_rx)
            
            if pattern not in patterns:
                patterns[pattern] = {
                    'base_path': f.parent,
                    'base_name': stem_wo_rx,
                    'receivers': {}
                }
            patterns[pattern]['receivers'][rx] = f
        
        print(f"Found {len(patterns)} unique file patterns")
        return patterns
    
    def process_one_pattern(self, pattern: str, entry: Dict, 
                           use_csi_ratio: bool = True,
                           csi_ratio_antennas: tuple = (1, 2)) -> bool:
        """
        Process one file pattern: load CSI, compute amplitude/phase, save .npy files.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            available_receivers = entry['receivers']
            if not available_receivers:
                return False
            
            # Load all available receivers
            receivers = sorted(available_receivers.keys())
            max_rx = max(receivers)
            results = [None] * (max_rx + 1)
            
            # Parallel loading
            max_workers = min(self.io_workers, len(receivers))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_rx = {
                    executor.submit(self.read_widar_csi, str(path)): (rx, path) 
                    for rx, path in available_receivers.items()
                }
                
                for future in as_completed(future_to_rx):
                    rx, _ = future_to_rx[future]
                    try:
                        arr = future.result()
                        if arr.shape[0] > 0:
                            results[rx] = arr
                    except Exception as e:
                        print(f"Error loading receiver {rx}: {e}")
            
            # Filter non-empty results
            non_empty = [arr for arr in results if arr is not None]
            if not non_empty:
                return False
            
            # Align to minimum time length
            min_T = min(arr.shape[0] for arr in non_empty)
            if min_T == 0:
                return False
            
            # Combine into (time, 90, num_receivers)
            R = len(receivers)
            csi_combined = np.zeros((min_T, 90, R), dtype=np.complex64)
            
            rx_to_idx = {rx: idx for idx, rx in enumerate(receivers)}
            for rx in receivers:
                if results[rx] is not None and results[rx].shape[0] >= min_T:
                    out_idx = rx_to_idx[rx]
                    csi_combined[:, :, out_idx] = results[rx][:min_T]
            
            # Apply CSI ratio if requested
            if use_csi_ratio:
                csi_data = self.compute_csi_ratio_global(
                    csi_combined,
                    reference_antenna=csi_ratio_antennas[0],
                    target_antenna=csi_ratio_antennas[1]
                )  # Shape: (time, 30 * num_receivers)
            else:
                # Flatten to (time, 90 * num_receivers)
                csi_data = csi_combined.reshape(min_T, -1)
            
            # Compute amplitude
            amplitude = np.abs(csi_data).astype(np.float32)
            
            # Compute phase with unwrapping
            phase = np.angle(csi_data)
            phase = np.unwrap(phase, axis=0).astype(np.float32)  # Unwrap along time axis
            
            # Save .npy files in organized folders
            base_path = entry['base_path']
            base_name = entry['base_name']
            
            # Create amplitude and phase directories
            amp_dir = base_path / "amplitude"
            phase_dir = base_path / "phase"
            amp_dir.mkdir(exist_ok=True)
            phase_dir.mkdir(exist_ok=True)
            
            amp_file = amp_dir / f"{base_name}.npy"
            phase_file = phase_dir / f"{base_name}.npy"
            
            np.save(amp_file, amplitude)
            np.save(phase_file, phase)
            
            return True
            
        except Exception as e:
            print(f"Error processing pattern {pattern}: {e}")
            return False
    
    def precompute_all(self, use_csi_ratio: bool = True, 
                      csi_ratio_antennas: tuple = (1, 2)):
        """
        Precompute amplitude and phase for all file patterns.
        
        Args:
            use_csi_ratio: Whether to apply CSI ratio before computing amplitude/phase
            csi_ratio_antennas: Tuple of (reference_antenna, target_antenna) for CSI ratio
        """
        patterns = self.find_file_patterns()
        
        if not patterns:
            print("No file patterns found!")
            return
        
        print(f"\nProcessing {len(patterns)} file patterns...")
        print(f"CSI ratio: {'enabled' if use_csi_ratio else 'disabled'}")
        if use_csi_ratio:
            print(f"  Reference antenna: {csi_ratio_antennas[0]}")
            print(f"  Target antenna: {csi_ratio_antennas[1]}")
        print(f"Phase unwrapping: enabled")
        print()
        
        successful = 0
        failed = 0
        
        # Process patterns sequentially with progress bar
        for pattern, entry in tqdm(patterns.items(), desc="Processing patterns"):
            try:
                success = self.process_one_pattern(
                    pattern, 
                    entry,
                    use_csi_ratio,
                    csi_ratio_antennas
                )
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"\nError processing {pattern}: {e}")
                failed += 1
        
        print(f"\nProcessing complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total: {len(patterns)}")


def main():
    """Main function to precompute CSI components."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Pre-compute amplitude and phase .npy files for WiDAR dataset'
    )
    parser.add_argument(
        '--base_path',
        type=str,
        required=True,
        help='Base path to WiDAR dataset'
    )
    
    args = parser.parse_args()
    
    # Create preprocessor with default settings
    preprocessor = CSIPreprocessor(base_path=args.base_path)
    
    # Run preprocessing with default settings (CSI ratio enabled, antennas 1,2)
    preprocessor.precompute_all(
        use_csi_ratio=True,
        csi_ratio_antennas=(1, 2)
    )


if __name__ == "__main__":
    main()