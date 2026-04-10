#!/usr/bin/env python
"""
Test script to verify path resolution for SHOTPlus_CPC
"""
import os.path as osp
from preset import preset

def get_domain_name(config_key):
    """Generate domain name for WiDAR configuration"""
    config = preset[config_key]
    rooms = "_".join(config["rooms"])
    users = f"users_{len(config['users'])}"

    if "csi_components" in config:
        components = config["csi_components"]
        if len(components) == 1 and components[0] == "amplitude":
            phase_info = "amp_only"
        elif len(components) == 1 and components[0] == "phase":
            phase_info = "phase_only"
        elif len(components) == 2:
            phase_info = "amp_phase"
        else:
            phase_info = f"{'_'.join(components)}"
    else:
        phase_info = "default"

    return f"{rooms}_{users}_{phase_info}"

# Get configuration
initial_seed = preset["training"]["seed"]
scenario_name = preset.get("scenario_name", "shot_experiment")
experiment_name = f"{preset['source_task']}_to_{preset['target_task']}"
source_domain = get_domain_name("source_data")
target_domain = get_domain_name("target_data")

# Get the base save directory with path resolution
save_dir_relative = preset["path"]["save_dir"]
print(f"Original save_dir from preset: {save_dir_relative}")
print(f"Is absolute? {osp.isabs(save_dir_relative)}")

# Get script directory
script_dir = osp.dirname(osp.abspath(__file__))
print(f"Script directory: {script_dir}")

# CPC output directory: WiDAR/SHOTPlus_CPC/WiDAR_Domain_Adaptation_results
if not osp.isabs(save_dir_relative):
    cpc_save_dir = osp.join(script_dir, save_dir_relative)
else:
    cpc_save_dir = save_dir_relative

print(f"CPC save directory: {cpc_save_dir}")

# Source model directory: WiDAR/SHOTPlus/WiDAR_Domain_Adaptation_results
widar_dir = osp.dirname(script_dir)  # Go up from SHOTPlus_CPC to WiDAR
shotplus_dir = osp.join(widar_dir, "SHOTPlus")

if not osp.isabs(save_dir_relative):
    shotplus_save_dir = osp.join(shotplus_dir, save_dir_relative)
else:
    shotplus_save_dir = save_dir_relative

print(f"SHOTPlus directory: {shotplus_dir}")
print(f"SHOTPlus save directory: {shotplus_save_dir}")

# Construct CPC output directory
base_output_dir = osp.join(
    cpc_save_dir,
    f"{scenario_name}_{preset['dataset_type']}",
    experiment_name,
    f"{source_domain}_to_{target_domain}",
    f"seed_{initial_seed}_epochs_{preset.get('training', {}).get('max_epoch', 'default')}"
)

# Construct SHOTPlus source directory
shotplus_scenario_name = scenario_name.replace("_CPC", "")
source_model_base_dir = osp.join(
    shotplus_save_dir,
    f"{shotplus_scenario_name}_{preset['dataset_type']}",
    experiment_name,
    f"{source_domain}_to_{target_domain}",
    f"seed_{initial_seed}_epochs_{preset.get('training', {}).get('max_epoch', 'default')}"
)

# Print results
print("\n" + "="*80)
print("PATH RESOLUTION TEST")
print("="*80)
print(f"\nDirectory Structure:")
print(f"  WiDAR directory: {widar_dir}")
print(f"  SHOTPlus directory: {shotplus_dir}")
print(f"  SHOTPlus_CPC directory: {script_dir}")
print(f"\nSave Directories:")
print(f"  CPC save directory: {cpc_save_dir}")
print(f"  SHOTPlus save directory: {shotplus_save_dir}")
print(f"\nCPC Output Directory (run_0):")
print(f"  {osp.join(base_output_dir, 'run_0')}")
print(f"  Exists: {osp.exists(osp.join(base_output_dir, 'run_0'))}")
print(f"\nSHOTPlus Source Directory (run_0):")
print(f"  {osp.join(source_model_base_dir, 'run_0')}")
print(f"  Exists: {osp.exists(osp.join(source_model_base_dir, 'run_0'))}")

# Check for source model files
source_run0 = osp.join(source_model_base_dir, 'run_0')
if osp.exists(source_run0):
    print(f"\nChecking source model files in {source_run0}:")
    for model_file in ['source_F.pt', 'source_B.pt', 'source_C.pt']:
        file_path = osp.join(source_run0, model_file)
        exists = osp.exists(file_path)
        print(f"  {model_file}: {'✓ Found' if exists else '✗ Missing'}")
else:
    print(f"\n✗ Source model directory does not exist: {source_run0}")
    print("  Please train SHOTPlus first!")

print("\n" + "="*80)
