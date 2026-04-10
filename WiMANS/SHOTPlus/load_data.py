"""
[file]          load_data.py
[description]   load annotation file and CSI amplitude/phase, encode labels, with normalization
"""
#
##
import os
import numpy as np
import pandas as pd
#
from preset import preset

#
##
def load_data_y(var_path_data_y,
                var_environment = None, 
                var_wifi_band = None, 
                var_num_users = None):
    """
    [description]
    : load annotation file (*.csv) as a pandas dataframe
    : according to selected environment(s), WiFi band(s), and number(s) of users
    [parameter]
    : var_path_data_y: string, path of annotation file
    : var_environment: list, selected environment(s), e.g., ["classroom"]
    : var_wifi_band: list, selected WiFi band(s), e.g., ["2.4"]
    : var_num_users: list, selected number(s) of users, e.g., ["0", "1", "2"]
    [return]
    : data_pd_y: pandas dataframe, labels of selected data
    """
    #
    ##
    data_pd_y = pd.read_csv(var_path_data_y, dtype = str)
    #
    if var_environment is not None:
        data_pd_y = data_pd_y[data_pd_y["environment"].isin(var_environment)]
    #
    if var_wifi_band is not None:
        data_pd_y = data_pd_y[data_pd_y["wifi_band"].isin(var_wifi_band)]
    #
    if var_num_users is not None:
        data_pd_y = data_pd_y[data_pd_y["number_of_users"].isin(var_num_users)]
    #
    return data_pd_y

#
##
def normalize_amplitude(data, method='minmax'):
    """
    [description]
    : normalize amplitude data
    [parameter]
    : data: numpy array, amplitude data
    : method: string, normalization method ('minmax', 'zscore', 'per_sample')
    [return]
    : normalized data
    """
    #
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min > 1e-10:
            return (data - data_min) / (data_max - data_min)
        else:
            return data
    #
    elif method == 'zscore':
        # Z-score normalization (zero mean, unit variance)
        data_mean = np.mean(data)
        data_std = np.std(data)
        if data_std > 1e-10:
            return (data - data_mean) / data_std
        else:
            return data - data_mean
    #
    elif method == 'per_sample':
        # Normalize each sample independently
        # Useful when samples have different scales
        normalized = np.zeros_like(data)
        for i in range(data.shape[0]):
            sample = data[i]
            sample_min = np.min(sample)
            sample_max = np.max(sample)
            if sample_max - sample_min > 1e-10:
                normalized[i] = (sample - sample_min) / (sample_max - sample_min)
            else:
                normalized[i] = sample
        return normalized
    #
    else:
        return data

#
##
def load_data_x(var_path_data_x, 
                var_label_list,
                normalize=None,
                data_type='amp'):
    """
    [description]
    : load CSI data (amplitude or phase) (*.npy)
    : filters out corrupted files and returns valid indices
    [parameter]
    : var_path_data_x: string, directory of CSI data files
    : var_label_list: list, selected labels
    : normalize: string or None, normalization method ('minmax', 'zscore', 'per_sample', None)
    : data_type: string, type of data ('amp', 'phase', 'ratio_amp', 'ratio_phase')
    [return]
    : data_x: numpy array, CSI data (only valid samples)
    : valid_indices: list, indices of valid samples (for filtering labels)
    """
    #
    ##
    var_path_list = [os.path.join(var_path_data_x, var_label + ".npy") for var_label in var_label_list]
    
    data_x = []
    valid_indices = []
    skipped_count = 0
    
    for idx, var_path in enumerate(var_path_list):
        #
        try:
            data_csi = np.load(var_path)
        except Exception as e:
            print(f"Failed to load {var_path}: {e}")
            skipped_count += 1
            continue
        
        # Check for NaN/Inf and skip corrupted files
        if np.any(np.isnan(data_csi)) or np.any(np.isinf(data_csi)):
            skipped_count += 1
            continue
        
        # Clean Inf/NaN values for ratio_amp (defensive)
        if 'ratio_amp' in data_type:
            if np.any(np.isinf(data_csi)):
                max_finite = np.nanmax(data_csi[np.isfinite(data_csi)])
                data_csi = np.nan_to_num(data_csi, nan=0.0, posinf=max_finite*2, neginf=0.0)
        
        # Pad to standard length
        var_pad_length = preset["source_data"]["length"] - data_csi.shape[0]
        data_csi_pad = np.pad(data_csi, ((var_pad_length, 0), (0, 0), (0, 0), (0, 0)))
        
        data_x.append(data_csi_pad)
        valid_indices.append(idx)
    
    if skipped_count > 0:
        print(f"⚠️  Skipped {skipped_count} corrupted files (NaN/Inf values)")
    
    print(f"✓ Loaded {len(data_x)}/{len(var_label_list)} valid files")
    
    data_x = np.array(data_x)
    
    # Apply normalization if specified
    if normalize is not None:
        # Only normalize amplitude, not phase (phase is already sanitized)
        if 'amp' in data_type or 'ratio_amp' in data_type:
            data_x = normalize_amplitude(data_x, method=normalize)
            print(f"✓ Applied {normalize} normalization")
    
    return data_x, valid_indices

#
##
def encode_data_y(data_pd_y, 
                  var_task):
    """
    [description]
    : encode labels according to specific task
    [parameter]
    : data_pd_y: pandas dataframe, labels of different tasks
    : var_task: string, indicate task
    [return]
    : data_y: numpy array, label encoding of task
    """
    #
    ##
    if var_task == "identity":
        #
        data_y = encode_identity(data_pd_y)
    #
    elif var_task == "activity":
        #
        data_y = encode_activity(data_pd_y, preset["encoding"]["activity"])
    #
    elif var_task == "location":
        #
        data_y = encode_location(data_pd_y, preset["encoding"]["location"])
    #
    return data_y

#
##
def encode_identity(data_pd_y):
    """
    [description]
    : encode identity labels in a pandas dataframe
    [parameter]
    : data_pd_y: pandas dataframe, labels of different tasks
    [return]
    : data_identity_onehot_y: numpy array, onehot encoding for identity labels
    """
    #
    ##
    data_location_pd_y = data_pd_y[["user_1_location", "user_2_location", 
                                    "user_3_location", "user_4_location", 
                                    "user_5_location", "user_6_location"]]
    # 
    data_identity_y = data_location_pd_y.to_numpy(copy = True).astype(str)
    #
    data_identity_y[data_identity_y != "nan"] = 1
    data_identity_y[data_identity_y == "nan"] = 0
    #
    data_identity_onehot_y = data_identity_y.astype("int8")
    #
    return data_identity_onehot_y

#
##
def encode_activity(data_pd_y, 
                    var_encoding):
    """
    [description]
    : encode activity labels in a pandas dataframe
    [parameter]
    : data_pd_y: pandas dataframe, labels of different tasks
    : var_encoding: dict, encoding of different activities
    [return]
    : data_activity_onehot_y: numpy array, onehot encoding for activity labels
    """
    #
    ##
    data_activity_pd_y = data_pd_y[["user_1_activity", "user_2_activity", 
                                    "user_3_activity", "user_4_activity", 
                                    "user_5_activity", "user_6_activity"]]
    #
    data_activity_y = data_activity_pd_y.to_numpy(copy = True).astype(str)
    #
    data_activity_onehot_y = np.array([[var_encoding[var_y] for var_y in var_sample] for var_sample in data_activity_y])
    #
    return data_activity_onehot_y

#
##
def encode_location(data_pd_y, 
                    var_encoding):
    """
    [description]
    : encode location labels in a pandas dataframe
    [parameter]
    : data_pd_y: pandas dataframe, labels of different tasks
    : var_encoding: dict, encoding of different locations
    [return]
    : data_location_onehot_y: numpy array, onehot encoding for location labels
    """
    #
    ##
    data_location_pd_y = data_pd_y[["user_1_location", "user_2_location", 
                                    "user_3_location", "user_4_location", 
                                    "user_5_location", "user_6_location"]]
    #
    data_location_y = data_location_pd_y.to_numpy(copy = True).astype(str)
    #
    data_location_onehot_y = np.array([[var_encoding[var_y] for var_y in var_sample] for var_sample in data_location_y])
    #
    return data_location_onehot_y

#
##
def test_load_data_y():
    """
    [description]
    : test load_data_y() function
    """
    #
    ##
    print(load_data_y(preset["path"]["data_y"], 
                      var_environment = ["classroom"]).describe())
    #
    print(load_data_y(preset["path"]["data_y"], 
                      var_environment = ["meeting_room"], 
                      var_wifi_band = ["2.4"]).describe())
    #
    print(load_data_y(preset["path"]["data_y"], 
                      var_environment = ["meeting_room"], 
                      var_wifi_band = ["2.4"], 
                      var_num_users = ["1", "2", "3"]).describe())

#
##
def test_load_data_x():
    """
    [description]
    : test load_data_x() function with normalization
    """
    #
    ##
    data_pd_y = load_data_y(preset["path"]["data_y"],
                            var_environment = ["meeting_room"], 
                            var_wifi_band = ["2.4"], 
                            var_num_users = None)
    #
    var_label_list = data_pd_y["label"].to_list()
    #
    # Test without normalization
    data_x, valid_idx = load_data_x(preset["path"]["data_x"], var_label_list, normalize=None)
    print(f"Without normalization - Shape: {data_x.shape}, Range: [{np.min(data_x):.4f}, {np.max(data_x):.4f}]")
    #
    # Test with minmax normalization
    data_x_minmax, _ = load_data_x(preset["path"]["data_x"], var_label_list, normalize='minmax')
    print(f"With minmax - Shape: {data_x_minmax.shape}, Range: [{np.min(data_x_minmax):.4f}, {np.max(data_x_minmax):.4f}]")
    #
    # Test with zscore normalization
    data_x_zscore, _ = load_data_x(preset["path"]["data_x"], var_label_list, normalize='zscore')
    print(f"With zscore - Shape: {data_x_zscore.shape}, Mean: {np.mean(data_x_zscore):.4f}, Std: {np.std(data_x_zscore):.4f}]")

#
##
def test_encode_identity():
    """
    [description]
    : test encode_identity() function
    """
    #
    ##
    data_pd_y = pd.read_csv(preset["path"]["data_y"], dtype = str)
    #
    data_identity_onehot_y = encode_identity(data_pd_y)
    #
    print(data_identity_onehot_y.shape)
    #
    print(data_identity_onehot_y[2000])

#
##
def test_encode_activity():
    """
    [description]
    : test encode_activity() function
    """
    #
    ##
    data_pd_y = pd.read_csv(preset["path"]["data_y"], dtype = str)
    #
    data_activity_onehot_y = encode_activity(data_pd_y, preset["encoding"]["activity"])
    #
    print(data_activity_onehot_y.shape)
    #
    print(data_activity_onehot_y[1560])

#
##
def test_encode_location():
    """
    [description]
    : test encode_location() function
    """
    #
    ##
    data_pd_y = pd.read_csv(preset["path"]["data_y"], dtype = str)
    #
    data_location_onehot_y = encode_location(data_pd_y, preset["encoding"]["location"])
    #
    print(data_location_onehot_y.shape)
    #
    print(data_location_onehot_y[1560])

#
##
if __name__ == "__main__":
    #
    ##
    test_load_data_y()
    #
    test_load_data_x()
    #
    test_encode_identity()
    #
    test_encode_activity()
    #
    test_encode_location()