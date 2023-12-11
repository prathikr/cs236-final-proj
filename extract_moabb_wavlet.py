import os 
import time
import warnings
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import moabb
from moabb import datasets
import mne, pywt
import random
from tqdm import tqdm

def denoise_signal(signal, wavelet='db1', thresholding='soft', level=4):
    # Decompose the signal using wavelet transform
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Threshold the wavelet coefficients
    threshold = np.std(coeffs[-1]) / 0.6745  # Adjust this threshold value as needed
    coeffs = [pywt.threshold(c, threshold, mode=thresholding) for c in coeffs]

    # Reconstruct the denoised signal
    denoised_signal = pywt.waverec(coeffs, wavelet)

    return denoised_signal

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

# Arguments
verbose = True
sample_dur = 0.5
datasets = [   #Chan >= 30
    # Motor imagery
    datasets.BNCI2014_001(),
    # datasets.BNCI2015_004(),
    # datasets.Cho2017(),
    #datasets.GrosseWentrup2009(),
    #datasets.Lee2019_MI(),
    #datasets.Ofner2017(),
    #datasets.PhysionetMI(),
    #datasets.Schirrmeister2017(),
    # datasets.Shin2017A(accept=True),
    #datasets.Shin2017B(),
    #datasets.Weibo2014(),
    
    # P300/ERP
    #datasets.BI2014b(),
    #datasets.BI2015a(),
    #datasets.BI2015b(),
    #datasets.EPFLP300(),
    #datasets.Huebner2017(),
    #datasets.Huebner2018(),
    #datasets.Lee2019_ERP(),
    #datasets.Sosulski2019(),

    # SSVEP
    #datasets.Lee2019_SSVEP(),
    #datasets.MAMEM1(),
    #datasets.MAMEM2(),
    #datasets.Wang2016(),
    ]

# Load the datasets in cache (optional)
start_time = time.time()
for dset in datasets:
    data = dset.get_data(subjects=dset.subject_list, cache_config=dict(path="./mne_data"))
end_time = time.time()
if verbose:
    print('-'*80+'\n'+f"Datasets loaded in {end_time-start_time:.2f} s")

# Define the folder where you want to save the data
output_folder_train = 'moabb_data_train_wavlet'
output_folder_val = 'moabb_data_val_wavlet'

# Create the output folder if it doesn't exist
os.makedirs(output_folder_train, exist_ok=True)
os.makedirs(output_folder_val, exist_ok=True)

history = {
    'datasets': [],
    'n_subjects': 0,
    'n_sessions': 0,
    'n_runs': 0,
    'n_samples': 0
    }
last_update = history.copy()

for dset in datasets:
    history['datasets'].append(dset.code)
    start_dset = time.time()

    split = int(len(dset.subject_list)*0.8)
    train_subject_list = dset.subject_list[:split]
    val_subject_list = dset.subject_list[split:]

    random.shuffle(train_subject_list)
    random.shuffle(val_subject_list)

    for subject in train_subject_list:
        history['n_subjects'] += 1

        # Load data
        data = dset.get_data(subjects=[subject])

        for session in list(data[subject].keys()):
            history['n_sessions'] += 1

            runs = list(data[subject][session].keys())
            ind = 1
            for run in tqdm(runs, total=len(runs), desc=f"Train {ind}/{len(runs)}"):
                ind += 1
                history['n_runs'] += 1

                raw = data[subject][session][run]

                # Separating wave into alpha and beta bands
                alpha_beta = raw.copy()
                alpha_beta.filter(8., 30., fir_design='firwin', skip_by_annotation='edge') 
                
                # Independent component analysis to identify and remove noise components
                contains_eog = False
                contains_emg = False
                for ch_info in raw.info['chs']:
                    if 'EOG' in str(ch_info['ch_name']):
                        contains_eog = True
                        break
                    if 'EMG' in str(ch_info['ch_name']):
                        contains_emg = True
                        break

                ica_transform = None
                if contains_eog:
                    ica = mne.preprocessing.ICA(method='infomax').fit(raw)
                    reject = dict(mag=5e-12, grad=4000e-13)
                    ica.fit(raw, reject=reject);
                    eog_average = mne.preprocessing.create_eog_epochs(raw).average()
                    eog_epochs = mne.preprocessing.create_eog_epochs(raw)
                    eog_inds, scores = ica.find_bads_eog(eog_epochs)
                    ica.exclude.extend(eog_inds)
                    ica_transform = raw.copy()
                    ica.apply(ica_transform)      
                elif contains_emg:
                    ica = mne.preprocessing.ICA(method='infomax').fit(raw)
                    reject = dict(mag=5e-12, grad=4000e-13)
                    ica.fit(raw, reject=reject);
                    emg_average = mne.preprocessing.create_emg_epochs(raw).average()
                    emg_epochs = mne.preprocessing.create_emg_epochs(raw)
                    emg_inds, scores = ica.find_bads_emg(emg_epochs)
                    ica.exclude.extend(emg_inds)
                    ica_transform = raw.copy()
                    ica.apply(ica_transform)

                def extract_npy(wave):
                    # Get EEG channels
                    eeg_ch_names = []
                    for ch_info in wave.info['chs']:
                        if 'EEG' in str(ch_info['coil_type']):
                            eeg_ch_names.append(ch_info['ch_name'])
                    assert len(eeg_ch_names) > 0, f'No EEG channels found for data[{subject}][{session}][{run}]'
    
                    # Get EEG data
                    eeg_ch_idx = np.where(np.isin(wave.info['ch_names'], eeg_ch_names))[0]
                    eeg_data = wave.get_data()[eeg_ch_idx]
    
                    # Extract samples in eeg_data
                    n_points = int(sample_dur * wave.info['sfreq'])
                    eeg_data = sliding_window_view(eeg_data, n_points, axis=1)[:,::n_points].transpose(1,0,2)

                    return eeg_data

                raw_npy = extract_npy(raw)
                alpha_beta_npy = extract_npy(alpha_beta)
                if contains_eog or contains_emg:
                    ica_npy = extract_npy(ica_transform)
                    
                # Save samples as .npy files shape (n_samples, n_channels, n_points)
                n_ch = 15
                for i, raw_sample in enumerate(raw_npy):
                    alpha_beta_sample = alpha_beta_npy[i]
                    if contains_eog or contains_emg:
                        ica_transform_sample = ica_npy[i]
                        
                    # Wavelet transform
                    wavelet_sample = denoise_signal(raw_sample, wavelet='db2', thresholding='soft', level=3)
                    if wavelet_sample.shape[1] == raw_sample.shape[1] + 1:
                        wavelet_sample = wavelet_sample[:,:-1]
                        
                    if contains_eog or contains_emg:
                        sample = np.array((ica_transform_sample[:n_ch], alpha_beta_sample[:n_ch], wavelet_sample[:n_ch])).reshape(n_ch * 3, -1)
                    else:
                        sample = np.array((raw_sample[:n_ch], alpha_beta_sample[:n_ch], wavelet_sample[:n_ch])).reshape(n_ch * 3, -1)
                    np.save(os.path.join(output_folder_train, f"{history['n_samples']}"), sample)
                    history['n_samples'] += 1

    for subject in val_subject_list:
        history['n_subjects'] += 1

        # Load data
        data = dset.get_data(subjects=[subject])

        for session in list(data[subject].keys()):
            history['n_sessions'] += 1

            ind = 1
            runs = list(data[subject][session].keys())
            for run in tqdm(runs, total=len(runs), desc=f"Val {ind}/{len(runs)}"):
                ind += 1
                history['n_runs'] += 1

                raw = data[subject][session][run]

                # Separating wave into alpha and beta bands
                alpha_beta = raw.copy()
                alpha_beta.filter(8., 30., fir_design='firwin', skip_by_annotation='edge') 
                
                # Independent component analysis to identify and remove noise components
                contains_eog = False
                contains_emg = False
                for ch_info in raw.info['chs']:
                    if 'EOG' in str(ch_info['ch_name']):
                        contains_eog = True
                        break
                    if 'EMG' in str(ch_info['ch_name']):
                        contains_emg = True
                        break
                ica_transform = None
                if contains_eog:
                    ica = mne.preprocessing.ICA(method='infomax').fit(raw)
                    reject = dict(mag=5e-12, grad=4000e-13)
                    ica.fit(raw, reject=reject);
                    eog_average = mne.preprocessing.create_eog_epochs(raw).average()
                    eog_epochs = mne.preprocessing.create_eog_epochs(raw)
                    eog_inds, scores = ica.find_bads_eog(eog_epochs)
                    ica.exclude.extend(eog_inds)
                    ica_transform = raw.copy()
                    ica.apply(ica_transform)      
                elif contains_emg:
                    ica = mne.preprocessing.ICA(method='infomax').fit(raw)
                    reject = dict(mag=5e-12, grad=4000e-13)
                    ica.fit(raw, reject=reject);
                    emg_average = mne.preprocessing.create_emg_epochs(raw).average()
                    emg_epochs = mne.preprocessing.create_emg_epochs(raw)
                    emg_inds, scores = ica.find_bads_emg(emg_epochs)
                    ica.exclude.extend(emg_inds)
                    ica_transform = raw.copy()
                    ica.apply(ica_transform)

                def extract_npy(wave):
                    # Get EEG channels
                    eeg_ch_names = []
                    for ch_info in wave.info['chs']:
                        if 'EEG' in str(ch_info['coil_type']):
                            eeg_ch_names.append(ch_info['ch_name'])
                    assert len(eeg_ch_names) > 0, f'No EEG channels found for data[{subject}][{session}][{run}]'
    
                    # Get EEG data
                    eeg_ch_idx = np.where(np.isin(wave.info['ch_names'], eeg_ch_names))[0]
                    eeg_data = wave.get_data()[eeg_ch_idx]
    
                    # Extract samples in eeg_data
                    n_points = int(sample_dur * wave.info['sfreq'])
                    eeg_data = sliding_window_view(eeg_data, n_points, axis=1)[:,::n_points].transpose(1,0,2)

                    return eeg_data

                raw_npy = extract_npy(raw)
                alpha_beta_npy = extract_npy(alpha_beta)
                if contains_eog or contains_emg:
                    ica_npy = extract_npy(ica_transform)
                    
                # Save samples as .npy files shape (n_samples, n_channels, n_points)
                n_ch = 15
                for i, raw_sample in enumerate(raw_npy):
                    alpha_beta_sample = alpha_beta_npy[i]
                    if contains_eog or contains_emg:
                        ica_transform_sample = ica_npy[i]
                        
                    # Wavelet transform
                    wavelet_sample = denoise_signal(raw_sample, wavelet='db2', thresholding='soft', level=3)
                    if wavelet_sample.shape[1] == raw_sample.shape[1] + 1:
                        wavelet_sample = wavelet_sample[:,:-1]
                        
                    if contains_eog or contains_emg:
                        sample = np.array((ica_transform_sample[:n_ch], alpha_beta_sample[:n_ch], wavelet_sample[:n_ch])).reshape(n_ch * 3, -1)
                    else:
                        sample = np.array((raw_sample[:n_ch], alpha_beta_sample[:n_ch], wavelet_sample[:n_ch])).reshape(n_ch * 3, -1)
                    np.save(os.path.join(output_folder_val, f"{history['n_samples']}"), sample)
                    history['n_samples'] += 1
                    

    end_dset = time.time()
    if verbose:
        n_new_subjects = history['n_subjects'] - last_update['n_subjects']
        n_new_samples = history['n_samples'] - last_update['n_samples']
        print('-'*80)
        print(f"{dset.code:>20} | {n_new_subjects:>4} subjects | {n_new_samples:>6} samples extracted. ({end_dset-start_dset:.2f} s)")
        print('-'*80)
        last_update = history.copy()

if verbose:
    print(f"Total: {len(history['datasets'])} datasets | {history['n_subjects']} subjects | {history['n_samples']} samples extracted. ({time.time()-start_time:.2f} s)")
