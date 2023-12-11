import os 
import time
import warnings
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import moabb
from moabb import datasets
import random
from tqdm import tqdm


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
    # datasets.GrosseWentrup2009(),
    # datasets.Lee2019_MI(),
    # datasets.Ofner2017(),
    # datasets.PhysionetMI(),
    # datasets.Schirrmeister2017(),
    # datasets.Shin2017A(accept=True),
    # datasets.Shin2017B(),
    # datasets.Weibo2014(),
    
    # # P300/ERP
    # datasets.BI2014b(),
    # datasets.BI2015a(),
    # datasets.BI2015b(),
    # datasets.EPFLP300(),
    # datasets.Huebner2017(),
    # datasets.Huebner2018(),
    # datasets.Lee2019_ERP(),
    # datasets.Sosulski2019(),

    # # SSVEP
    # datasets.Lee2019_SSVEP(),
    # datasets.MAMEM1(),
    # datasets.MAMEM2(),
    # datasets.Wang2016(),
    ]

# Load the datasets in cache (optional)
start_time = time.time()
for dset in datasets:
    data = dset.get_data(subjects=dset.subject_list)
end_time = time.time()
if verbose:
    print('-'*80+'\n'+f"Datasets loaded in {end_time-start_time:.2f} s")

# Define the folder where you want to save the data
output_folder_train = 'moabb_data_train'
output_folder_val = 'moabb_data_val'

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

            ind = 1
            runs = list(data[subject][session].keys())
            for run in tqdm(runs, total=len(runs), desc=f"Train {ind}/{len(runs)}"):
                ind += 1
                history['n_runs'] += 1

                raw = data[subject][session][run]

                # Get EEG channels
                eeg_ch_names = []
                for ch_info in raw.info['chs']:
                    if 'EEG' in str(ch_info['coil_type']):
                        eeg_ch_names.append(ch_info['ch_name'])
                assert len(eeg_ch_names) > 0, f'No EEG channels found for data[{subject}][{session}][{run}]'

                # Get EEG data
                eeg_ch_idx = np.where(np.isin(raw.info['ch_names'], eeg_ch_names))[0]
                eeg_data = raw.get_data()[eeg_ch_idx]

                # Extract samples in eeg_data
                n_points = int(sample_dur * raw.info['sfreq'])
                eeg_data = sliding_window_view(eeg_data, n_points, axis=1)[:,::n_points].transpose(1,0,2)
                
                # Save samples as .npy files shape (n_samples, n_channels, n_points)
                for i, sample in enumerate(eeg_data):
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

                # Get EEG channels
                eeg_ch_names = []
                for ch_info in raw.info['chs']:
                    if 'EEG' in str(ch_info['coil_type']):
                        eeg_ch_names.append(ch_info['ch_name'])
                assert len(eeg_ch_names) > 0, f'No EEG channels found for data[{subject}][{session}][{run}]'

                # Get EEG data
                eeg_ch_idx = np.where(np.isin(raw.info['ch_names'], eeg_ch_names))[0]
                eeg_data = raw.get_data()[eeg_ch_idx]

                # Extract samples in eeg_data
                n_points = int(sample_dur * raw.info['sfreq'])
                eeg_data = sliding_window_view(eeg_data, n_points, axis=1)[:,::n_points].transpose(1,0,2)
                
                # Save samples as .npy files shape (n_samples, n_channels, n_points)
                for i, sample in enumerate(eeg_data):
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
