import os
import glob
import numpy as np
import mne

RAW_DIR = "raw_data/sleep-edf"
OUT_DIR = "data/processed"

# Target channels required for the model (3 channels)
CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]

def find_subject_pairs(raw_dir):
    psg_files = sorted(glob.glob(os.path.join(raw_dir, "*PSG.edf")))
    hyp_files = sorted(glob.glob(os.path.join(raw_dir, "*Hypnogram.edf")))
    
    pairs = []
    for psg in psg_files:
        filename = os.path.basename(psg)
        subject_prefix = filename[:6]  # e.g., SC4001
        
        matching_hyp = [h for h in hyp_files if os.path.basename(h).startswith(subject_prefix)]
        if matching_hyp:
            pairs.append((psg, matching_hyp[0], subject_prefix))
    return pairs

def load_subject(psg_path, hyp_path):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    
    # Clean channel names to avoid trailing whitespace mismatches
    raw.rename_channels(lambda name: name.strip())
    
    # Defensive channel verification
    available = raw.ch_names
    missing = [ch for ch in CHANNELS if ch not in available]
    if missing:
        raise ValueError(
            f"Missing expected channels {missing} in {psg_path}. "
            f"Available channels: {available}"
        )
        
    # Explicitly pick only the target 3 channels
    raw.pick(CHANNELS)
    
    annotations = mne.read_annotations(hyp_path)
    raw.set_annotations(annotations, emit_warning=False)
    return raw, annotations

def annotations_to_epoch_labels(annotations, total_duration_sec, epoch_duration=30):
    stage_map = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,  # Merge Stage 3 & 4 into N3
        'Sleep stage R': 4,
        'Sleep stage ?': -1,
        'Movement time': -1
    }
    
    num_epochs = int(total_duration_sec // epoch_duration)
    labels = np.full(num_epochs, -1, dtype=int)
    
    for ann in annotations:
        description = ann['description']
        if description in stage_map:
            label = stage_map[description]
            onset = ann['onset']
            duration = ann['duration']
            
            start_idx = int(onset // epoch_duration)
            end_idx = int((onset + duration) // epoch_duration)
            end_idx = min(end_idx, num_epochs)
            
            labels[start_idx:end_idx] = label
            
    return labels

def trim_wake_padding(labels, wake_label=0, padding_epochs=60):
    sleep_indices = np.where((labels != wake_label) & (labels != -1))[0]
    if len(sleep_indices) == 0:
        return np.array([]), (0, 0)
    
    first_sleep = sleep_indices[0]
    last_sleep = sleep_indices[-1]
    
    start_idx = max(0, first_sleep - padding_epochs)
    end_idx = min(len(labels), last_sleep + padding_epochs + 1)
    
    return labels[start_idx:end_idx], (start_idx, end_idx)

def segment_signal(raw, start_epoch, end_epoch, epoch_duration=30):
    sfreq = raw.info['sfreq']
    start_sec = start_epoch * epoch_duration
    end_sec = end_epoch * epoch_duration
    
    data, _ = raw[:, int(start_sec * sfreq):int(end_sec * sfreq)]
    samples_per_epoch = int(epoch_duration * sfreq)
    total_samples = data.shape[1]
    num_epochs = total_samples // samples_per_epoch
    
    data = data[:, :num_epochs * samples_per_epoch]
    segmented = data.T.reshape(num_epochs, samples_per_epoch, data.shape[0]).transpose(0, 2, 1)
    return segmented
