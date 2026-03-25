import tempfile
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

import mne
import ipywidgets
import pyvistaqt

data_path = Path(mne.datasets.sample.data_path(verbose=False))
sample_dir = data_path / "MEG" / "sample"
subjects_dir = data_path / "subjects"
print("Checkpoint 1")
raw_path = sample_dir / "sample_audvis_filt-0-40_raw.fif"
raw = mne.io.read_raw(raw_path)
raw.pick(picks=["eeg", "eog", "stim"]).crop(tmax=60).load_data()
print("Checkpoint 2")
report = mne.Report(title="Raw example")

report.add_raw(raw=raw, title="Raw", psd=False)  # omit PSD plot
report.save("report_raw.html", overwrite=True)

# Events
events_path = sample_dir / "sample_audvis_filt-0-40_raw-eve.fif"
events = mne.find_events(raw=raw)
sfreq = raw.info["sfreq"]

report = mne.Report(title="Events example")
report.add_events(events=events_path, title="Events from Path", sfreq=sfreq)
report.add_events(events=events, title='Events from "events"', sfreq=sfreq)
report.save("report_events.html", overwrite=True)

# Epochs
# Epochs refer to essentially a loop of a machine learning model, i.e. running it again to correct.
# Similar to alphas, help with fitting.
# Too few leads to underfitting, too many leads to overfitting.
event_id = {
    "auditory/left": 1,
    "auditory/right": 2,
    "visual/left": 3,
    "visual/right": 4,
    "face": 5,
    "buttonpress": 32,
}

metadata, _, _ = mne.epochs.make_metadata(
    events=events, event_id=event_id, tmin=-0.2, tmax=0.5, sfreq=raw.info["sfreq"]
)
epochs = mne.Epochs(raw=raw, events=events, event_id=event_id, metadata=metadata)

report = mne.Report(title="Epochs example")
report.add_epochs(epochs=epochs, title='Epochs from "epochs"')
report.save("report_epochs.html", overwrite=True)

# Evoked
evoked_path = sample_dir / "sample_audvis-ave.fif"
cov_path = sample_dir / "sample_audvis-cov.fif"

evokeds = mne.read_evokeds(evoked_path, baseline=(None, 0))
evokeds_subset = evokeds[:2]  # The first two
for evoked in evokeds_subset:
    evoked.pick("eeg")  # just for speed of plotting

report = mne.Report(title="Evoked example")
report.add_evokeds(
    evokeds=evokeds_subset,
    titles=["evoked 1", "evoked 2"],  # Manually specify titles
    noise_cov=cov_path,
    n_time_points=5,
)
report.save("report_evoked.html", overwrite=True)

# Covariance, i.e. Noise
# Covariance matrix can be implemented in a function with two noisy, random variables.
# Can be used to determine a linear trend of two major variables.
cov_path = sample_dir / "sample_audvis-cov.fif"

report = mne.Report(title="Covariance example")
report.add_covariance(cov=cov_path, info=raw_path, title="Covariance")
report.save("report_cov.html", overwrite=True)

# Projection
# Shift between higher and lower dimensional spaces.
ecg_proj_path = sample_dir / "sample_audvis_ecg-proj.fif"
report = mne.Report(title="Projectors example")
report.add_projs(info=raw_path, title="Projs from info")

# Now a joint plot
events = mne.read_events(sample_dir / "sample_audvis_ecg-eve.fif")
raw_full = mne.io.read_raw(sample_dir / "sample_audvis_raw.fif").crop(0, 60).load_data()
ecg_evoked = mne.Epochs(
    raw=raw_full,
    events=events,
    tmin=-0.5,
    tmax=0.5,
    baseline=(None, None),
).average()
report.img_max_width = None  # do not constrain image width
report.add_projs(
    info=ecg_evoked,
    projs=ecg_proj_path,
    title="ECG projs from path",
    joint=True,  # use joint version of the plot
)
report.save("report_projs.html", overwrite=True)
del raw_full, events, ecg_evoked

# Independant Component Analysis (ICA)
# i.e. Isolate distinct sources from given data.
ica = mne.preprocessing.ICA(
    n_components=5,  # fit 5 ICA components
    fit_params=dict(tol=0.01),  # assume very early on that ICA has converged
)

ica.fit(inst=raw)

# create epochs based on EOG events, find EOG artifacts in the data via pattern
# matching, and exclude the EOG-related ICA components
eog_epochs = mne.preprocessing.create_eog_epochs(raw=raw)
eog_components, eog_scores = ica.find_bads_eog(
    inst=eog_epochs,
    ch_name="EEG 001",  # a channel close to the eye
    threshold=1,  # lower than the default threshold
)
ica.exclude = eog_components

report = mne.Report(title="ICA example")
report.add_ica(
    ica=ica,
    title="ICA cleaning",
    picks=ica.exclude,  # plot the excluded EOG components
    inst=raw,
    eog_evoked=eog_epochs.average(),
    eog_scores=eog_scores,
    n_jobs=None,  # could be increased!
)
report.save("report_ica.html", overwrite=True)

# MRI Slices using BEM (Boundary Element Model)
report = mne.Report(title="BEM example")
report.add_bem(
    subject="sample",
    subjects_dir=subjects_dir,
    title="MRI & BEM",
    decim=40,
    width=256,
)
report.save("report_mri_and_bem.html", overwrite=True)

# Sensor Alignment -> Coregistration
# Can determine placement of sensors on a 3d Human Head using readings taken from EEG.
trans_path = sample_dir / "sample_audvis_raw-trans.fif"

report = mne.Report(title="Coregistration example")
report.add_trans(
    trans=trans_path,
    info=raw_path,
    subject="sample",
    subjects_dir=subjects_dir,
    alpha=1.0,
    title="Coregistration",
)
report.save("report_coregistration.html", overwrite=True)

# Forward Solutions
# 