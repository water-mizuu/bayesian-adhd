import numpy as np
import pandas as pd
import mne as mne
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.signal import welch


SAMPLE_RATE = 128 # Hz

def apply_ica_cleaning(df: pd.DataFrame, sfreq: float):
    """Apply ICA to remove muscle and eye artifacts from EEG dataframe."""
    ch_names = df.columns.tolist()
    ch_types = ["eeg"] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(df.to_numpy().T, info, verbose=False)

    # --- Filter for ICA (1–80 Hz typical) ---
    raw.filter(1.0, 60.0, fir_design="firwin", verbose=False)

    # --- Fit ICA ---
    ica = mne.preprocessing.ICA(n_components=0.95, method="fastica", random_state=42, verbose=False)
    ica.fit(raw, picks="eeg", decim=3)

    # --- Detect EOG-like (blink) components if present ---
    try:
        eog_inds, eog_scores = ica.find_bads_eog(raw)
    except Exception:
        eog_inds = []

    # --- Detect muscle-like components (high-frequency ratio heuristic) ---
    sources = ica.get_sources(raw).get_data()
    n_ic = sources.shape[0]
    sfreq = raw.info["sfreq"]

    from scipy.signal import welch
    muscle_candidates = []
    for ic_idx in range(n_ic):
        f, Pxx = welch(sources[ic_idx], sfreq, nperseg=512)
        hf_mask = (f >= 20)
        hf_ratio = Pxx[hf_mask].sum() / np.sum(Pxx)
        if hf_ratio > 0.3:  # threshold – tune as needed
            muscle_candidates.append(ic_idx)

    exclude = list(set(eog_inds + muscle_candidates))
    ica.exclude = exclude
    if len(exclude) > 0:
        print(f"ICA removed components: {exclude}")
        raw_clean = ica.apply(raw.copy())
    else:
        print("No ICA components removed.")
        raw_clean = raw

    # --- Convert back to pandas DataFrame ---
    cleaned = pd.DataFrame(raw_clean.get_data().T, columns=ch_names)
    return cleaned

def compute_band_power(signal, sfreq):
    """Compute average power using Parseval's theorem."""
    fft_vals = np.fft.rfft(signal)
    psd = np.abs(fft_vals) ** 2
    return np.mean(psd)

def compute_eeg_band_stats(df: pd.DataFrame, sfreq, band_limits):
    """Compute per-band power and ratios."""
    # electrode_columns = df.select_dtypes(include=[np.number]).columns
    electrode_columns = pd.Index(["Fp1", "Fp2", "F3", "F4", "Fz", "C3", "C4"])
    stats = {}

    for (lower, upper), name in band_limits:
        band_df = apply_filter(df, lower, upper)
        stats[name] = {}

        for electrode in electrode_columns:
            signal = band_df[electrode].to_numpy()
            stats[name][electrode] = compute_band_power(signal, sfreq)

    # Convert to DataFrame
    band_powers = pd.DataFrame(stats)
    total_power = band_powers.sum(axis=1)
    relative_powers = band_powers.divide(total_power, axis=0)

    # Theta/Beta ratio (per electrode)
    tbr = band_powers["theta"] / band_powers["beta"]

    # Return everything for later analysis
    return band_powers, relative_powers, tbr

def compute_tbr_over_time(df, sfreq, window_sec=2, overlap=0.5):
    """
    Compute Theta/Beta ratio for each electrode over overlapping time windows.
    """

    electrodes = df.select_dtypes(include=[np.number]).columns
    window_len = int(window_sec * sfreq)
    step = int(window_len * (1 - overlap))  # move by (1 - overlap) * window
    if step <= 0:
        raise ValueError("Overlap too high; resulting step size <= 0")

    n_windows = (len(df) - window_len) // step + 1

    times = []
    tbr_matrix = np.zeros((len(electrodes), n_windows))

    for i in range(n_windows):
        start = i * step
        end = start + window_len
        segment = df.iloc[start:end]
        t_sec = start / sfreq
        times.append(t_sec)

        for e_idx, electrode in enumerate(electrodes):
            signal = segment[electrode].to_numpy()
            freqs, psd = welch(signal, sfreq, nperseg=window_len // 2)

            theta_mask = (freqs >= 4) & (freqs < 8)
            beta_mask = (freqs >= 13) & (freqs < 30)

            theta_power = np.trapezoid(psd[theta_mask], freqs[theta_mask])
            beta_power = np.trapezoid(psd[beta_mask], freqs[beta_mask]) + 1e-6  # avoid div-by-zero

            tbr_matrix[e_idx, i] = theta_power / beta_power

    tbr_df = pd.DataFrame(tbr_matrix, index=electrodes, columns=np.round(times, 2))
    return tbr_df

def plot_tbr_heatmap(tbr_df, title="Theta/Beta Ratio Over Time"):
    plt.figure(figsize=(12, 6))
    sns.heatmap(tbr_df, cmap="magma", cbar_kws={'label': 'Theta/Beta Ratio'})
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Electrodes")
    plt.tight_layout()
    return plt.gcf()


def apply_filter(df: pd.DataFrame, lower=0.5, upper=30):
    electrode_columns = df.select_dtypes(include=[np.number]).columns

    n = len(df)
    freqs = np.fft.rfftfreq(n, d=1 / SAMPLE_RATE)
    filtered_df = pd.DataFrame(index=df.index)

    for electrode in electrode_columns:
        signal = df[electrode].to_numpy()
        fft_vals = np.fft.rfft(signal)
        fft_filter = (freqs >= lower) & (freqs <= upper)
        fft_vals[~fft_filter] = 0
        filtered_signal = np.fft.irfft(fft_vals, n=n)
        filtered_df[electrode] = filtered_signal

    return filtered_df

if __name__ == "__main__":
    ## First, we load the "recordings"
    df = pd.read_csv("./adhdata.csv")
    export_path = Path("validation_data")
    export_path.mkdir(parents=True, exist_ok=True)

    summary_records = []
    for i in range(df['ID'].nunique()):
        id = df['ID'].unique()[i]
        print(f"Processing id {id}")

        subject_export_path = export_path.joinpath(f"{id}")
        subject_export_path.mkdir(parents=True, exist_ok=True)

        export = lambda x: subject_export_path.joinpath(x)

        target_recording: pd.DataFrame = None
        target_recording = df[df['ID'] == id]
        classification = target_recording['Class'].unique()[0]

        target_recording = target_recording.drop(['ID', 'Class'], axis=1)
        target_recording.to_csv(export(f"{id}.csv"), index=False)

        cleaned_recording = apply_ica_cleaning(target_recording, SAMPLE_RATE)
        cleaned_recording.to_csv(export(f"{id}_cleaned.csv"), index=False)

        ch_names = cleaned_recording.columns.tolist()
        ch_types = ['eeg'] * len(ch_names)

        frequencies = [
            ((0, 80), "raw"),
            ((0.5, 30), "filtered"),
            ((0.5, 4), "delta"),
            ((4, 8), "theta"),
            ((8, 13), "alpha"),
            ((13, 30), "beta"),
        ]

        for (lower, upper), name in frequencies:
            filtered_data = apply_filter(cleaned_recording, lower=lower, upper=upper).to_numpy().T
            # Visualize the cleaned EEG data.
            info = mne.create_info(ch_names=ch_names, sfreq=SAMPLE_RATE, ch_types=ch_types)
            filtered = mne.io.RawArray(filtered_data, info)

            fig = filtered.plot(scalings='auto', show=False)
            fig.savefig(export(f"{name}_recording.png"), dpi=300)
            plt.close(fig)

            fig = filtered.compute_psd(fmax=40).plot(show=False)
            fig.savefig(export(f"{name}_psd.png"), dpi=300)
            plt.close(fig)

        band_powers, rel_powers, tbr = \
            compute_eeg_band_stats(cleaned_recording, SAMPLE_RATE, frequencies)

        # Compute group-level averages
        mean_band_power = band_powers.mean(axis=0)
        mean_rel_power = rel_powers.mean(axis=0)
        mean_tbr = tbr.mean()

        summary_records.append({
            "ID": id,
            "Class": classification,
            "TBR": mean_tbr,
        })

        print("Average band power per band:")
        print(mean_band_power)
        print("\nRelative band power per band:")
        print(mean_rel_power)
        print(f"\nOverall mean Theta/Beta ratio: {mean_tbr:.3f}")

        filtered_data = apply_filter(cleaned_recording, lower=0.5, upper=30.0).to_numpy().T
        tbr_df = compute_tbr_over_time(
            df=pd.DataFrame(filtered_data.T, columns=ch_names).iloc[2:-2],
            sfreq=SAMPLE_RATE,
            window_sec=2,
            overlap=0.5
        )
        fig = plot_tbr_heatmap(tbr_df, title=f"{name.capitalize()} ({classification}) Band — Theta/Beta Over Time - {mean_tbr}")
        fig.savefig(export(f"{name}_tbr_heatmap.png"), dpi=300)
        plt.close(fig)

    summary_df = pd.DataFrame(summary_records)

    sns.violinplot(data=summary_df, x="Class", y="TBR", inner="box", palette="Set2")
    sns.swarmplot(data=summary_df, x="Class", y="TBR", color="k", alpha=0.5)
    plt.title("Theta/Beta Ratio by Classification")
    plt.ylabel("Theta/Beta Ratio (TBR)")
    plt.xlabel("Class")
    plt.show()
