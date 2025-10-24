import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import welch
from scipy import stats


SAMPLE_RATE = 128  # Hz


def compute_frequency_heatmap(df: pd.DataFrame, sfreq: float, nperseg: int = 512):
    """
    Compute power spectral density for each electrode across frequency bins.
    
    Parameters:
    -----------
    df : pd.DataFrame
        EEG data with electrodes as columns
    sfreq : float
        Sampling frequency in Hz
    nperseg : int
        Length of each segment for Welch's method
    
    Returns:
    --------
    freqs : np.ndarray
        Frequency bins
    psd_matrix : np.ndarray
        Power spectral density matrix (electrodes x frequencies)
    electrodes : list
        List of electrode names
    """
    electrodes = df.columns.tolist()
    n_electrodes = len(electrodes)
    
    # Compute PSD for first electrode to get frequency bins
    signal = df[electrodes[0]].to_numpy()
    freqs, _ = welch(signal, sfreq, nperseg=nperseg)
    n_freqs = len(freqs)
    
    # Initialize PSD matrix
    psd_matrix = np.zeros((n_electrodes, n_freqs))
    
    # Compute PSD for each electrode
    for i, electrode in enumerate(electrodes):
        signal = df[electrode].to_numpy()
        _, psd = welch(signal, sfreq, nperseg=nperseg)
        psd_matrix[i, :] = psd
    
    return freqs, psd_matrix, electrodes


def z_normalize_per_subject(psd_matrix: np.ndarray) -> np.ndarray:
    """
    Apply z-normalization (standardization) per subject.
    Each electrode's PSD values are normalized to have mean=0 and std=1.
    
    Parameters:
    -----------
    psd_matrix : np.ndarray
        Power spectral density matrix (electrodes x frequencies)
    
    Returns:
    --------
    z_normalized : np.ndarray
        Z-normalized PSD matrix
    """
    # Convert to dB scale first (log transform)
    psd_db = 10 * np.log10(psd_matrix + 1e-12)  # Add small value to avoid log(0)
    
    # Z-normalize across all frequencies for each electrode
    z_normalized = np.zeros_like(psd_db)
    for i in range(psd_db.shape[0]):
        z_normalized[i, :] = stats.zscore(psd_db[i, :])
    
    return z_normalized


def plot_frequency_heatmap(freqs: np.ndarray, 
                          psd_matrix: np.ndarray, 
                          electrodes: list,
                          title: str = "Frequency Domain Heatmap (Z-normalized)",
                          freq_max: float = 40.0,
                          figsize: tuple = (14, 8)):
    """
    Create a heatmap visualization of frequency domain power.
    
    Parameters:
    -----------
    freqs : np.ndarray
        Frequency bins
    psd_matrix : np.ndarray
        Z-normalized PSD matrix (electrodes x frequencies)
    electrodes : list
        List of electrode names
    title : str
        Plot title
    freq_max : float
        Maximum frequency to display (Hz)
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Filter frequencies up to freq_max
    freq_mask = freqs <= freq_max
    freqs_filtered = freqs[freq_mask]
    psd_filtered = psd_matrix[:, freq_mask]
    
    # Create DataFrame for seaborn
    df_plot = pd.DataFrame(
        psd_filtered,
        index=electrodes,
        columns=np.round(freqs_filtered, 2)
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df_plot,
        cmap='coolwarm',
        center=0,  # Center colormap at 0 (mean after z-normalization)
        cbar_kws={'label': 'Z-score (Power)'},
        ax=ax,
        xticklabels=20  # Show every 20th frequency label to avoid crowding
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Electrodes', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_average_frequency_profile(freqs: np.ndarray,
                                   psd_matrix: np.ndarray,
                                   electrodes: list,
                                   title: str = "Average Frequency Profile",
                                   freq_max: float = 40.0,
                                   figsize: tuple = (12, 6)):
    """
    Plot the average frequency profile across all electrodes.
    
    Parameters:
    -----------
    freqs : np.ndarray
        Frequency bins
    psd_matrix : np.ndarray
        Z-normalized PSD matrix (electrodes x frequencies)
    electrodes : list
        List of electrode names
    title : str
        Plot title
    freq_max : float
        Maximum frequency to display (Hz)
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    freq_mask = freqs <= freq_max
    freqs_filtered = freqs[freq_mask]
    psd_filtered = psd_matrix[:, freq_mask]
    
    # Compute mean and std across electrodes
    mean_psd = np.mean(psd_filtered, axis=0)
    std_psd = np.std(psd_filtered, axis=0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot mean with shaded std
    ax.plot(freqs_filtered, mean_psd, linewidth=2, label='Mean', color='navy')
    ax.fill_between(
        freqs_filtered,
        mean_psd - std_psd,
        mean_psd + std_psd,
        alpha=0.3,
        color='navy',
        label='±1 SD'
    )
    
    # Add vertical lines for frequency bands
    bands = [
        (0.5, 4, 'Delta', 'purple'),
        (4, 8, 'Theta', 'blue'),
        (8, 13, 'Alpha', 'green'),
        (13, 30, 'Beta', 'orange')
    ]
    
    for low, high, name, color in bands:
        if low <= freq_max:
            ax.axvspan(low, min(high, freq_max), alpha=0.1, color=color, label=name)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Z-score (Power)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def process_subject(subject_id: str, 
                   data_path: Path,
                   export_path: Path,
                   sfreq: float = SAMPLE_RATE,
                   use_cleaned: bool = True):
    """
    Process a single subject and generate frequency domain heatmaps.
    
    Parameters:
    -----------
    subject_id : str
        Subject ID (e.g., 'v107')
    data_path : Path
        Path to validation data directory
    export_path : Path
        Path to save output files
    sfreq : float
        Sampling frequency in Hz
    use_cleaned : bool
        Whether to use cleaned or raw data
    """
    print(f"Processing subject {subject_id}...")
    
    # Load data
    subject_path = data_path / subject_id
    if use_cleaned:
        csv_file = subject_path / f"{subject_id}_cleaned.csv"
    else:
        csv_file = subject_path / f"{subject_id}.csv"
    
    if not csv_file.exists():
        print(f"  Warning: {csv_file} not found, skipping...")
        return None
    
    df = pd.read_csv(csv_file)
    
    # Compute frequency domain representation
    freqs, psd_matrix, electrodes = compute_frequency_heatmap(df, sfreq)
    
    # Z-normalize per subject
    psd_normalized = z_normalize_per_subject(psd_matrix)
    
    # Create subject export path
    subject_export_path = export_path / subject_id
    subject_export_path.mkdir(parents=True, exist_ok=True)
    
    # Generate heatmap
    fig1 = plot_frequency_heatmap(
        freqs,
        psd_normalized,
        electrodes,
        title=f"Subject {subject_id} - Frequency Domain (Z-normalized)",
        freq_max=40.0
    )
    fig1.savefig(subject_export_path / f"{subject_id}_frequency_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Generate average profile
    fig2 = plot_average_frequency_profile(
        freqs,
        psd_normalized,
        electrodes,
        title=f"Subject {subject_id} - Average Frequency Profile",
        freq_max=40.0
    )
    fig2.savefig(subject_export_path / f"{subject_id}_frequency_profile.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"  Saved heatmaps to {subject_export_path}")
    
    return {
        'subject_id': subject_id,
        'freqs': freqs,
        'psd_normalized': psd_normalized,
        'electrodes': electrodes
    }


def process_all_subjects(data_path: Path, 
                        export_path: Path,
                        sfreq: float = SAMPLE_RATE):
    """
    Process all subjects in the validation data directory.
    
    Parameters:
    -----------
    data_path : Path
        Path to validation data directory
    export_path : Path
        Path to save output files
    sfreq : float
        Sampling frequency in Hz
    """
    export_path.mkdir(parents=True, exist_ok=True)
    
    # Get all subject directories
    subject_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    results = []
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        result = process_subject(subject_id, data_path, export_path, sfreq)
        if result is not None:
            results.append(result)
    
    print(f"\nProcessed {len(results)} subjects successfully.")
    return results


def create_group_average_heatmap(results: list,
                                 export_path: Path,
                                 freq_max: float = 40.0):
    """
    Create a group-average frequency heatmap across all subjects.
    
    Parameters:
    -----------
    results : list
        List of dictionaries containing subject results
    export_path : Path
        Path to save output files
    freq_max : float
        Maximum frequency to display (Hz)
    """
    if not results:
        print("No results to process for group average.")
        return
    
    print("\nCreating group-average heatmap...")
    
    # Stack all normalized PSDs
    all_psds = [r['psd_normalized'] for r in results]
    group_psd = np.mean(all_psds, axis=0)
    
    freqs = results[0]['freqs']
    electrodes = results[0]['electrodes']
    
    # Create group average heatmap
    fig = plot_frequency_heatmap(
        freqs,
        group_psd,
        electrodes,
        title=f"Group Average - Frequency Domain (n={len(results)} subjects)",
        freq_max=freq_max
    )
    fig.savefig(export_path / "group_average_frequency_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create group average profile
    fig2 = plot_average_frequency_profile(
        freqs,
        group_psd,
        electrodes,
        title=f"Group Average - Frequency Profile (n={len(results)} subjects)",
        freq_max=freq_max
    )
    fig2.savefig(export_path / "group_average_frequency_profile.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"Saved group average heatmaps to {export_path}")


if __name__ == "__main__":
    # Set paths
    data_path = Path("validation_data")
    export_path = Path("validation_data/frequency_heatmaps")
    
    # Process all subjects
    results = process_all_subjects(data_path, export_path, sfreq=SAMPLE_RATE)
    
    # Create group average
    create_group_average_heatmap(results, export_path, freq_max=40.0)
    
    print("\n✓ Frequency domain visualization complete!")
