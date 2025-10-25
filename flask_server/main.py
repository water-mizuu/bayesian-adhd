from flask import Flask, render_template, request, jsonify
from werkzeug.datastructures import FileStorage
from scaler import EEGScaler
from model import EEG_CNN_LSTM_HPO
from typing import TypedDict
from error_handle import error_handle

import torch

import io
import mne
import math
import pickle
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend

from threader import threaded


# Initialize the Flask app
app = Flask(__name__)

SAMPLE_RATE = 128             # Hz
WINDOW_SECONDS = 4            # window duration
OVERLAP = 0.5                 # 50% overlap
SAMPLES_PER_WINDOW = int(SAMPLE_RATE * WINDOW_SECONDS)
WINDOW_STEP = int(SAMPLES_PER_WINDOW * (1 - OVERLAP))    # step size for sliding window
TARGET_FREQUENCY_BINS = np.arange(2.0, 40.5, 0.5)        # target frequency bins
ALLOWED_EXTENSIONS = {'csv'}                             # Allowed file extensions


scaler: EEGScaler = None # This is late initialization.
model: EEG_CNN_LSTM_HPO = None # Late initialization.

class Hyperparameters(TypedDict):
    batch_size: int
    cnn_dense: int
    cnn_dropout: np.float64
    cnn_kernel_size_1: int
    cnn_kernel_size_2: int
    cnn_kernels_1: int
    cnn_kernels_2: int
    learning_rate: np.float64
    lstm_dense: int
    lstm_hidden_size: int
    lstm_layers: int
    optimizer: str

def allowed_file(filename: str):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_scaler():
    global scaler
    with open('exports/saved_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

def load_model():
    global model

    params: Hyperparameters = \
            {'batch_size': 80,
             'cnn_dense': 256,
             'cnn_dropout': np.float64(0.38218620920862145),
             'cnn_kernel_size_1': 5,
             'cnn_kernel_size_2': 5,
             'cnn_kernels_1': 48,
             'cnn_kernels_2': 32,
             'learning_rate': np.float64(0.0017576118123159641),
             'lstm_dense': 32,
             'lstm_hidden_size': 128,
             'lstm_layers': 3,
             'optimizer': 'rmsprop'}

    model = EEG_CNN_LSTM_HPO(
        cnn_kernels_1=params['cnn_kernels_1'],
        cnn_kernel_size_1=params['cnn_kernel_size_1'],
        cnn_kernels_2=params['cnn_kernels_2'],
        cnn_dropout=float(params['cnn_dropout']),
        cnn_dense=params['cnn_dense'],
        lstm_hidden_size=params['lstm_hidden_size'],
        lstm_layers=params['lstm_layers'],
        lstm_dense=params['lstm_dense'],
        dropout=float(params['cnn_dropout']),  # use cnn_dropout as a simple shared dropout param
        num_classes=2,
    )

    device = torch.device("cpu")
    weights = torch.load("exports/eeg_cnn_lstm_hpo.pth", weights_only=True, map_location=device)

    model.load_state_dict(weights)
    model.eval()

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def bandpass_df(df: pd.DataFrame, low=None, high=None, order=4, window_sec=1):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_samples = len(df)
    window_size = int(window_sec * SAMPLE_RATE)

    nyquist = 0.5 * SAMPLE_RATE
    if low is None and high is None:
        return df

    # Normalized cutoffs
    low_cut = low / nyquist if low else None
    high_cut = high / nyquist if high else None

    # Filter design
    if low_cut and high_cut:
        btype, Wn = 'band', [low_cut, high_cut]
    elif low_cut:
        btype, Wn = 'highpass', low_cut
    elif high_cut:
        btype, Wn = 'lowpass', high_cut

    b, a = butter(order, Wn, btype=btype)

    # Apply filter in 1-second windows
    for col in numeric_cols:
        signal = df[col].to_numpy()
        filtered = np.zeros_like(signal)

        for start in range(0, n_samples, window_size):
            end = min(start + window_size, n_samples)
            segment = signal[start:end]

            # Avoid errors for very short trailing segments
            if len(segment) < order * 3:
                filtered[start:end] = segment
                continue

            filtered[start:end] = filtfilt(b, a, segment, method="gust")

        df[col] = filtered

    return df

@threaded
def visualize_df(df: pd.DataFrame, eeg_type: str):
    low, high = (0.5, 40)    if eeg_type == "filtered" else \
                (0.5, 4)     if eeg_type == "delta" else \
                (4, 8)       if eeg_type == "theta" else \
                (8, 12)      if eeg_type == "alpha" else \
                (12, 30)     if eeg_type == "beta" else \
                (30, 40)     if eeg_type == "gamma" else \
                (None, None)

    ch_names = df.select_dtypes(include=[np.number]).columns.tolist()
    ch_types = ['eeg'] * len(ch_names)

    # df = df - df.mean()  # remove DC offset
    # df = df / df.abs().max() * 100  # rescale to ~±100 µV range for plotting
    df = bandpass_df(df, low=low, high=high)
    df = df.to_numpy().T

    # Visualize the cleaned EEG data.
    info = mne.create_info(ch_names=ch_names, sfreq=SAMPLE_RATE, ch_types=ch_types)
    print("INFO: ", info)

    _, sample_count = df.shape
    duration = sample_count / SAMPLE_RATE
    filtered = mne.io.RawArray(df, info)
    print(duration)
    buf = io.BytesIO()
    fig = filtered.plot(scalings='auto',
                        duration=30.0,
                        n_channels=len(ch_names),
                        show=False,
                        show_scrollbars=False,
                        show_options=False,
                        clipping=None,
                        block=False)

    fig.set_size_inches(math.ceil(duration / 6), 6)
    fig.savefig(buf, format="png", dpi=600)
    plt.close(fig)
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return f"data:image/png;base64,{img_base64}"


def process_window(window: pd.DataFrame, window_count: int):
    output = []
    electrode_columns = window.select_dtypes(include=[np.number]).columns

    n = len(window)
    original_freqs = np.fft.rfftfreq(n, d=1 / SAMPLE_RATE)

    # compute power spectra for all electrodes
    electrode_powers: dict[str, np.ndarray] = {}
    for electrode in electrode_columns:
        signal = window[electrode].to_numpy()
        fft_vals = np.fft.rfft(signal)
        power = np.abs(fft_vals) ** 2

        # interpolate to common freq bins
        electrode_powers[electrode] = np.interp(TARGET_FREQUENCY_BINS, original_freqs, power)

    # build rows: one per frequency bin
    for i, f in enumerate(TARGET_FREQUENCY_BINS):
        output.append({
            "window": window_count,
            "frequency": f,
            **{electrode: electrode_powers[electrode][i] for electrode in electrode_columns}
        })

    return output

def classify_df(df: pd.DataFrame):
    window_count = 0
    n_samples = len(df)
    output = []

    # Sliding window with overlap
    for start in range(0, n_samples - SAMPLES_PER_WINDOW + 1, WINDOW_STEP):
        window = df.iloc[start:start + SAMPLES_PER_WINDOW]
        for frequency in process_window(window, window_count):
            output.append(frequency)

        window_count += 1

    # Create the dataframe from the windows.
    windows_dataframe = pd.DataFrame(output)

    frequency_count = len(windows_dataframe['frequency'].unique())
    window_count = len(windows_dataframe['window'].unique())
    numeric_df = windows_dataframe.drop(['window', 'frequency'], axis=1)

    # shape: (windows, freqs, features)
    full_ndarray = numeric_df.values.reshape((window_count, frequency_count, numeric_df.shape[1]))
    full_ndarray = scaler.transform(full_ndarray)
    full_ndarray = full_ndarray[..., np.newaxis]

    with torch.no_grad():
        tensor = torch.tensor(full_ndarray, dtype=torch.float32).permute(0, 3, 1, 2)
        predictions = model(tensor).softmax(1).detach().numpy()
        adhd, control = np.sum(predictions, axis=0) / np.sum(predictions)

        if adhd > control:
            print(f"The subject is indicative of ADHD with {adhd * 100:.2f}% confidence.")

            return float(adhd)
        else:
            print(f"The subject is not indicative of ADHD with {control * 100:.2f}% confidence.")

            return float(-control)


def visualize_csv(file: FileStorage, type: str):
    binary = file.stream.read()
    try:
        csv_string = binary.decode('utf-8')
        csv_string = csv_string.replace('\r', '')
    except UnicodeDecodeError:
        return "Could not decode file content as UTF-8. It might be a binary file.", 500

    csv_io = io.StringIO(csv_string)
    df = pd.read_csv(csv_io)
    csv_io.close()

    return visualize_df(df, type)

def process_csv(file: FileStorage):
    binary = file.stream.read()
    try:
        csv_string = binary.decode('utf-8')
        csv_string = csv_string.replace('\r', '')
    except UnicodeDecodeError:
        return "Could not decode file content as UTF-8. It might be a binary file.", 500

    csv_io = io.StringIO(csv_string)
    df = pd.read_csv(csv_io)
    csv_io.close()

    return classify_df(df)


def compute_band_powers(df: pd.DataFrame):
    """
    Compute absolute and relative power for each frequency band.
    
    Frequency bands:
    - Delta: 0.5-4 Hz
    - Theta: 4-8 Hz
    - Alpha: 8-13 Hz
    - Beta: 13-30 Hz
    - Gamma: 30-60 Hz
    
    Returns:
    --------
    dict : Contains absolute and relative power for each band and electrode
    """
    from scipy.signal import welch
    
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 60)
    }
    
    electrodes = df.select_dtypes(include=[np.number]).columns.tolist()
    result = {
        'absolute_power': {},
        'relative_power': {},
        'total_power': {},
        'band_ratios': {}
    }
    
    for electrode in electrodes:
        signal = df[electrode].to_numpy()
        
        # Compute power spectral density using Welch's method
        freqs, psd = welch(signal, fs=SAMPLE_RATE, nperseg=min(256, len(signal)//4))
        
        # Compute absolute power for each band
        absolute_powers = {}
        for band_name, (low, high) in bands.items():
            # Find frequencies in this band
            band_mask = (freqs >= low) & (freqs < high)
            # Integrate power in this band using trapezoidal rule
            band_power = np.trapz(psd[band_mask], freqs[band_mask])
            absolute_powers[band_name] = float(band_power)
        
        # Compute total power across all bands
        total_power = sum(absolute_powers.values())
        
        # Compute relative power (as fraction of total)
        relative_powers = {
            band_name: (power / total_power) if total_power > 0 else 0.0
            for band_name, power in absolute_powers.items()
        }
        
        # Store results for this electrode
        result['absolute_power'][electrode] = absolute_powers
        result['relative_power'][electrode] = relative_powers
        result['total_power'][electrode] = float(total_power)
    
    # Compute average across all electrodes
    result['average_absolute_power'] = {
        band: np.mean([result['absolute_power'][elec][band] for elec in electrodes])
        for band in bands.keys()
    }
    
    result['average_relative_power'] = {
        band: np.mean([result['relative_power'][elec][band] for elec in electrodes])
        for band in bands.keys()
    }
    
    # Compute clinically relevant ratios
    avg_theta = result['average_absolute_power']['theta']
    avg_beta = result['average_absolute_power']['beta']
    avg_alpha = result['average_absolute_power']['alpha']
    
    result['band_ratios'] = {
        'theta_beta_ratio': float(avg_theta / avg_beta) if avg_beta > 0 else 0.0,
        'theta_alpha_ratio': float(avg_theta / avg_alpha) if avg_alpha > 0 else 0.0,
        'alpha_theta_ratio': float(avg_alpha / avg_theta) if avg_theta > 0 else 0.0
    }
    
    return result


def analyze_csv_bands(file: FileStorage):
    """Process CSV file and return band power analysis."""
    binary = file.stream.read()
    try:
        csv_string = binary.decode('utf-8')
        csv_string = csv_string.replace('\r', '')
    except UnicodeDecodeError:
        return {'error': 'Could not decode file content as UTF-8'}, 500

    csv_io = io.StringIO(csv_string)
    df = pd.read_csv(csv_io)
    csv_io.close()
    
    # Clean the data first
    print("📊 Preprocessing EEG data for band analysis...")
    
    # Compute band powers
    band_powers = compute_band_powers(df)
    
    return band_powers


@error_handle
def visualize_file(file: FileStorage, type="raw"):
    if file.filename is None:
        return None

    *_, ext = file.filename.split(".")
    print(f"VISUALIZE FILE, {ext}")

    match ext:
        case "csv":
            return visualize_csv(file, type)
        case _:
            return {'error': 'File extension not supported'}

    return None

def process_file(file: FileStorage):
    if file.filename is None:
        return None

    *_, ext = file.filename.split(".")

    match ext:
        case "csv":
            return process_csv(file)
        case _:
            return {'error': 'File extension not supported'}

    return None

@app.route('/')
def home():
    return render_template('index.html')  # Render the home page with upload form

@app.route('/visualize_eeg/<type>', methods=['POST'])
def visualize_eeg(type):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    match visualize_file(file, type):
        case None:
            return jsonify({'error': 'Something went wrong while reading the file.'}), 500
        case {'error': error}:
            print(error)
            return jsonify({'error': error}), 504
        case result:
            return jsonify({'result': result}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    match process_file(file):
        case None:
            return jsonify({'error': 'Something went wrong while reading the file.'}), 500
        case {'error': error}:
            return jsonify({'error': error}), 504
        case result:
            return jsonify({'prediction': True, 'result': result}), 200


@app.route('/analyze_bands', methods=['POST'])
def analyze_bands():
    """
    API endpoint to compute absolute and relative power for each frequency band.

    Returns:
    --------
    JSON with:
    - absolute_power: Power in each band for each electrode (μV²)
    - relative_power: Power as fraction of total (0-1) for each electrode
    - total_power: Total power across all bands for each electrode
    - average_absolute_power: Average across all electrodes for each band
    - average_relative_power: Average relative power across all electrodes
    - band_ratios: Clinically relevant ratios (theta/beta, etc.)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Please upload a CSV file.'}), 400

    try:
        result = analyze_csv_bands(file)

        if isinstance(result, tuple) and len(result) == 2:
            # Error case
            error_dict, status_code = result
            return jsonify(error_dict), status_code

        return jsonify(result), 200

    except Exception as e:
        print(f"Error in analyze_bands: {str(e)}")
        return jsonify({'error': f'Failed to analyze bands: {str(e)}'}), 500


if __name__ == '__main__':
    import os
    DEBUG = True
    load_scaler()
    load_model()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    app.run(debug=DEBUG, port=int(os.getenv('PORT', 5000)), threaded=True)