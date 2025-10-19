from flask import Flask, render_template, request, jsonify
from werkzeug.datastructures import FileStorage
from scaler import EEGScaler
from model import EEG_CNN_LSTM_HPO
from typing import TypedDict
from error_handle import error_handle

import torch

import io
import mne
import pickle
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('Agg')  # Non-GUI backend

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

def preview_df(df: pd.DataFrame):
    ch_names = df.select_dtypes(include=[np.number]).columns.tolist()
    ch_types = ['eeg'] * len(ch_names)
    filtered_data = df.to_numpy().T
    # Visualize the cleaned EEG data.
    info = mne.create_info(ch_names=ch_names, sfreq=SAMPLE_RATE, ch_types=ch_types)
    filtered = mne.io.RawArray(filtered_data, info)

    buf = io.BytesIO()

    fig = filtered.plot(scalings='auto',
                        n_channels=len(ch_names),
                        show=False,
                        show_scrollbars=False,
                        show_options=False,
                        clipping=None,
                        block=False,
                        )
    fig.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return f"data:image/png;base64,{img_base64}"

class WindowOutput(TypedDict):
    window: int
    frequency: np.floating
    electrodes: list[np.ndarray]

def process_window(window: pd.DataFrame, window_count: int):
    output: list[WindowOutput] = []
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


def preview_csv(file: FileStorage):
    binary = file.stream.read()
    try:
        csv_string = binary.decode('utf-8')
        csv_string = csv_string.replace('\r', '')
    except UnicodeDecodeError:
        return "Could not decode file content as UTF-8. It might be a binary file.", 500

    csv_io = io.StringIO(csv_string)
    df = pd.read_csv(csv_io)
    csv_io.close()

    return preview_df(df)

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


@error_handle
def preview_file(file: FileStorage):
    if file.filename is None:
        return None

    *_, ext = file.filename.split(".")

    match ext:
        case "csv":
            return preview_csv(file)
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

@app.route('/preview', methods=['POST'])
def preview():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    match preview_file(file):
        case None:
            return jsonify({'error': 'Something went wrong while reading the file.'}), 500
        case {'error': error}:
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

if __name__ == '__main__':
    import os
    DEBUG = True
    load_scaler()
    load_model()
    app.run(debug=DEBUG, port=int(os.getenv('PORT', 5000)))