import numpy as np
import pandas as pd

from pathlib import Path

SAMPLE_RATE = 128             # Hz
WINDOW_SECONDS = 4            # window duration
OVERLAP = 0.5                 # 50% overlap
SAMPLES_PER_WINDOW = int(SAMPLE_RATE * WINDOW_SECONDS)
WINDOW_STEP = int(SAMPLES_PER_WINDOW * (1 - OVERLAP))    # step size for sliding window
TARGET_FREQUENCY_BINS = np.arange(2.0, 40.5, 0.5)        # target frequency bins

## First, we load the "recordings"
df = pd.read_csv("./adhdata.csv")
export_path = Path("extracted_eegs")
export_path.mkdir(parents=True, exist_ok=True)

for i in range(df['ID'].nunique()):
    id = df['ID'].unique()[i]

    target_recording = df[df['ID'] == id]
    target_recording = target_recording.drop(['ID', 'Class'], axis=1)
    target_recording.to_csv(export_path.joinpath(f"{id}.csv"), index=False)