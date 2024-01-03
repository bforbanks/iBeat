import numpy as np
import librosa
import matplotlib.patches as mpatches
import os

flac_files = [
    f
    for f in os.listdir("C:/Users/Benja/OneDrive/Eksamensprojekt/FixData")
    if f.endswith(".flac")
]

print(f"Staring beat detection in {str(len(flac_files))} files...")
print("This might take a while...")
print("Output will be saved to beats.csv")

for i in range(10):
    file = flac_files[i]
    y, sr = librosa.load(f"C:/Users/Benja/OneDrive/Eksamensprojekt/FixData/{file}")

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    new_beats = librosa.frames_to_time(beats, sr=sr)
    # append beats to csv file
    with open("beats.csv", "a") as f:
        f.write(
            f"{file.split('-')[1].split('.')[0]};"
            + ";".join((np.array(new_beats) * 1000).astype(int).astype(str))
            + "\n"
        )
    print(f"{i + 1}/{len(flac_files)} files done")
