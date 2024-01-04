import math
import os
import random
from librosa import load
import librosa
from matplotlib import pyplot as plt
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.signal import stft
import soundfile as sf
import time

flac_files = [
    f
    for f in os.listdir("C:/Users/Benja/OneDrive/Eksamensprojekt/FixData")
    if f.endswith(".flac")
]

print(len(flac_files))

random.seed(123)
duration_in_seconds = 10


def save_snippet(file, audio, start_in_seconds, sr):
    audio_file_path = f"C:/Users/Benja/OneDrive/Eksamensprojekt/SnippetDataWav/{file.split('.')[0]}-{start_in_seconds}.wav"

    # Extract the first 10 seconds of the audio
    # librosa.load loads audio as a floating point time series, so duration needs to be specified in seconds
    cutted_audio = audio[
        start_in_seconds * sr : (start_in_seconds + duration_in_seconds) * sr
    ]

    # Save the first 10 seconds
    sf.write(audio_file_path, cutted_audio, sr)

    # Normalize the audio data
    sound = cutted_audio / np.max(np.abs(cutted_audio))

    print("Sound shape:", sound.shape)

    if sound.shape != (441000,):
        print("wrong shape in sound samples")
        print(sound.shape)

    # make the same plot with librosa and compare them in one figure
    fig, ax = plt.subplots(7, 2, figsize=(160 // 7 * 2, 90))
    plt.axis("off")
    n_fft = 256
    n_hop = 64

    # plotFunc(cutted_audio, ax[1, 1], n_fft, n_hop)
    # generate a grid where differnt n_fft and n_hop are plotted

    for i in range(7):
        for j in range(2):
            print(i, j)
            plotFunc(cutted_audio, ax[i, j], n_fft * 2 ** (i), n_hop * 2 ** (j))

    plt.tight_layout()
    plt.savefig(
        f"C:/Users/Benja/OneDrive/Eksamensprojekt/SnippetDataSpectrogram/{file.split('.')[0]}-{start_in_seconds}_3.png",
        dpi=100,
        format="png",
    )
    plt.clf()


def plotFunc(cutted_audio, ax, n_fft, n_hop):
    time1 = time.time()
    stft_plot1 = abs(
        librosa.stft(
            cutted_audio,
            n_fft=n_fft,
            hop_length=n_hop,
        )
    )
    time2 = time.time()
    librosa.display.specshow(
        librosa.amplitude_to_db(stft_plot1, ref=np.max),
        y_axis="log",
        x_axis="time",
        ax=ax,
    )
    ax.set_title(f"{n_fft} / {n_hop} [{int((time2 - time1)*1000)} ms]")


for i in range(1):
    file_name = flac_files[i]
    # Load the FLAC file and slice the first 10 seconds (pydub works in milliseconds)
    audio, sr = load(
        f"C:/Users/Benja/OneDrive/Eksamensprojekt/FixData/{file_name}", sr=44100
    )

    save_snippet(file_name, audio, 0, sr)
    prev = 0

    # also export 2 more random snippets from the same file
    for j in range(2):
        start_in_seconds = random.randint(10, math.floor((len(audio) - 10) / sr))

        # make sure the snippet is not too close to the previous one
        while abs(start_in_seconds - prev) < 10:
            start_in_seconds = random.randint(10, math.floor((len(audio) - 10) / sr))

        save_snippet(file_name, audio, start_in_seconds, sr)
        prev = start_in_seconds

    print(f"{i + 1}/{len(flac_files)} files done")
