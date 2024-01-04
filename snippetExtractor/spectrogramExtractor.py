import os
import random
from matplotlib import pyplot as plt
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.signal import stft

files = [
    f
    for f in os.listdir("C:/Users/Benja/OneDrive/Eksamensprojekt/SnippetDataWav")
    if f.endswith(".wav")
]

print(len(files))

random.seed(123)


def save_snippet(file, audio, start):
    samples = np.array(audio.get_array_of_samples())
    sound = samples.astype(float) / np.max(np.abs(samples))
    print(sound.shape)
    if sound.shape != (882000,):
        print("wrong shape in sound samples")
        print(sound.shape)

    frequency, time, Z = stft(
        sound,
    )
    if Z.shape != (129, 6892):
        print("wrong shape in stft")
        print(Z.shape)

    plt.pcolormesh(
        time,
        frequency,
        np.abs(Z),
        vmin=0,
        vmax=1,
    )
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [ms]")
    plt.title(i)

    # plt.show()

    plt.savefig(
        f"C:/Users/Benja/OneDrive/Eksamensprojekt/SnippetDataSpectrogram/{file.split('.')[0]}-{start}.png",
        dpi=600,
    )
    plt.clf()


for i in range(len(files)):
    file_name = files[i]
    # Load the FLAC file and slice the first 10 seconds (pydub works in milliseconds)
    audio = AudioSegment.from_file(
        f"C:/Users/Benja/OneDrive/Eksamensprojekt/SnippetDataWav/{file_name}"
    )

    # resample the audio to 44.1 kHz
    audio = audio.set_frame_rate(44100)

    save_snippet(file_name, audio, 0)

    print(f"{i + 1}/{len(files)} files done")
