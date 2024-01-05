import math
import os
import random
from librosa import load
import librosa
import numpy as np
import soundfile as sf
import time
from tqdm import tqdm

flac_files = [
    f
    for f in os.listdir("C:/Users/Benja/OneDrive/Eksamensprojekt/DataRaw")
    if f.endswith(".flac")
]

print(
    f"Staring spectrogram calculation and wav convertion of {str(len(flac_files))} files..."
)

random.seed(123)
duration_in_seconds = 10


def save_snippet(file, audio, start_in_seconds, sr):
    audio_file_path = f"C:/Users/Benja/OneDrive/Eksamensprojekt/DataWav/{file.split('.')[0]}-{start_in_seconds * 1000}.wav"

    # Extract the first 10 seconds of the audio
    # librosa.load loads audio as a floating point time series, so duration needs to be specified in seconds
    cutted_audio = audio[
        start_in_seconds * sr : (start_in_seconds + duration_in_seconds) * sr
    ]

    # Save the first 10 seconds
    sf.write(audio_file_path, cutted_audio, sr, format="wav")

    # Normalize the audio data
    sound = cutted_audio / np.max(np.abs(cutted_audio))

    if sound.shape != (441000,):
        # write this missing file to a text file
        with open(
            "C:/Users/Benja/OneDrive/Eksamensprojekt/DataLogs/errors.csv", "a"
        ) as f:
            f.write(f"{file.split('.')[0]}-{start_in_seconds*1000};{sound.shape[0]}\n")

    # make the same plot with librosa and compare them in one figure
    # _, ax = plt.subplots(2, 2, figsize=(30, 20))
    # plt.axis("off")

    plotFunc(cutted_audio, 512, 256, file, start_in_seconds)
    plotFunc(cutted_audio, 1024, 256, file, start_in_seconds)
    plotFunc(cutted_audio, 8192, 256, file, start_in_seconds)

    # file_path = f"C:/Users/Benja/OneDrive/Eksamensprojekt/DataSpectrogram-All/{file.split('.')[0]}-{start_in_seconds * 1000}"
    # plt.tight_layout()
    # plt.savefig(
    #     f"{file_path}.png",
    #     dpi=200,
    #     format="png",
    # )
    # plt.clf()


def plotFunc(cutted_audio, n_fft, n_hop, file, start_in_seconds):
    # time1 = time.time()
    stft_plot1 = abs(
        librosa.stft(
            cutted_audio,
            n_fft=n_fft,
            hop_length=n_hop,
        )
    )
    # time2 = time.time()
    # librosa.display.specshow(
    #     librosa.amplitude_to_db(stft_plot1, ref=np.max),
    #     y_axis="log",
    #     x_axis="time",
    #     ax=ax,
    # )
    spectrogram = librosa.amplitude_to_db(stft_plot1, ref=np.max)
    # print(spectrogram.shape)
    # save the spectrogram as a numpy array
    file_path = f"C:/Users/Benja/OneDrive/Eksamensprojekt/DataSpectrogram-{n_fft}/{file.split('.')[0]}-{start_in_seconds * 1000}"
    np.save(
        file_path + f"-{n_fft}-{n_hop}",
        spectrogram,
    )
    # ax.set_title(f"{n_fft} / {n_hop} [{int((time2 - time1)*1000)} ms]")


time1 = time.time()
for i in tqdm(range(len(flac_files)), desc="Extracting and converting snippets..."):
    file_name = flac_files[i]
    # Load the FLAC file and slice the first 10 seconds (pydub works in milliseconds)
    audio, sr = load(
        f"C:/Users/Benja/OneDrive/Eksamensprojekt/DataRaw/{file_name}", sr=44100
    )

    save_snippet(file_name, audio, 0, sr)
    prev = 0

    # also export 2 more random snippets from the same file
    for j in range(2):
        start_in_seconds = random.randint(10, math.floor(len(audio) / sr) - 10)

        # make sure the snippet is not too close to the previous one
        while abs(start_in_seconds - prev) < 11:
            start_in_seconds = random.randint(10, math.floor(len(audio) / sr) - 10)

        save_snippet(file_name, audio, start_in_seconds, sr)
        prev = start_in_seconds

time2 = time.time()
print(f"Time elapsed: {time2 - time1}")
