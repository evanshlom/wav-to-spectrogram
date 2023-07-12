import librosa as librosa
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import librosa.display
import os

# Got the librosa original code from this comparison between librosa and nvidia-dali:
#https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/audio_processing/spectrogram.html#Calculating-the-Spectrogram-using-DALI

#test_data_root = os.environ['DALI_EXTRA_PATH']
#sample_data = os.path.join(test_data_root, 'db', 'audio', 'wav', '237-134500-0000.wav')

# Size of the FFT, which will also be used as the window length
n_fft=2048

# Step or stride between windows. If the step is smaller than the window lenght, the windows will overlap
hop_length=512

# Load sample audio file
sample_data = 'wavs\Literally.wav'
y, sr = librosa.load(sample_data)

# Calculate the spectrogram as the square of the complex magnitude of the STFT
spectrogram_librosa = np.abs(librosa.stft(
    y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hann', pad_mode='reflect')) ** 2


spectrogram_librosa_db = librosa.power_to_db(spectrogram_librosa, ref=np.max)


def show_spectrogram(spec, title, sr, hop_length, y_axis='log', x_axis='time'):
    librosa.display.specshow(spec, sr=sr, y_axis=y_axis, x_axis=x_axis, hop_length=hop_length)
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig('output/spectrogram-literally.png')
    #plt.show()

show_spectrogram(spectrogram_librosa_db, 'Reference power spectrogram', sr, hop_length)