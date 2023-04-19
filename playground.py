import torch
import torchaudio


def part1():
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa.feature

    # file loading
    y, sr = librosa.load("audio_16k/bird-sing.wav")
    n = len(y)

    WINDOW_SIZE = 512

    # STFT
    Y = librosa.stft(y=y, n_fft=WINDOW_SIZE) # coeff_per_freq, n_windows
    fig, ax = plt.subplots()
    img = librosa.display.specshow(np.abs(Y), x_axis='time', y_axis='linear', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f')
    ax.set(title='Spectrogram')

    # Mel
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=WINDOW_SIZE, hop_length=WINDOW_SIZE//4, n_mels=80)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S_mel, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                             y_axis='mel', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


def part2():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft

    def sine_by_freq(t, freq):
        return np.sin(t * freq * 2*np.pi)

    def DFT(x):
        """Compute the discrete Fourier Transform of the 1D array x"""
        x = np.asarray(x, dtype=float)
        N = x.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, x)

    # t = 3
    N_SECS = 3

    sr = 16000

    fs = 1 / float(sr)
    t = np.linspace(0, N_SECS, N_SECS * sr)

    FREQ1 = 500
    sin1 = sine_by_freq(t, FREQ1)

    FREQ2 = 4000
    sin2 = sine_by_freq(t, FREQ2)

    N_SAMPLES_TO_PLOT = 200
    t_start = t[:N_SAMPLES_TO_PLOT]
    plt.plot(t_start, sin1[:N_SAMPLES_TO_PLOT], label="500 Hz", alpha=0.7)
    plt.plot(t_start, sin2[:N_SAMPLES_TO_PLOT], label="4K Hz", alpha=0.7)
    plt.plot(t_start, sin1[:N_SAMPLES_TO_PLOT] + sin2[:N_SAMPLES_TO_PLOT], label="sum", alpha=0.7)
    plt.xlabel("time (seconds)")
    plt.legend(loc="best")


    fft1 = np.fft.fft(sin1)
    fft2 = np.fft.fft(sin2)
    fft3 = np.fft.fft(sin2+sin1)
    freqs = np.fft.fftfreq(len(sin1))

    fig, ax = plt.subplots(3, 1, sharex=True)

    len_to_cut = int(len(sin1)/2)

    ax[0].plot(freqs[len_to_cut:], abs(fft1)[len_to_cut:])
    ax[0].set_ylabel('|FT(sin1)|')

    ax[1].plot(freqs[len_to_cut:], abs(fft2)[len_to_cut:])
    ax[1].set_ylabel('|FT(sin2)|')

    ax[2].plot(freqs[len_to_cut:], abs(fft3)[len_to_cut:])
    ax[2].set_xlabel('Frequency')
    ax[2].set_ylabel('|FT(sin2 + sin3)|')
    plt.show()

    # dft1 = DFT(sin1)
    # plt.figure()
    # plt.plot(np.arange(0, N_SECS * fs) / 10., abs(dft1))


    plt.show()

def part3():
    """
    Part C: playing with sampling rate.

    In this part we would observe the influences of sampling frequency by using a toy setup where we control the sampling frequency of a given cascaded sine wave.

    1. Generate a sine wave corresponding to `sum_{i=0 to 4}(sine(2^{i} * 1000 [Hz]))` using a 32[KHz] frequency, with duration of 3 secs.
    2. Define a function `resample(signal, new_fs, old_fs)` that downsamples the given signal from old_fs frequency to new_fs frequency.
       Use `torchaudio.transforms.Resample` for this purpose.
    3. Downsamle the generated signal to 16[KHz] and to 8[KHz] by by 'skipping' samples, i.e. downsampling 32Khz to 16KHz is simply taking every other sample..
    4. Use `torchaudio.transforms.Resample` to artificially resume 32[KHz] frequency for the downsampled signals
       (this is to ensure that your fft plots would contain spectral frequencies ranging from 0-16k, why is that?).
    5. Observe the FFT plots of the generated signals, do you see any phenomena occuring? describe the phenomena.
       why is this happening? explain. Hint: what happens at 2[KHz], 4[KHz]? is it different for other frequencies?

    Note: your FFT plots should plot all frequencies ranging 0 - 16000
    :return:
    """
    import numpy as np
    import matplotlib.pyplot as plt

    def sine_by_i(duration, i):
        return np.sin(duration * 2 * np.pi* 2**i * 1000)

    def resample(signal, new_fs, old_fs):
        signal_tensor = torch.from_numpy(signal).unsqueeze(0)
        resample_transform = torchaudio.transforms.Resample(old_fs, new_fs)
        resample_transform.type('torch.DoubleTensor')
        downsampled_signal_tensor = resample_transform(signal_tensor)
        downsampled_signal = downsampled_signal_tensor.numpy()[0]
        return downsampled_signal

    seconds = 3
    sr = 32000
    sr16 = 16000
    sr8 = 8000
    N_SAMPLES_TO_PLOT = 200
    t = np.linspace(0, seconds, seconds * sr)

    sin_arrays = list()
    for i in range(5):
        sin_arrays.append(sine_by_i(t, i))
    sin_sum = np.sum(sin_arrays, axis=0)


    unsampled16 = resample(sin_sum, sr16, sr)
    time16 = np.linspace(0, seconds, seconds * sr16)

    unsampled8 = resample(sin_sum, sr8, sr)
    time8 = np.linspace(0, seconds, seconds * sr8)

    fig, ax = plt.subplots(3, 1)

    ax[0].plot(t[:N_SAMPLES_TO_PLOT], sin_sum[:N_SAMPLES_TO_PLOT])
    ax[0].set_title("down sampling")
    ax[0].set_ylabel('32 Khz')

    ax[1].plot(time16[:N_SAMPLES_TO_PLOT//2], unsampled16[:N_SAMPLES_TO_PLOT//2])
    ax[1].set_ylabel('16 Khz')

    ax[2].plot(time8[:N_SAMPLES_TO_PLOT//4], unsampled8[:N_SAMPLES_TO_PLOT//4])
    ax[2].set_xlabel('time')
    ax[2].set_ylabel('8 Khz')

    plt.show()


    upsampled16 = resample(unsampled16, sr, sr16)
    upsampled8 = resample(unsampled8, sr, sr8)

    fig, ax = plt.subplots(3, 1)
    ax[0].set_title("up sampling")

    ax[0].plot(t[:N_SAMPLES_TO_PLOT], sin_sum[:N_SAMPLES_TO_PLOT])
    ax[0].set_ylabel('32 Khz')

    ax[1].plot(t[:N_SAMPLES_TO_PLOT], upsampled16[:N_SAMPLES_TO_PLOT])
    ax[1].set_ylabel('32 Khz')

    ax[2].plot(t[:N_SAMPLES_TO_PLOT], upsampled8[:N_SAMPLES_TO_PLOT])
    ax[2].set_xlabel('time')
    ax[2].set_ylabel('32 Khz')

    plt.show()

    fft1 = np.fft.fft(sin_sum)
    fft2 = np.fft.fft(upsampled16)
    fft3 = np.fft.fft(upsampled8)
    freqs = np.fft.fftfreq(len(fft1))

    fig, ax = plt.subplots(3, 1, sharex=True)

    ax[0].set_title("Fourier of up sampled waveforms")

    len_to_cut = int(len(fft1) / 2)

    ax[0].plot(freqs[:len_to_cut], abs(fft1)[:len_to_cut])
    ax[0].set_ylabel('original')

    ax[1].plot(freqs[:len_to_cut], abs(fft2)[:len_to_cut])
    ax[1].set_ylabel('16 Khz')

    ax[2].plot(freqs[:len_to_cut], abs(fft3)[:len_to_cut])
    ax[2].set_xlabel('Frequency (Hz)')
    ax[2].set_ylabel('8 khz')
    plt.show()

    """
        We see the implications of Nyquist theorem.
        only frequencies which are smaller then 0.5 * sr could be restored.
        hence the 2Khz and 4Khz are seen in every sample, because the smallest sampling rate is 8Khz
        but the bigger frequencies disappear.
    """




if __name__ == "__main__":
    part3()
