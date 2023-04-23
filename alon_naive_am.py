
def partA():
    """
    Part A: Interpolating over time.

    1. load 'audio_16k/Basta_16k.wav' audio file (note that it is on stereo)
    2. use `torch.nn.functional.interpolate` with `mode='bilinear` to stretch / compress the signal with 1.2, 0.8 factor respectfully.
    3. save these samples to outputs directory as 'interpolation_0_8.wav', 'interpolation_1_2.wav' and listen to them, do you notice something odd? why do you think this happens? - answear in a markdown cell below.
    """

    # import librosa
    import torch.nn.functional as F
    import torchaudio
    import os

    def create_if_not_exists(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    # create outputs dir
    OUTPUTS_DIR = "outputs"
    create_if_not_exists(OUTPUTS_DIR)

    # 1. load 'audio_16k/Basta_16k.wav' audio file (note that it is on stereo)
    y, sr = torchaudio.load("audio_16k/Basta_16k.wav")

    SCALES = [0.8, 1.2]
    for stretch_scale in SCALES:
        # F.interpolate expects 4D signals in bilinear mode
        stretched_y = F.interpolate(y[None, None, :], mode='bilinear', scale_factor=stretch_scale)[0, 0, :]
        outfile = f"{OUTPUTS_DIR}/interpolation_{str(stretch_scale).replace('.', '_')}.wav"
        print(f"saving outfile: {outfile}")
        torchaudio.save(outfile, stretched_y, sr)


def partB():
    """
    Part B: Naive time stretch (tempo shift).
    In this part you would be required to write a function that perform a SIMPLE augmentation over the audio:
    1. `naive_tempo_shift(wav, factor)` = stretch an audiofile by a given factor, e.g 0.8 factor should result a slowdown to 0.8x the original audio (output a LONGER wav).
    2. load 'audio_16k/Basta_16k.wav' and generate a tempo shift of x{0.8, 1.2} and save these generated audio files to outputs/naive_pitch_shift_{factor using _ instead if .}.wav

    Note: This should be a Naive implementation, achieveable using torch.stft, torch.istft, torch.fft.fft, torch.fft.ifft alone and programable in a few lines.
    """

if __name__ == "__main__":
    partA()
    # partB()
    pass