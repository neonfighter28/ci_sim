# Simple Simulation of a cochlea implant using FFT
# Authors: ecsuka <ecsuka@ethz.ch> and others
# v1.0.0 (21.03.23)
# Run with `$ python main.py`
# Developed and tested with Python 3.10.6
# (tags/v3.10.6:9c7b4bd, Aug  1 2022, 21:53:49)

####### VOLUME WARNING #######
# Because of STFT transformation artifacts the audio may be unpleasant to
# listen to!
####### VOLUME WARNING #######

import numpy as np
from scipy import signal

from audio import Audio


def cochlear_implant_simulation(slice: np.ndarray, rate: int):
    """
    Performs the CI Simulation on the input audio and returns the output audio

    The strategy used simulates a cochlear implant by compressing the audio
    into a number of frequency bands, equal to the number of electrodes, spaced
    logarithmically, as to mimic the human ear, then applying the n-out-of-m
    strategy to select the most dominant frequency bands. At last, the audio is
    normalized and filtered using a low-pass filter to simulate the ear canal.

    This consists of the following steps:
    1. Convert the input to mono and compute its FFT
    2. Generate the frequency bands
    3. Compute the total power for every frequency band
    4. Apply the n-out-of m strategy, whereby we select the most dominant bands
       Note that this can be disabled by changing the parameter use_n_out_of_m
    5. Remove the bands that arent in the top n_out_of_m
    6. Compute the inverse FFT to get the processed audio signal
    7. Apply the low-pass filter
    8. Return audio

    Args:
        slice (np.ndarray): Audio data for slice
        rate (int): audio rate

    Returns:
        np.ndarray: Processed audio data
    """

    # Set parameters for the cochlear implant simulation
    use_n_out_of_m = True
    f_min = 200  # Hz
    f_max = 5000  # Hz
    num_electrodes = 20
    step_size = 20  # ms
    n_out_of_m = 12 if use_n_out_of_m else num_electrodes

    audio_data = slice

    # Array of logarithmically spaced frequency bands between f_min and f_max
    freq_bands = np.logspace(np.log10(f_min), np.log10(f_max), num=num_electrodes + 1)

    # Compute the FFT of the input audio signal
    fft_data = np.fft.fft(audio_data)

    # Get the frequency axis
    freqs = np.fft.fftfreq(audio_data.size, 1 / rate)

    # Process the FFT data by selecting the top n_out_of_m frequency bands
    # Allocate memory
    fft_data_processed = np.zeros_like(fft_data)
    band_powers = np.zeros(num_electrodes)

    # This "compresses" the audio between the frequency bands depending
    # on the number of electrodes
    for i in range(num_electrodes):
        f_low = freq_bands[i]
        f_high = freq_bands[i + 1]

        # Get the indices corresponding to the current frequency band
        band_indices = np.where((freqs >= f_low) & (freqs < f_high))

        # Compute the total power for the current frequency band
        band_powers[i] = np.sum(np.abs(fft_data[band_indices]) ** 2)

    # Apply the n-out-of-m strategy

    # Get the highest powered frequency bands, by selecting the
    # last n-out-of-m bands in a sorted array
    top_n_indices = np.argsort(band_powers)[-n_out_of_m:]
    for i in range(num_electrodes):
        # Remove the bands data if its not in the top n-out-of-m
        if i not in top_n_indices:
            f_low = freq_bands[i]
            f_high = freq_bands[i + 1]
            band_indices = np.where((freqs >= f_low) & (freqs < f_high))
            fft_data_processed[band_indices] = 0
        # Otherwise copy the FFT data
        else:
            f_low = freq_bands[i]
            f_high = freq_bands[i + 1]
            band_indices = np.where((freqs >= f_low) & (freqs < f_high))
            fft_data_processed[band_indices] = fft_data[band_indices]

    # Compute the inverse FFT to obtain the processed audio signal
    audio_processed = np.fft.ifft(fft_data_processed).real

    # Normalize the audio to prevent it from being too loud
    # By dividing it through its max abs value
    # audio_processed = audio_processed / np.max(np.abs(audio_processed))

    # Simple low-pass filter simulating the ear canal
    # Removes a bit of the "underwater" effect, but isn't perfect
    lowpass_cutoff = 1000  # Hz
    sos = signal.butter(2, lowpass_cutoff, btype="low", fs=rate, output="sos")
    audio_processed = signal.sosfilt(sos, audio_processed)

    # Save the processed audio as a WAV file
    return audio_processed


if __name__ == "__main__":
    # Increase step size to lower impact of STFT artifacts
    step_size = 20  # ms
    audio = Audio()

    # split audio
    slices = audio.split_aud(step_size)
    # simulate
    audio_processed = np.concatenate(
        [cochlear_implant_simulation(slice, audio.rate) for slice in slices]
    )

    audio.save(audio_processed / np.max(np.abs(audio_processed)))
