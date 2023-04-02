from tkinter.filedialog import askopenfilename, asksaveasfilename

import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile


class Audio:
    """
    A class representing an audio file.

    Attributes:
        rate (int): The sampling rate of the audio file, in Hz.
        data (np.ndarray): The audio data, as an array of samples.
    """

    rate: int
    data: np.ndarray

    def __init__(self) -> None:
        """Constructs a new Audio object and loads an audio file."""
        self.load()

    def load(self) -> None:
        """
        Prompts the user to select an audio file, reads the file,
        and initializes the Audio object.

        Supported audio formats: WAV (.wav), MP3 (.mp3)
        """
        path = askopenfilename(
            initialdir="./SoundData",
            filetypes=[("Audio files", ".mp3 .wav")],
        )
        # Read audio file
        if path.endswith(".wav"):
            rate, data = wavfile.read(path)
        elif path.endswith(".mp3"):
            audio = AudioSegment.from_file(path, format="mp3")
            data = np.array(audio.get_array_of_samples())
            rate = audio.frame_rate
        else:
            print("Unsupported file format")
            return
        # convert to mono
        self.data = data if len(data.shape) == 1 else data[:, 0]
        self.rate = rate

    def save(self, data: np.ndarray) -> None:
        """
        Saves audio data to a WAV file.

        Args:
            data (np.ndarray): The audio data to save
        """
        path = asksaveasfilename(
            defaultextension=".wav",
            initialfile="fft_out.wav",
            filetypes=[("WAV files", ".wav")],
        )
        wavfile.write(path, self.rate, data)

    def split_aud(self, stepsize: int) -> list[np.ndarray]:
        """Splits audio into parts of length stepsize

        Args:
            step_size(int): step size in ms

        Returns:
            list[np.ndarray]: Audio slices
        """
        # Calculate amount of samples in step size interval
        step_size_samples = int(self.rate * stepsize / 1000)

        # Split audio data into sub-arrays of step size interval
        audio_slices = [
            self.data[i : i + step_size_samples]
            for i in range(0, len(self.data), step_size_samples)
        ]

        return audio_slices
