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
            filetypes=[("WAV files", ".wav"), ("MP3 files", ".mp3")],
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
        self.data = data
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
            filetypes=[("WAV files", ".wav")]
            )
        wavfile.write(path, self.rate, data)

    @property
    def mono(self) -> np.ndarray:
        """
        Returns a mono audio channel, without altering the original audio data.

        If the audio data is already mono, return the original data.
        Otherwise, return only the first audio channel.

        Returns:
            np.ndarray: An array of audio data,
                        representing a single audio channel.
        """
        if len(self.data.shape) == 1:
            return self.data
        else:
            return self.data[:, 0]
