from tkinter.filedialog import askopenfilename

import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile


class Audio:
    rate: int
    data: np.ndarray

    def __init__(self,
                 *,
                 path=askopenfilename(initialdir="./SoundData")
                 ) -> None:
        """Reads audio file and initializes class

        Args:
            path (str, optional): /path/to/audio/file.
            Defaults to askopenfilename(initialdir="./SoundData").
        """
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

    def get_mono(self) -> np.ndarray:
        """Returns one audio channel, doesn't alter self.data

        Returns:
            np.ndarray: array of audio data
        """
        if len(self.data.shape) == 1:
            return self.data
        else:
            return self.data[:, 0]

    def save(self, path, data) -> None:
        """Saves audio to file

        Args:
            path (str): /path/to/output
            data (np.ndarray): array of audio data
        """
        wavfile.write(path, self.rate, data)
