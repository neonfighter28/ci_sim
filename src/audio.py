from tkinter.filedialog import askopenfilename
from scipy.io import wavfile
from pydub import AudioSegment

class Audio:
    def __init__(self, *, path=askopenfilename(initialdir="./SoundData")):
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

    def getMono(self):
        if len(self.data.shape) == 1:
            return self.data
        else:
            return self.data[:, 0]

    def save(self, path, data):
        wavfile.write(path, self.rate, data)
