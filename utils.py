import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wave
import os
import webrtcvad
from librosa import stft
import math

def audio_record(CHANNELS=2, RATE=16000, _RECORD_SECONDS=3):
    audio1 = pyaudio.PyAudio()
    audio2 = pyaudio.PyAudio()

    CHUNK = int(RATE / 10)
    FORMAT = pyaudio.paInt16

    # start recording
    stream1 = audio1.open(format=pyaudio.paInt16,
                          channels=2,
                          rate=RATE,
                          input=True,
                          input_device_index=2,
                          frames_per_buffer=CHUNK)

    stream2 = audio2.open(format=pyaudio.paInt16,
                            channels=2,
                            rate=RATE,
                            input=True,
                            input_device_index=1,
                            frames_per_buffer=CHUNK)



    print("Recording")
    frames1 = []
    frames2 = []

    for i in range(0, int(RATE / CHUNK * _RECORD_SECONDS)):
        data1 = stream1.read(CHUNK)
        data2 = stream2.read(CHUNK)
        frames1.append(data1)
        frames2.append(data2)
    print("finish Recording")

    # stop recording
    stream1.stop_stream()
    stream1.close()
    audio1.terminate()
    stream2.stop_stream()
    stream2.close()
    audio2.terminate()

    waveFile = wave.open("E:\\Test/1.wav", 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio1.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames1))
    waveFile.close()
    waveFile = wave.open("E:\\Test/2.wav", 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio2.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames2))
    waveFile.close()
    frames1 = b''.join(frames1)
    frames2 = b''.join(frames2)

    return frames1, frames2

def audio_record2(CHANNELS=2, RATE=16000, _RECORD_SECONDS=3):
    audio1 = pyaudio.PyAudio()
    audio2 = pyaudio.PyAudio()

    CHUNK = 256#int(RATE / 100)
    FORMAT = pyaudio.paInt16

    # start recording
    stream1 = audio1.open(format=pyaudio.paInt16,
                          channels=2,
                          rate=RATE,
                          input=True,
                          input_device_index=2,
                          frames_per_buffer=CHUNK)

    stream2 = audio2.open(format=pyaudio.paInt16,
                            channels=2,
                            rate=RATE,
                            input=True,
                            input_device_index=1,
                            frames_per_buffer=CHUNK)



    #print("Recording")
    frames1 = []
    frames2 = []

    for i in range(0, int(RATE / CHUNK * _RECORD_SECONDS)):
        data1 = stream1.read(CHUNK)
        data2 = stream2.read(CHUNK)
        frames1.append(data1)
        frames2.append(data2)
    #print("finish Recording")

    # stop recording
    stream1.stop_stream()
    stream1.close()
    audio1.terminate()
    stream2.stop_stream()
    stream2.close()
    audio2.terminate()

    waveFile = wave.open("E:\\Test/1.wav", 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio1.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames1))
    waveFile.close()
    waveFile = wave.open("E:\\Test/2.wav", 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio2.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames2))
    waveFile.close()
    frames1 = b''.join(frames1)
    frames2 = b''.join(frames2)

    return frames1, frames2

class WebRTCVAD:
    def __init__(self, sample_rate=16000, level=3):
        """

        Args:
            sample_rate: audio sample rate
            level: between 0 and 3. 0 is the least aggressive about filtering out non-speech, 3 is the most aggressive.
        """
        self.sample_rate = sample_rate

        self.frame_ms = 30
        self.frame_bytes = int(2 * self.frame_ms * self.sample_rate / 1000)   # S16_LE, 2 bytes width

        self.vad = webrtcvad.Vad(level)
        self.active = False
        self.data = b''
        self.history = list()

    def is_speech(self, data):
        self.data += data
        while len(self.data) >= self.frame_bytes:
            frame = self.data[:self.frame_bytes]
            self.data = self.data[self.frame_bytes:]
            if self.vad.is_speech(frame, self.sample_rate):
                self.history.append(1)
            else:
                #sys.stdout.write('0')
                self.history.append(0)
            result = np.array(self.history)
        self.history = list()
        return result

    def reset(self):
        self.data = b''
        self.active = False
        self.history.clear()


def audio_list():
    po = pyaudio.PyAudio()

    for index in range(po.get_device_count()):
        desc = po.get_device_info_by_index(index)
        print("DEVICE:%s INDEX: %s RATE: %s" % (desc["name"], index, int(desc["defaultSampleRate"])))

def plot_vad(raw_array, vad_result, filtered, flag='vad_result') :
    plt.subplot(8, 1, 1)
    plt.plot(raw_array[:, 0])
    plt.subplot(8, 1, 3)
    plt.plot(raw_array[:, 1])
    plt.subplot(8, 1, 5)
    plt.plot(raw_array[:, 2])
    plt.subplot(8, 1, 7)
    plt.plot(raw_array[:, 3])

    if flag == 'vad_result' :
        plt.subplot(8,1,2)
        plt.plot(vad_result[:,0])
        plt.subplot(8,1,4)
        plt.plot(vad_result[:,1])
        plt.subplot(8,1,6)
        plt.plot(vad_result[:,2])
        plt.subplot(8,1,8)
        plt.plot(vad_result[:,3])
    elif flag == 'filtered' :
        plt.subplot(8, 1, 2)
        plt.plot(filtered[:, 0])
        plt.subplot(8, 1, 4)
        plt.plot(filtered[:, 1])
        plt.subplot(8, 1, 6)
        plt.plot(filtered[:, 2])
        plt.subplot(8, 1, 8)
        plt.plot(filtered[:, 3])
    plt.show()

def polar2cart(r, theta) :
    x = r * math.cos(theta)
    y = r * math.sin(theta)

    return x, y

def cart2polar(x, y) :
    r = math.sqrt(pow(x,2) + pow(y,2))
    theta = math.atan2(y, x)

    return r, theta

def rad2deg(rad) :
    return rad * 180 / math.pi

def deg2rad(deg) :
    return deg * math.pi / 180