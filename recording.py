from librosa import stft
from utils import *
import sys
import signal
from socket import *

N = 1024
vad = WebRTCVAD()

savepath = "E:\\Records3/"
if not os.path.isdir(savepath) :
    os.makedirs(savepath)
ang = "0"
def audio_recording(CHANNELS=2, RATE=16000, _RECORD_SECONDS=60, step=1):
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

    frames1 = b''.join(frames1)
    frames2 = b''.join(frames2)
    c1, c2 = stereo2mono(frames1)
    c3, c4 = stereo2mono(frames2)

    waveFile = wave.open(savepath + "step_" +str(step) +"_"+ang + "_1.wav", 'wb')
    waveFile.setnchannels(1)
    waveFile.setsampwidth(audio1.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(c1)
    waveFile.close()
    waveFile = wave.open(savepath + "step_" +str(step) +"_"+ang + "_2.wav", 'wb')
    waveFile.setnchannels(1)
    waveFile.setsampwidth(audio1.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(c2)
    waveFile.close()

    waveFile = wave.open(savepath + "step_" +str(step) +"_"+ang + "_3.wav", 'wb')
    waveFile.setnchannels(1)
    waveFile.setsampwidth(audio2.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(c3)
    waveFile.close()
    waveFile = wave.open(savepath + "step_" +str(step) +"_"+ang + "_4.wav", 'wb')
    waveFile.setnchannels(1)
    waveFile.setsampwidth(audio2.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(c4)
    waveFile.close()

    return frames1, frames2

def stereo2mono(stereo) :
    buf = np.fromstring(stereo, dtype='int16')
    frame1 = buf[0::2]
    frame2 = buf[1::2]
    byte1 = frame1.tostring()
    byte2 = frame2.tostring()

    return byte1, byte2

def byte2array(frames1, frames2, frames3, frames4, fs=16000, record_time=3) :
    raw_array = np.zeros((fs * record_time, 4))
    raw_array[:, 0] = np.fromstring(frames1, np.int16)
    raw_array[:, 1] = np.fromstring(frames2, np.int16)
    raw_array[:, 2] = np.fromstring(frames3, np.int16)
    raw_array[:, 3] = np.fromstring(frames4, np.int16)

    return raw_array

def audio_vad(frames1, frames2, frames3, frames4, fs = 16000, record_time = 3) :
    history1 = vad.is_speech(frames1)
    history2 = vad.is_speech(frames2)
    history3 = vad.is_speech(frames3)
    history4 = vad.is_speech(frames4)

    raw_array = byte2array(frames1, frames2, frames3, frames4, fs, record_time)
    vad_result = np.zeros((fs*record_time, 4))
    filtered = np.zeros((fs*record_time, 4))

    for i in range(len(history1)) :
        if history1[i] == 1 or history2[i] == 1 or history3[i] == 1 or history4[i] == 1 :
            vad_result[i * 480 : (i + 1) * 480, 0] = 1
            filtered[i * 480 : (i + 1) * 480, 0] = raw_array[i * 480 : (i + 1) * 480, 0]
            vad_result[i * 480 : (i + 1) * 480, 1] = 1
            filtered[i * 480 : (i + 1) * 480, 1] = raw_array[i * 480 : (i + 1) * 480, 1]
            vad_result[i * 480 : (i + 1) * 480, 2] = 1
            filtered[i * 480: (i + 1) * 480, 2] = raw_array[i * 480 : (i + 1) * 480, 2]
            vad_result[i * 480 : (i + 1) * 480, 3] = 1
            filtered[i * 480: (i + 1) * 480, 3] = raw_array[i * 480 : (i + 1) * 480, 3]

    voice_act = list()
    active = False
    idx = 0
    for i in range(len(vad_result)) :
        if vad_result[i,0] == 1 and active == False :
            active = True
            idx = i
        if vad_result[i,0] == 0 and active == True :
            active = False
            voice_act.append(raw_array[idx:i, :])
        if i == len(vad_result) -1 and active == True :
            active = False
            voice_act.append(raw_array[idx:i, :])

    return vad_result, filtered, voice_act


def array_spectrogram(raw_array, N = 1024, fs=16000) :
    spec = list()
    spec0 = stft(raw_array[:, 0], n_fft=N, hop_length=int(N / 2), win_length=N, window='bohman')
    spec1 = stft(raw_array[:, 1], n_fft=N, hop_length=int(N / 2), win_length=N, window='bohman')
    spec2 = stft(raw_array[:, 2], n_fft=N, hop_length=int(N / 2), win_length=N, window='bohman')
    spec3 = stft(raw_array[:, 3], n_fft=N, hop_length=int(N / 2), win_length=N, window='bohman')

    spec.append(spec0)
    spec.append(spec1)
    spec.append(spec2)
    spec.append(spec3)

    return spec

def srp_phat(raw_array, W, Q_polar) :#, ax1, ax2) :

    Q = W.shape[0]
    num_of_microphone = 4
    spec = array_spectrogram(raw_array=raw_array)
    Y = np.zeros(Q, dtype=np.complex_)

    for iframe in range(spec[1].shape[1]) :
        for iter_Q in range(Q):
            iter_mic = 0
            for iter_mic1 in range(num_of_microphone) :
                for iter_mic2 in range(iter_mic1 + 1, num_of_microphone) :
                    X1 = spec[iter_mic1][:, iframe]
                    X2 = spec[iter_mic2][:, iframe]
                    X = np.multiply(X1, X2.conjugate())
                    X = np.divide(X, np.abs(X))
                    Y[iter_Q] += np.dot(W[iter_Q, iter_mic, :],X)
                    iter_mic += 1
    Y = np.real(Y)
    top4 = Y.argsort()[-4:][::-1]
    #bestY = np.zeros(Y.shape)
    #bestY[top3] = Y[top3]
    polar = Q_polar[:, top4]
    for i in range(4) :
        polar[1,i] = rad2deg(polar[1,i])
    #print(polar)
    angle = np.average(polar[1,:])

    if abs(np.max(polar[1,:]) - np.min(polar[1,:])) > 30 or 85 <= angle <= 95 :
        return int(angle), True
    else :
        return int(angle), True

def signal_handler(sig, frame) :
    print("Exit")
    sys.exit(0)

def main() :
    signal.signal(signal.SIGINT, signal_handler)
    _ = input("Press Enter to Start")

    for steps in range(10) :
        stereo1, stereo2 = audio_recording(CHANNELS=2, RATE=16000, _RECORD_SECONDS=60, step = steps)
        frame1, frame4 = stereo2mono(stereo1)
        frame2, frame3 = stereo2mono(stereo2)
        #raw_array = byte2array(frame1, frame2, frame3, frame4, record_time=5)
        #for i in range(4) :
        #    plt.subplot(4,1,i+1)
        #    plt.plot(raw_array[:,i])
        #plt.show()

if __name__ == '__main__':
    main()
