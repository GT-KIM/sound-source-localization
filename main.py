from librosa import stft
from utils import *
import sys
import signal
from socket import *

N = 1024
vad = WebRTCVAD()
_vad = webrtcvad.Vad(2)

def stereo2mono(stereo) :
    buf = np.fromstring(stereo, dtype='int16')
    frame1 = buf[0::2]
    frame2 = buf[1::2]
    byte1 = frame1.tostring()
    byte2 = frame2.tostring()

    return byte1, byte2

def byte2array(frames1, frames2, frames3, frames4, fs=16000, record_time=0.064) :
    raw_array = np.zeros((int(fs * record_time*len(frames1)), 4))
    for i in range(len(frames1)) :
        raw_array[i*1024:(i+1)*1024, 0] = np.fromstring(frames1[i], np.int16)
        raw_array[i*1024:(i+1)*1024, 1] = np.fromstring(frames2[i], np.int16)
        raw_array[i*1024:(i+1)*1024, 2] = np.fromstring(frames3[i], np.int16)
        raw_array[i*1024:(i+1)*1024, 3] = np.fromstring(frames4[i], np.int16)

    return raw_array

def audio_vad_old(frames1, frames2, frames3, frames4, fs = 16000, record_time = 2) :
    raw_array = byte2array(frames1, frames2, frames3, frames4, fs, record_time)
    history1 = vad.is_speech(frames1)
    history2 = vad.is_speech(frames2)
    history3 = vad.is_speech(frames3)
    history4 = vad.is_speech(frames4)

    vad_result = np.zeros((fs*record_time, 4))
    filtered = np.zeros((fs*record_time, 4))

    for i in range(66) :
        if history1[i] == 1 or history2[i] == 1 or history3[i] == 1 or history4[i] == 1 :
            vad_result[i * 480 : (i + 1) * 480, 0] = 1
            filtered[i * 480 : (i + 1) * 480, 0] = raw_array[i * 480 : (i + 1) * 480, 0]
            vad_result[i * 480 : (i + 1) * 480, 1] = 1
            filtered[i * 480 : (i + 1) * 480, 1] = raw_array[i * 480 : (i + 1) * 480, 1]
            vad_result[i * 480 : (i + 1) * 480, 2] = 1
            filtered[i * 480: (i + 1) * 480, 2] = raw_array[i * 480 : (i + 1) * 480, 2]
            vad_result[i * 480 : (i + 1) * 480, 3] = 1
            filtered[i * 480: (i + 1) * 480, 3] = raw_array[i * 480 : (i + 1) * 480, 3]
    plt.subplot(12,1,1)
    plt.plot(raw_array[:,0])
    plt.subplot(12,1,2)
    plt.plot(raw_array[:,1])
    plt.subplot(12,1,3)
    plt.plot(raw_array[:,2])
    plt.subplot(12,1,4)
    plt.plot(raw_array[:,3])
    plt.subplot(12,1,5)
    plt.plot(vad_result[:,0])
    plt.subplot(12,1,6)
    plt.plot(vad_result[:,1])
    plt.subplot(12,1,7)
    plt.plot(vad_result[:,2])
    plt.subplot(12,1,8)
    plt.plot(vad_result[:,3])
    plt.subplot(12,1,9)
    plt.plot(filtered[:,0])
    plt.subplot(12,1,10)
    plt.plot(filtered[:,1])
    plt.subplot(12,1,11)
    plt.plot(filtered[:,2])
    plt.subplot(12,1,12)
    plt.plot(filtered[:,3])
    plt.show()

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

def audio_vad(frames1, frames2, frames3, frames4, fs = 16000, record_time = 0.064) :
    active_frame = list()
    frame11 = frames1[:2*480]
    frame12 = frames1[2*480:2*960]
    frame21 = frames2[:2*480]
    frame22 = frames2[2*480:2*960]
    frame31 = frames3[:2*480]
    frame32 = frames3[2*480:2*960]
    frame41 = frames4[:2*480]
    frame42 = frames4[2*480:2*960]
    active = (_vad.is_speech(frame11, fs) + _vad.is_speech(frame12, fs) + _vad.is_speech(frame21, fs) +
              _vad.is_speech(frame22, fs) + _vad.is_speech(frame31, fs) + _vad.is_speech(frame32, fs) +
              _vad.is_speech(frame41, fs) + _vad.is_speech(frame42, fs)) > 6
    active_frame.append(int(active))
    active_frame = np.array(active_frame)
    return active_frame

def array_spectrogram(raw_array, N = 1024, fs=16000) :
    spec = list()
    spec0 = stft(raw_array[:, 0].astype('float32'), n_fft=N, hop_length=N, win_length=N,center=False, window='bohman')
    spec1 = stft(raw_array[:, 1].astype('float32'), n_fft=N, hop_length=N, win_length=N,center=False, window='bohman')
    spec2 = stft(raw_array[:, 2].astype('float32'), n_fft=N, hop_length=N, win_length=N,center=False, window='bohman')
    spec3 = stft(raw_array[:, 3].astype('float32'), n_fft=N, hop_length=N, win_length=N,center=False, window='bohman')

    spec.append(spec0)
    spec.append(spec1)
    spec.append(spec2)
    spec.append(spec3)

    return spec

def srp_phat(raw_array,vad_flag, W, Q_polar) :#, ax1, ax2) :

    Q = W.shape[0]
    num_of_microphone = 4
    spec = array_spectrogram(raw_array=raw_array)
    Y = np.zeros(Q, dtype=np.complex_)

    for iframe in range(spec[1].shape[1]) :
        if vad_flag[iframe] == 1 :
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
    print(angle)
    if abs(np.max(polar[1,:]) - np.min(polar[1,:])) > 30 or 85 <= angle <= 95 :
        return int(angle), False
    else :
        return int(angle), True

def signal_handler(sig, frame) :
    print("Exit")
    sys.exit(0)

def main() :
    offline = np.load("E:\\Test/W_fastest.npz")
    W = offline['W']
    Q_array = offline['Q']
    Q_polar = offline['Q_polar']
    signal.signal(signal.SIGINT, signal_handler)

    connectionSock = socket(AF_INET, SOCK_STREAM)
    connectionSock.connect(('127.0.0.1', 1234))
    _ = input("Press Enter to Start")
    global_frame1 = list()
    global_frame2 = list()
    global_frame3 = list()
    global_frame4 = list()
    global_flag = list()

    while True :
        stereo1, stereo2 = audio_record(CHANNELS=2, RATE=16000, _RECORD_SECONDS=0.064)
        frame1, frame4 = stereo2mono(stereo1)
        frame2, frame3 = stereo2mono(stereo2)
        vad_flags = audio_vad(frame1, frame2, frame3, frame4)
        global_frame1.append(frame1)
        global_frame2.append(frame2)
        global_frame3.append(frame3)
        global_frame4.append(frame4)
        global_flag.append(vad_flags)

        if len(global_flag) > 5 :
            global_flag.pop(0)
            global_frame1.pop(0)
            global_frame2.pop(0)
            global_frame3.pop(0)
            global_frame4.pop(0)
        print(global_flag.count(1))

        if len(global_flag) == 5 and global_flag.count(1) > 2 :
            raw_array = byte2array(global_frame1, global_frame2, global_frame3, global_frame4, record_time=0.064)
            angle, flag = srp_phat(raw_array,global_flag, W=W, Q_polar=Q_polar)
            global_frame1 = list()
            global_frame2 = list()
            global_frame3 = list()
            global_frame4 = list()
            global_flag = list()
        else :
            #print("Speech is too short")
            flag=False
        if flag == True :
            action_done_flag = 0
            connectionSock.send(bytes([angle]))
            action_done_flag = connectionSock.recv(1)
            if action_done_flag == b'\x01' :
                print("Moving Finished")
        #if flag == True :
        #    print(angle)
if __name__ == '__main__':
    main()
