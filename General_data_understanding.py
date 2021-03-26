import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale 

file = 'blues.00000.wav'

#waveform 
#signal of the track
#sr = sampling rate (1/t) is the frequency 
signal, sr = librosa.load(file, sr=22050, duration= 10) # sr*T = 22050 * 30
# librosa.display.waveplot(signal, sr=sr)
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.show()


#fit -> spectrum (FFT)
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))  #functiong that gives a evenly space numbers 

#we just want the half of the frequency analysis 
left_f = frequency[:int(len(frequency)/2)]
left_m = magnitude[:int(len(frequency)/2)]

# plt.plot(left_f, left_m)
# plt.xlabel('Frequency')
# plt.ylabel('Magnitude')
# plt.show()

#stft -> spectrogram (amplitude as function of frequency and time)
n_fft = 2048 #window of analysis
hop_length = 512 

stft = librosa.core.stft(signal, hop_length = hop_length, n_fft = n_fft)
spectrogram = np.abs(stft)

log_spec = librosa.amplitude_to_db(spectrogram)

# librosa.display.specshow(log_spec, sr=sr, hop_length=hop_length)
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.colorbar()
# plt.show()

#MFCCs
#Extract the mfccs performing short Fourie Transform 
MFFCs = librosa.feature.mfcc(signal, n_fft= n_fft,
                             hop_length = hop_length,
                             n_mfcc = 13)

MFCCT = MFFCs.T   
print(MFCCT.tolist())                          

# librosa.display.specshow(MFCCT, sr=sr, hop_length=hop_length)
# plt.xlabel('Time')
# plt.ylabel('MFCCs')
# plt.colorbar()
# plt.show()

#spectral centroids 
spectral_centroids = librosa.feature.spectral_centroid(signal,sr=sr)[0]
frames = range(len(spectral_centroids))
time = librosa.frames_to_time(frames)


## Preprocessing 

# Normalization 
def normalize(x, axis=0):
    return minmax_scale(x, axis=axis)

#Plotting the Spectral Centroid along the waveform 
#librosa.display.waveplot(signal, sr=sr, alpha = 0.4)
#plt.plot(time, normalize(spectral_centroids), color='r')
#plt.xlabel('Time')
#plt.ylabel('Hz')
#plt.show()

#Pre Emphasis
signal_filt = librosa.effects.preemphasis(signal)

s_orig = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref= np.max)
s_preemph = librosa.amplitude_to_db(np.abs(librosa.stft(signal_filt)), ref=np.max)

# librosa.display.specshow(s_orig, y_axis='log', x_axis='time')
# plt.title('Original signal')
# plt.show()
# librosa.display.specshow(s_preemph, y_axis='log', x_axis='time')
# plt.title('Pre-emphasized signal')
# plt.show()





