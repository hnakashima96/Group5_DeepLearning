#basic library 
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale 

#track example to understand the data
file = 'blues.00000.wav'

#waveform 
#signal of the track
#sr = sampling rate (1/t) is the frequency 
signal, sr = librosa.load(file, sr=22050, duration= 10) # sr*T = 22050 * 30
librosa.display.waveplot(signal, sr=sr)
#visualization of the waveform
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.savefile('waveform.png')
plt.close()

#fit -> spectrum (FFT)
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))  #functiong that gives a evenly space numbers 

#we just want the half of the frequency analysis 
left_f = frequency[:int(len(frequency)/2)]
left_m = magnitude[:int(len(frequency)/2)]

#visualization of the spectrum 
plt.plot(left_f, left_m)
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.savefile('spectrum.png')
plt.close()

#stft -> spectrogram (amplitude as function of frequency and time)
n_fft = 2048 #window of analysis
hop_length = 512 

stft = librosa.core.stft(signal, hop_length = hop_length, n_fft = n_fft)
spectrogram = np.abs(stft)

log_spec = librosa.amplitude_to_db(spectrogram)

#visualization of the spectrogram
librosa.display.specshow(log_spec, sr=sr, hop_length=hop_length)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar()
plt.savefile('spectrogram.png')
plt.close()

#MFCCs
#Extract the mfccs performing short Fourie Transform 
MFCCs = librosa.feature.mfcc(signal, n_fft= n_fft,
                             hop_length = hop_length,
                             n_mfcc = 13)

MFCCT = MFCCs.T   
print(MFCCT.tolist())                          
#visualization of the MFCC
librosa.display.specshow(MFCCT, sr=sr, hop_length=hop_length)
plt.xlabel('Time')
plt.ylabel('MFCCs')
plt.colorbar()
plt.savefile('MFCC.png')
plt.close()
