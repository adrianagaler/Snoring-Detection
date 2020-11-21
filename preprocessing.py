import os   
import scipy.io.wavfile as wav
from speechpy import processing, feature
import numpy as np
from scipy.fftpack import dct
from PIL import Image as im 
import matplotlib.pyplot as plt
import ipdb
 
def main(main_path='.'):
    directory=main_path+'/Snoring_Dataset' # path to the dataset folder
    # assumes you have a processed data folder which contains 2 empty folders: '0' and '1'
    save_directory = main_path+"/processed_data" # where processed data goes;
    # os.listdir(directory + '/1')]
    # number_of_files = 0
    extensions = ["/1","/0"]
    for extension in extensions:
        for filename in os.listdir(directory + extension):
            if filename.endswith(".wav"): 
                
                fs, signal = wav.read(directory + extension + "/" + filename) 

                if signal.ndim != 1:
                    length = signal.shape[0] / fs
                    # print(f"length = {length}s")
                    # time = np.linspace(0., length, signal.shape[0])
                    # plt.plot(time, signal[:, 0], label="Left channel")
                    # plt.plot(time, signal[:, 1], label="Right channel")
                    # plt.show()
                    signal = signal[:,0]


                print(" shape of signal ", signal.shape) 
                print( " signal dimension ", signal.ndim)

                print(" sampling frequency ", fs)
                # print("\n \n ")
                # number_of_files +=1
                signal = processing.preemphasis(signal, cof=0.98) 

                # Stacking frames 
                frames = processing.stack_frames(signal, sampling_frequency=fs,
                                            frame_length=0.030,   # the frame size of 30 ms 
                                            frame_stride=0.030,   # decide what the stride should be 
                                            zero_padding=False)
                print(" number of frames ", frames.shape)

                # # Extracting power spectrum
                power_spectrum = processing.power_spectrum(frames, fft_points=512)
                print('power spectrum shape=', power_spectrum.shape)

                # Mel filterbanks Calculation  - using https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py as example
                # ......... the first 10 filters are placed linearly around 100, 200, . . . 1000 Hz. Above 1 kHz,
                # these bands are placed with logarithmic Mel-scale
                first_10_filterbanks = get_filterbanks(nfilt=10,nfft=512,samplerate=fs,lowfreq=100,highfreq=1000)
                last22_filterbanks = get_filterbanks(nfilt=22,nfft=512,samplerate=fs,lowfreq=1000, highfreq=None)

                # first_10  = feature.mfe(signal, sampling_frequency=fs, frame_length=0.03, frame_stride=0.01,
                #             num_filters=10, fft_length=512, low_frequency=0, high_frequency=1000)
                # last_22  = feature.lmfe(signal, sampling_frequency=fs, frame_length=0.03, frame_stride=0.01,
                #             num_filters=22, fft_length=512, low_frequency=1000)
                print(" first 10 shape ", first_10_filterbanks.shape)
                print(" last 22 shape ", last22_filterbanks.shape)
                mel_matrix = np.concatenate((first_10_filterbanks, last22_filterbanks), axis=0)

                assert(len(mel_matrix) == 32)
                print(len(mel_matrix[0]))
                assert(len(mel_matrix[0]) == len(power_spectrum[0]))

                # # Compute spectrogram  and filterbank energies
                energy = np.sum(power_spectrum,1) # this stores the total energy in each frame
                energy = np.where(energy == 0,np.finfo(float).eps,energy) # if energy is zero, we get problems with log

                features = np.dot(power_spectrum,mel_matrix.T) # compute the filterbank energies
                features = np.where(features == 0,np.finfo(float).eps,features) # if feature is zero, we get problems with log
                log_features = np.log(features)

                # DCT  -- more info: https://labrosa.ee.columbia.edu/doc/HTKBook21/node54.html
                numcep = 32 # number of cepstral coefficients
                dct_log_features = dct(log_features, type=2, axis=1)[:,:numcep]
                assert(dct_log_features.shape == (32, 32))

                # creating image object of 
                # above array 
                data = im.fromarray(dct_log_features) 
                data = data.convert("L")
                # Update the saving directory
                data.save(save_directory + extension + "/" + filename[:-3] + 'png')
                np.save(save_directory + extension + "/" + filename[:-4] + 'psd.npy',power_spectrum)
                np.save(save_directory + extension + "/" + filename[:-4] + 'dct.npy',dct_log_features)
                ############# Extract MFCC features #############
                # mfcc = feature.mfcc(signal, sampling_frequency=fs,
                #                  frame_length=0.020, frame_stride=0.01,
                #                  num_cepstral=32,
                #                  num_filters=32, fft_length=512, low_frequency=0,
                #                  high_frequency=None)
                
                #  the logarithm of all filterbank energies and
                # then their discrete cosine transform (DCT) are calculated to decorrelate the filter bank coefficients
                
    



# The following functions were taken from https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py

def get_filterbanks(nfilt=10,nfft=512,samplerate=44100,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the sample rate of the signal we are working with, in Hz. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank


def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)

def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

# Background notes: 
# - humans perceive frequencies logarithmically 


main()
