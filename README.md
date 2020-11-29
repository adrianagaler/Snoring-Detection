# Data Preprocessing 

## Run 
 - *pre-requisites:* install the software (see below)
 - run `python3 preprocessing.py` (make sure to update the names of the files and directories)

## Khan: 
A dataset of 1000 sound samples is developed in this project. The dataset contains 2 classes—snoring
sounds and non-snoring sounds. Each class has 500 samples. The snoring sounds were collected from
different online sources [27–31]. The non-snoring sounds were also collected from similar online sources.
Then silences were trimmed from the sound files and the files were split to equal-sized one-second
duration files using WavePad Sound Editor [32]. Thus, each sample has a duration of one second

Ten categories of non-snoring sounds are collected, and each category has
50 samples. The ten categories are baby crying, the clock ticking, the door opened and closed, total
silence and the minor sound of the vibration motor of the gadget, toilet flashing, siren of emergency
vehicle, rain and thunderstorm, streetcar sounds, people talking, and background television news.

### Summary: 
1. FFT points: 512
2. number of filters in the filterbank: 10 + 22 = 32
3. number of cepstral coefficients: 32
4. sound sample frame size: 30 ms 
5. processed sample is a 32x32 image 


## Feature Extraction
1. the Mel frequency cepstral coefficients (MFCCs) are calculated for each sample
 - to compress information into a small number of coefficients based on an understanding of the human
ear.
 - the time-domain audio signal is first divided into 32 frames of 30 ms

2. Construct Mel Filter Banks: 
 - create the powerspectrum, by converting from frequency to energy space, using the power_spectrum function in the speechpy lib
 - create 10 equally spread bands given the minimum frequency (100) and the max frequency (1000) in steps of 100
 - convert the points in the linearly interpolated space from Hz to Mels 
 - create the other 22 equally spread bands (filterbanks) for lowest freq = 1000 up to max 
 - apply the mel matrix of 32 merged bands to the powerspectrum 
 - get the coefficients by calculating the energies and the DCT values


**Software**
 SpeechPy Library: https://speechpy.readthedocs.io/en/latest/intro/introductions.html ; https://speechpy.readthedocs.io/_/downloads/en/latest/pdf/
 Install using: pip install speechpy 
 OR 
 Locally:git clone https://github.com/astorfi/speech_feature_extraction.git then python setup.py develop

## More on Mel Frequency Cepstral Coefficients: https://www.youtube.com/watch?v=4_SH2nfbQZ8
 

 ## Extensions 
 Pre-emphasis to amplify high frequencies, as described here: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

# Deployment
- train_snoring_model.ipynb has training script 
- To edit training model script see 'Snoring-Detection/deployment/tensorflow1/tensorflow/examples/speech_commands'
- rename snoring dataset directories to 'snoring' and 'no_snoring' 
- change data pathways 
- include "\_background_noise\_" directory from wake words dataset in dataset directory 
- make sure to check all paths to make sure they are specific to your setup 
