## Run with MFCC: 
- Change the sample_rate to 44100 either as parameter or in the default value flags in freeze.py and train.py scripts. 
- Make sure "window_stride" is at 20.0 in freeze and train

- Float model is 16508 bytes
- Quantized model is 4944 bytes
- Final test accuracy = 52.8% (N=108)
- Issue with overfitting (validation acc goes above traning accuracy)

