- quantized accuracy: 86.1%
- model size 10,704
- Data: wavs at 16kHz
- Model type: tiny_conv 
- Preprocessing: micro

Deployment: 

1. works relatively well, but it is stuck on green or red 
2. should introduce the silence and unknown categories, but that requires retraining because it expects only 2 outputs not 4 right now and throws an error on 4. 