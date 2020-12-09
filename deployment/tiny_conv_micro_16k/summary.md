- model accuracy: 66.7%
- Float model is 36036 bytes
- quantized model size 10704 bytes 
- Data: wavs at 16kHz
- Model type: tiny_conv 
- Preprocessing: micro


Deplpyment: 
1. board does not react. Only very rare green light with false positives
2. It might be that for the rest of the noises (non snoring) it treats them as silence, thats why there  is no logging for Heard non snoring( it corresponds to the silence index)