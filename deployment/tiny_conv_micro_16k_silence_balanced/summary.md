- quantized accuracy: 93.0%
- Quantized model is 18712 bytes
- Data: wavs at 16kHz
- Model type: tiny_conv 
- Preprocessing: micro
- Float model accuracy is 91.823899% (Number of test samples=159)
- Quantized model accuracy is 91.194969% (Number of test samples=159
- wanted words: snoring (500), off (500) 
- silence, unknown

Deployment: 

- a suppression_ms = 4s does not work (very little sensitivity and lots of misses on snoring) 
(4s is the interval between snores)

// not very sensitive to snoring (false negatives)
                int32_t average_window_duration_ms = 1200,
                             uint8_t detection_threshold = 200,
                             int32_t suppression_ms = 1500,
                             int32_t minimum_count = 3
// even less sensitive 
int32_t average_window_duration_ms = 1500,
                             uint8_t detection_threshold = 200,
                             int32_t suppression_ms = 1500,
                             int32_t minimum_count = 3

                    
// too sensitive to no_snoring (false positives)
                int32_t average_window_duration_ms = 1200,
                             uint8_t detection_threshold = 150,
                             int32_t suppression_ms = 1500,
                             int32_t minimum_count = 3

// best results (very few FP and FN )
int32_t average_window_duration_ms = 800,
                             uint8_t detection_threshold = 200,
                             int32_t suppression_ms = 1500,
                             int32_t minimum_count = 2
                             );
int32_t average_window_duration_ms = 800,
                             uint8_t detection_threshold = 200,
                             int32_t suppression_ms = 2000,
                             int32_t minimum_count = 2

int32_t average_window_duration_ms = 600,
                             uint8_t detection_threshold = 200,
                             int32_t suppression_ms = 1000,
                             int32_t minimum_count = 2
