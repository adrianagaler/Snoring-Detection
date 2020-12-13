1. Low Latency
    - training accuracy : 90.6 % 
    - Quantized size : 884360 bytes 
    - Cannot deploy due to memory corruption (inputs don't fit in buffer allocated, model expects larger) and size 
2. Low latency svdf 
    - training accuracy : 87.4 % 
    - Quantized size : 37,449 bytes
    - Cannot deploy due to memory corruption (deployment code not compatible with model) and size 
3. Single FC
    - training accuracy : 75.5 % 
    - Quantized size : 8872 bytes
    - Invoke time: 1mS
    - Memory consumption: 2544 bytes 
4. Tiny Conv
    - training accuracy : 91.8 % 
    - Quantized size : 18712 bytes
    - Invoke time: 60 mS
    - Memory consumption: 7152 bytes 
5. Conv 
    - training accuracy : 93.7 % 
    - Quantized size :  308168 bytes
6. Tiny Embedding Conv
    - training accuracy : 89.9  %    
    - Quantized size : 8864  bytes
    - Error: "Requested feature_data_ size 536907080 doesn't match 1960. Feature generation failed"


Deployment Modifications: 

1. recognize_commands.h

Change the parameters to RecognizeCommands() object as follows 
tflite::ErrorReporter* error_reporter,
int32_t average_window_duration_ms = 800,
uint8_t detection_threshold = 200,
int32_t suppression_ms = 1500,
int32_t minimum_count = 2

2. in micro_speeech.ino
For models above 18k size change constexpr int kTensorArenaSize = 100 * 1024;

3. micro_features_model.cpp 
Add the model.cc and the size

4. micro_features_micro_model_settings.h
Change constexpr int kCategoryCount = 4;
Keep the unknown index = 1, silence index = 0

5. micro_features_micro_model_settings.cpp
const char* kCategoryLabels[kCategoryCount] = {
    "silence",
    "no_snoring",
    "snoring",
    "no_snoring",
};

6. arduino_command_responder.cpp

if (found_command[0] == 's' & found_command[1] == 'n') {
      last_command_time = current_time;
      digitalWrite(LEDG, LOW);  // Green for snoring
    }

    if (found_command[0] == 'n' & found_command[1] == 'o') {
      last_command_time = current_time;
      digitalWrite(LEDR, LOW);  // Red for no snoring
    }


