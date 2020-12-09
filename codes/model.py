#!/usr/bin/env python
# coding: utf-8

# # Train a Simple Audio Recognition Model
import ipdb
import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-arc', type=str, default = 'single_fc')
parser.add_argument('-preprocess', type=str, default = 'micro')
args = parser.parse_args()

PREPROCESS = args.preprocess # Other options 'micro', 'average', 'mfcc'
MODEL_ARCHITECTURE = args.arc # Other options include: single_fc, conv, tiny_conv,
                      # low_latency_conv, low_latency_svdf, tiny_embedding_conv


WANTED_WORDS = "snoring,no_snoring"

# The number of steps and learning rates can be specified as comma-separated
# lists to define the rate at each stage. For example,
# TRAINING_STEPS=12000,3000 and LEARNING_RATE=0.001,0.0001
# will run 12,000 training loops in total, with a rate of 0.001 for the first
# 8,000, and 0.0001 for the final 3,000.
TRAINING_STEPS = "2500,2500"
LEARNING_RATE = "0.001,0.001"

# Calculate the total number of steps, which is used to identify the checkpoint
# file name.
TOTAL_STEPS = str(sum(map(lambda string: int(string), TRAINING_STEPS.split(","))))

# Print the configuration to confirm it
print("Training these words: %s" % WANTED_WORDS)
print("Training steps in each stage: %s" % TRAINING_STEPS)
print("Learning rate in each stage: %s" % LEARNING_RATE)
print("Total number of training steps: %s" % TOTAL_STEPS)



number_of_labels = WANTED_WORDS.count(',') + 1
number_of_total_labels = number_of_labels

# Constants which are shared during training and inference

WINDOW_STRIDE = 20


# Constants used during training only
VERBOSITY = 'DEBUG' # WARN, DEBUG
EVAL_STEP_INTERVAL = '25'
SAVE_STEP_INTERVAL = '100'

# Constants for training directories and filepaths
DATASET_DIR = '/n/dtak/jyao/Snoring-Detection/Snoring_Dataset_@16000'
LOGS_DIR = MODEL_ARCHITECTURE+'_logs/'
TRAIN_DIR = MODEL_ARCHITECTURE+'_train/' # for training checkpoints and other files.
import os
os.system('rm -rf {} {}'.format(LOGS_DIR,TRAIN_DIR))
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)
if not os.path.exists(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)

# Constants for inference directories and filepaths
MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
  os.mkdir(MODELS_DIR)
MODEL_TF = os.path.join(MODELS_DIR, MODEL_ARCHITECTURE+'_model.pb')
MODEL_TFLITE = os.path.join(MODELS_DIR, MODEL_ARCHITECTURE+'_model.tflite')
FLOAT_MODEL_TFLITE = os.path.join(MODELS_DIR, MODEL_ARCHITECTURE+'_float_model.tflite')
MODEL_TFLITE_MICRO = os.path.join(MODELS_DIR, MODEL_ARCHITECTURE+'_model.cc')
SAVED_MODEL = os.path.join(MODELS_DIR, MODEL_ARCHITECTURE+'_saved_model')

QUANT_INPUT_MIN = 0.0
QUANT_INPUT_MAX = 26.0
QUANT_INPUT_RANGE = QUANT_INPUT_MAX - QUANT_INPUT_MIN


import tensorflow as tf



os.system("python ../deployment/tensorflow1/tensorflow/examples/speech_commands/train.py --data_dir={} --data_url='' --wanted_words={} --preprocess={} --window_stride={} --model_architecture={} --how_many_training_steps={} --learning_rate={} --train_dir={} --summaries_dir={} --verbosity={} --eval_step_interval={} --save_step_interval={} --dropout=0 --optimizer='momentum'".format(DATASET_DIR,WANTED_WORDS,PREPROCESS,WINDOW_STRIDE,MODEL_ARCHITECTURE,TRAINING_STEPS, LEARNING_RATE,TRAIN_DIR,LOGS_DIR,VERBOSITY,EVAL_STEP_INTERVAL,SAVE_STEP_INTERVAL))


os.system('rm -rf {}'.format(SAVED_MODEL))
os.system("python ../deployment/tensorflow1/tensorflow/examples/speech_commands/freeze.py --wanted_words={} --window_stride_ms={} --preprocess={} --model_architecture={} --start_checkpoint={}{}.ckpt-{} --save_format=saved_model --output_file={}".format(WANTED_WORDS,WINDOW_STRIDE,PREPROCESS,MODEL_ARCHITECTURE,TRAIN_DIR,MODEL_ARCHITECTURE,TOTAL_STEPS,SAVED_MODEL))



import sys
# We add this path so we can import the speech processing modules.
sys.path.append("../deployment/tensorflow1/tensorflow/examples/speech_commands/")
import input_data
import models
import numpy as np


# In[35]:


SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0
BACKGROUND_VOLUME_RANGE = 0
TIME_SHIFT_MS = 100.0

DATA_URL = '' #'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10


# In[36]:


model_settings = models.prepare_model_settings(
    len(input_data.prepare_words_list(WANTED_WORDS.split(','))),
    SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
    WINDOW_STRIDE, FEATURE_BIN_COUNT, PREPROCESS)
audio_processor = input_data.AudioProcessor(
    DATA_URL, DATASET_DIR,
    WANTED_WORDS.split(','), VALIDATION_PERCENTAGE,
    TESTING_PERCENTAGE, model_settings, LOGS_DIR)


# In[37]:


#with tf.Session() as sess:
with tf.compat.v1.Session() as sess:
  float_converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
  float_tflite_model = float_converter.convert()
  float_tflite_model_size = open(FLOAT_MODEL_TFLITE, "wb").write(float_tflite_model)
  print("Float model is %d bytes" % float_tflite_model_size)

  converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.inference_input_type = tf.compat.v1.lite.constants.INT8 #tf.lite.constants.INT8
  converter.inference_output_type = tf.compat.v1.lite.constants.INT8 #tf.lite.constants.INT8
  def representative_dataset_gen():
    for i in range(100):
      data, _ = audio_processor.get_data(1, i*1, model_settings,
                                         BACKGROUND_FREQUENCY, 
                                         BACKGROUND_VOLUME_RANGE,
                                         TIME_SHIFT_MS,
                                         'testing',
                                         sess)

      flattened_data = np.array(data.flatten(), dtype=np.float32).reshape(1, 1960)
      yield [flattened_data]
  converter.representative_dataset = representative_dataset_gen
  tflite_model = converter.convert()
  tflite_model_size = open(MODEL_TFLITE, "wb").write(tflite_model)
  print("Quantized model is %d bytes" % tflite_model_size)


def run_tflite_inference(tflite_model_path, model_type="Float"):
  # Load test data
  np.random.seed(0) # set random seed for reproducible test results.
  #with tf.Session() as sess:
  with tf.compat.v1.Session() as sess:
    test_data, test_labels = audio_processor.get_data(
        -1, 0, model_settings, BACKGROUND_FREQUENCY, BACKGROUND_VOLUME_RANGE,
        TIME_SHIFT_MS, 'testing', sess)
  test_data = np.expand_dims(test_data, axis=1).astype(np.float32)
  print(" AUDIO PROCESSOR OUTPUT ", audio_processor.output_.shape)
  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(tflite_model_path)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  # For quantized models, manually quantize the input data from float to integer
  if model_type == "Quantized":
    input_scale, input_zero_point = input_details["quantization"]
    test_data = test_data / input_scale + input_zero_point
    test_data = test_data.astype(input_details["dtype"])

  correct_predictions = 0
  for i in range(len(test_data)):
    interpreter.set_tensor(input_details["index"], test_data[i])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    top_prediction = output.argmax()
    correct_predictions += (top_prediction == test_labels[i])

  print('%s model accuracy is %f%% (Number of test samples=%d)' % (
      model_type, (correct_predictions * 100) / len(test_data), len(test_data)))


# In[39]:


# Compute float model accuracy
run_tflite_inference(FLOAT_MODEL_TFLITE)

# Compute quantized model accuracy
run_tflite_inference(MODEL_TFLITE, model_type='Quantized')


# ## Generate a TensorFlow Lite for MicroControllers Model
# Convert the TensorFlow Lite model into a C source file that can be loaded by TensorFlow Lite for Microcontrollers.

# In[16]:


# Install xxd if it is not available

# first install xxd
# !apt-get update && apt-get -qq install xxd

# Convert to a C source file
os.system('xxd -i {} > {}'.format(MODEL_TFLITE,MODEL_TFLITE_MICRO))

# Update variable names
#REPLACE_TEXT = MODEL_TFLITE.replace('/', '_').replace('.', '_')
#!sed -i 's/'{REPLACE_TEXT}'/g_model/g' {MODEL_TFLITE_MICRO}


# In[17]:


# print(REPLACE_TEXT)
print(MODEL_TFLITE_MICRO)
print(MODEL_TFLITE)


