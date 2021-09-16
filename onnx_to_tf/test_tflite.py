import numpy as np
import tensorflow as tf

TFLITE_FILE_PATH = "hey_fourth_brain.tflite"

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
interpreter.allocate_tensors()

signatures = interpreter.get_signature_list()
print(signatures)
# my_signature = interpreter.get_signature_runner()

# Get input and output tensors.
input_details = interpreter.get_input_details()
print(f"input details : {input_details}")
output_details = interpreter.get_output_details()
print(f"output_details : {output_details}")

input_shape = input_details[0]["shape"]
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

# output = my_signature(input=tf.constant(input_data))
# print(output)

# Test the model on random input data.
interpreter.set_tensor(input_details[0]["index"], input_data)
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]["index"])
