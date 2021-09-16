import onnx

from onnx_tf.backend import prepare
import tensorflow as tf

import numpy as np


onnx_model = onnx.load("onnx_model.onnx")  # load onnx model

tf_rep = prepare(onnx_model)  # prepare tf representation

# Input nodes to the model
print("inputs:", tf_rep.inputs)

# Output nodes from the model
print("outputs:", tf_rep.outputs)

# All nodes in the model
print("tensor_dict:")
print(tf_rep.tensor_dict)

tf_rep.export_graph("hey_fourth_brain")  # export the model

# Below didnt work, it was changing the sizes & got below error
# conv.cc:349 input->dims->data[3] != filter->dims->data[3] (0 != 1)
# Converting a SavedModel.
# converter = tf.lite.TFLiteConverter.from_saved_model("hey_fourth_brain")
# converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
#    "hey_fourth_brain/saved_model.pb", tf_rep.inputs, tf_rep.outputs, input_shapes={"input": [1, 1, 40, 61]}
# )

# so used below method
model = tf.saved_model.load("hey_fourth_brain")
input_shape = [1, 1, 40, 61]
func = tf.function(model).get_concrete_function(input=tf.TensorSpec(shape=input_shape, dtype=np.float32, name="input"))
converter = tf.lite.TFLiteConverter.from_concrete_functions([func])

tflite_model = converter.convert()
open("hey_fourth_brain.tflite", "wb").write(tflite_model)
