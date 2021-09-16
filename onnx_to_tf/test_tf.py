import numpy as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

model = tf.saved_model.load("hey_fourth_brain")

print(list(model.signatures.keys()))

infer = model.signatures["serving_default"]
print(infer.structured_outputs)

input_shape = [1, 1, 40, 61]
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

output = infer(tf.constant(input_data))["output"]
print(output)
print(f"model predicted - {output.numpy().argmax()}")
