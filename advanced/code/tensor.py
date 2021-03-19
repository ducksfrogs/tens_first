import tensorflow as tf

print(tf.add(1,2))
print(tf.add([1,2],[3,4]))

x = tf.random.uniform([3,3])

print("Is there usable GPU.")
print(tf.config.experimental.list_physical_devices("GPU"))

print("GUP #0")
print(x.device.endswith('GPU:0'))

import time

def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)

    result = time.time() - start

print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)


if tf.config.experimental.list_physical_devices("GPU"):
    print("On GPU")
    with tf.device("GPU:0"):
        x = tf.random.uniform([1000, 1000])
