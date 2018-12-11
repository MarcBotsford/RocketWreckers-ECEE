import tensorflow as tf
device_name = tf.test.is_gpu_available(
  cuda_only=False,
  min_cuda_compute_capability=None
)
print(device_name)