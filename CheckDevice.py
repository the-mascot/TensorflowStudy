from tensorflow.python.client import device_lib
# Tensoflow-gpu 환경 체크
print(device_lib.list_local_devices())
