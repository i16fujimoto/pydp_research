import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_privacy as tfp
import nest_asyncio
import matplotlib.pyplot as plt
import seaborn as sns

dataset_paths = {
  'train.csv'
}

# Create some test data for the sake of the example,
# normally we wouldn't do this.
for i, (id, path) in enumerate(dataset_paths.items()):
  with open(path, 'w') as f:
    for _ in range(i):
      f.write(f'test,0.0,{i}\n')

# Values that will fill in any CSV cell if its missing,
# must match the dtypes above.
record_defaults = ['', 0.0, 0]

@tf.function
def create_tf_dataset_for_client_fn(dataset_path):
   return tf.data.experimental.CsvDataset(
     dataset_path, record_defaults=record_defaults )


source = tff.simulation.datasets.FilePerUserClientData(
dataset_paths, create_tf_dataset_for_client_fn)


print(source.client_ids)


for x in source.create_tf_dataset_for_client('client_3'):
  print(x)

