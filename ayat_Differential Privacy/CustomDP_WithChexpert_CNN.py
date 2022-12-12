import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_federated as tff
import tensorflow_privacy as tfp
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
import nest_asyncio
import os
from six.moves import range
import six
from random import sample
import data_loaders as dl
import h5py
import tempfile
import h5py
import hdf5_client_data
from configparser import ConfigParser
nest = tf.nest

from dp_accounting import dp_event
from dp_accounting import privacy_accountant
from dp_accounting import privacy_loss_distribution
from dp_accounting.pld import pld_privacy_accountant
from tensorflow.keras.applications import InceptionResNetV2 as IRNV2
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, GlobalAveragePooling2D, AveragePooling2D,MaxPooling2D, Activation
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16,DenseNet121
NeighborRel = privacy_accountant.NeighboringRelation
PLD = privacy_loss_distribution

@tff.federated_computation
def hello_world():
    return 'Hello, World!'

config_file = "./sample_config.ini"
cp = ConfigParser()
cp.read(config_file)

output_dir = cp["DEFAULT"].get("output_dir")
class_names = cp["DEFAULT"].get("class_names")
base_model_name = cp["DEFAULT"].get("base_model_name")
image_source_dir = cp["DEFAULT"].get("image_source_dir")

csv_dir = cp["TRAIN"].get("dataset_csv_dir")
batch_size = cp["TRAIN"].getint("batch_size")
epochs = cp["TRAIN"].getint("epochs")
output_weights_name = cp["TRAIN"].get("output_weights_name")
model_weights_path = os.path.join('./CheXpert-Federated-master/experiments/1/', output_weights_name)
initial_learning_rate = cp["TRAIN"].getfloat("initial_learning_rate")

client_list = cp["FEDERATED"].get("client_list").split(",")
client_list_test = cp["FEDERATED"].get("client_list_test").split(",")
shuffle_buffer = cp["FEDERATED"].getint("shuffle_buffer")
num_clients = cp["FEDERATED"].getint("number_clients")

output_dir, class_names, base_model_name, image_source_dir, csv_dir, batch_size, epochs, output_weights_name, model_weights_path, initial_learning_rate, client_list, shuffle_buffer, num_clients


total_clients=3849

def load_data():
    output = {}
    for client_item in client_list_test:
        _file_name = client_item + '.csv'
        _file_path = os.path.join(csv_dir, _file_name)
        _x, _y = dl.load_data_file(_file_path, '')
        _client_map = {}
        _client_map['label'] = _y
        _client_map['pixels'] = _x
        id=client_item[6:12:1]
        output[id] = _client_map
    return output

def load_data_test():
    output = {}
    for client_item in client_list:
        _file_name = client_item + '.csv'
        _file_path = os.path.join(csv_dir, _file_name)
        _x, _y = dl.load_data_file(_file_path, '')
        _client_map = {}
        _client_map['label'] = _y
        _client_map['pixels'] = _x
        id = client_item[6:12:1]
        output[id] = _client_map
    return output

def create_fake_hdf5(arg_data):
    fd, filepath = tempfile.mkstemp()
    # close the pre-opened file descriptor immediately to avoid leaking.
    os.close(fd)
    with h5py.File(filepath, 'w') as f:
        examples_group = f.create_group('examples')
        for user_id, data in six.iteritems(arg_data):
            user_group = examples_group.create_group(str(user_id))
            for name, values in six.iteritems(data):
                user_group.create_dataset(name, data=values)
    return filepath

def preprocess(dataset):
    def element_fn(element):
        return collections.OrderedDict(
            x=tf.expand_dims(element['pixels'], -1), y=element['label'])

    return (dataset.repeat(1).map(element_fn).shuffle(buffer_size=500).batch(32,drop_remainder=False))

def preprocess_test(dataset):
    def element_fn(element):
        return collections.OrderedDict(
            x=tf.expand_dims(element['pixels'], -1), y=element['label'])
    return dataset.map(element_fn).batch(32, drop_remainder=False)

client_data_train = hdf5_client_data.HDF5ClientData(create_fake_hdf5(load_data()))
client_data_test = hdf5_client_data.HDF5ClientData(create_fake_hdf5(load_data_test()))

train_data= client_data_train.preprocess(preprocess)
test_data = preprocess_test(client_data_test.create_tf_dataset_from_all_clients())

example_dataset = train_data.create_tf_dataset_for_client(
    train_data.client_ids[0])


def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x))
            for x in client_ids]

sample_clients = client_data_train.client_ids[0:num_clients]
federated_train_data = make_federated_data(client_data_train, sample_clients)



def build_model():
    base_model = DenseNet121(include_top=False, weights='imagenet')

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(12, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)


    return tff.learning.from_keras_model(
        keras_model=model,
        loss=tf.keras.losses.BinaryCrossentropy(),
        input_spec=test_data.element_spec,
        metrics=[tf.keras.metrics.BinaryAccuracy()])

tff.backends.native.set_local_python_execution_context(clients_per_thread=5)

def make_plot(data_frame):
    plt.figure(figsize=(15, 5))
    dff = data_frame.rename(
        columns={'binary_accuracy': 'Accuracy', 'loss': 'Loss'})
    plt.subplot(121)
    sns.lineplot(data=dff, x='Round', y='Accuracy', hue='NoiseMultiplier', palette='dark')
    plt.subplot(122)
    sns.lineplot(data=dff, x='Round', y='Loss', hue='NoiseMultiplier', palette='dark')
    plt.show()

def train(rounds, noise_multiplier, clients_per_round, data_frame):
    # Using the `dp_aggregator` here turns on differential privacy with adaptive
    # clipping.
    dp_query=tff.aggregators.DifferentiallyPrivateFactory.gaussian_adaptive(
       noise_multiplier=noise_multiplier,
       initial_l2_norm_clip=0.1,
       target_unclipped_quantile=0.5,
       clients_per_round=clients_per_round,
       clipped_count_stddev=None,
    )
    sampling_prob = clients_per_round / total_clients
    learning_process = tff.learning.algorithms.build_unweighted_fed_avg(
        build_model,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.001),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0,momentum=0.9),
        model_aggregator=dp_query,
        use_experimental_simulation_loop=True)
    eval_process = tff.learning.build_federated_evaluation(build_model,
                                                           use_experimental_simulation_loop=True)
    # Training loop.
    state = learning_process.initialize()
    for round in range(rounds):
        if round % 5 == 0:
            model_weights = learning_process.get_model_weights(state)
            metrics = eval_process(model_weights, [test_data])["eval"]
            if round < 10 or round % 10 == 0:
                print(f'Round {round:3d}: {metrics}')
            data_frame = data_frame.append({'Round': round,
                                            'NoiseMultiplier': noise_multiplier,
                                            **metrics}, ignore_index=True)

        # Sample clients for a round. Note that if your dataset is large and
        # sampling_prob is small, it would be faster to use gap sampling.
        x = np.random.uniform(size=total_clients)
        sampled_clients = [
            train_data.client_ids[i] for i in range(total_clients)
            if x[i] < sampling_prob]
        sampled_train_data = [
            train_data.create_tf_dataset_for_client(client)
            for client in sampled_clients]
        # Use selected clients for update.
        result = learning_process.next(state, sampled_train_data)
        state = result.state
        metrics = result.metrics
    model_weights = learning_process.get_model_weights(state)
    metrics = eval_process(model_weights, [test_data])['eval']
    print(f'Round {rounds:3d}: {metrics}')
    data_frame = data_frame.append({'Round': rounds,
                                    'NoiseMultiplier': noise_multiplier,
                                    **metrics}, ignore_index=True)

    return data_frame

total_clients= 3849
data_frame = pd.DataFrame()
rounds = 200
clients_per_round = 50

for noise_multiplier in [0.0, 0.5, 0.75, 1.0]:
    print(f'Starting training with noise multiplier: {noise_multiplier}')
    data_frame = train(rounds, noise_multiplier, clients_per_round, data_frame)
    print()

make_plot(data_frame)

data_frame = pd.DataFrame()
base_noise_multiplier = 0.5
base_clients_per_round = 50
target_delta = 1e-5
target_eps = 2

def get_epsilon(clients_per_round):
    # If we use this number of clients per round and proportionally
    # scale up the noise multiplier, what epsilon do we achieve?
    q = clients_per_round / total_clients
    noise_multiplier = base_noise_multiplier
    noise_multiplier *= clients_per_round / base_clients_per_round
    gaussian_event = dp_event.GaussianDpEvent(noise_multiplier)
    accountant = pld_privacy_accountant.PLDAccountant()
    accountant.compose(gaussian_event, 1)
    eps=accountant.get_epsilon(target_delta)
    print(eps)
    print(noise_multiplier)
    return clients_per_round, eps, noise_multiplier

def find_needed_clients_per_round():
    hi = get_epsilon(base_clients_per_round)
    if hi[1] < target_eps:
        return hi

    # Grow interval exponentially until target_eps is exceeded.
    while True:
        lo = hi
        hi = get_epsilon(1.5 * lo[0])
        if hi[1] < target_eps:
            break

    # Binary search.
    while hi[0] - lo[0] > 1:
        mid = get_epsilon((lo[0] + hi[0]) // 2)
        if mid[1] > target_eps:
            lo = mid
        else:
            hi = mid

    return hi

clients_per_round, _, noise_multiplier = find_needed_clients_per_round()
print(f'To get ({target_eps}, {target_delta})-DP, use {clients_per_round} '
      f'clients with noise multiplier {noise_multiplier}.')


rounds = 200
noise_multiplier = noise_multiplier
clients_per_round = clients_per_round
data_frame = pd.DataFrame()
data_frame = train(rounds, noise_multiplier, clients_per_round, data_frame)
make_plot(data_frame)