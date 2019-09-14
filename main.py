import tensorflow as tf
# import tensorflow.keras.applications as applications
import numpy as np
from scipy.spatial.distance import euclidean
import math

from warnings import filterwarnings
filterwarnings("ignore")

import utils

################### Settings ###################
n_tasks = 10
base_autoencoder_name = 'ResNet50'
data_shape = (32, 32, 3)
autoencoder_hidden_layer_size = 256
weight_decay = 0.005
learning_rate = 0.01
batch_size = 128
epochs = 600
################################################

n_classes, n_train_samples_per_class, n_test_samples_per_class, X_train, y_train, X_test, y_test = utils.load_data()

mean_image, std_image = utils.compute_mean_and_std('ResNet50', X_train, data_shape)

encoders = []
decoders = []

# train autoencoders
for task in range(5):
    # create autoencoder for the current new task
    autoencoder = utils.build_autoencoder(base_autoencoder_name, data_shape, mean_image, std_image,
                                          autoencoder_hidden_layer_size, int(n_classes / n_tasks), weight_decay)

    # get train and test data for the current task
    X_train_task, y_train_task, X_test_task, y_test_task = utils.get_task_data(task,
                                                                               int(
                                                                                   n_classes / n_tasks) * n_train_samples_per_class,
                                                                               int(
                                                                                   n_classes / n_tasks) * n_test_samples_per_class,
                                                                               X_train,
                                                                               y_train,
                                                                               X_test,
                                                                               y_test)

    # train the autoencoder for the current task
    print("---------------- Training autoencoder for the task %d ----------------" % task)
    autoencoder.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
    autoencoder.fit(X_train_task, y_train_task, batch_size, epochs, shuffle=True)
    print("\n\n")

    encoders.append(tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output))
    decoders.append(tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('decoder').output))

    # task cumulative test
    if task > 0:
        reconstruction_errors = np.zeros((len(y_test_task), task + 1))
        task_labels = [int(cl / n_tasks) for cl in
                       np.argmax(y_test_task, axis=1)]  # compute the task each class label belongs to
        if task > 0:
            reconstruction_errors = np.zeros((len(y_test_task), task + 1))
            task_labels = [int(cl / n_tasks) for cl in
                           np.argmax(y_test_task, axis=1)]  # compute the task each class label belongs to
            for t in range(task + 1):
                encoded_features = (encoders[t].predict(X_test_task, batch_size=256))[:, 0, 0, :]
                decoded_features = (decoders[t].predict(X_test_task, batch_size=256))[:, 0, 0, :]
                reconstruction_errors[:, t] = [euclidean(enc_feat, dec_feat) for enc_feat, dec_feat in
                                               zip(encoded_features, decoded_features)]

            probabilities = np.zeros((len(y_test_task), task + 1))
            for i in range(len(reconstruction_errors)):
                denom = np.sum([math.exp(-err / 2) for err in reconstruction_errors[i]])
                for t, err in enumerate(reconstruction_errors[i]):
                    probabilities[i, t] = math.exp(-err / 2) / denom

            print("[Task %d] Autoencoders accuracy = %.4f" % (task,
                                                              np.sum([np.argmin(prob) == task_lab for prob, task_lab in
                                                                      zip(probabilities, task_labels)]) / len(
                                                                  X_test_task)))
