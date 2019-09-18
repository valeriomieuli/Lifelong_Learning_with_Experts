import tensorflow.keras as keras
import numpy as np
from scipy.spatial.distance import euclidean
import math

from warnings import filterwarnings

filterwarnings("ignore")

import utils

################### Settings ###################
n_tasks = 10
base_autoencoder_name = 'ResNet50'
batch_size = 128
data_shape = (32, 32, 3)

autoencoder_hidden_layer_sizes = [128, 256, 512]
weight_decays = [1, 0.1, 0.001]
learning_rates = [0.1, 0.01, 0.001, 0.0001]
epsilons = [1e-07]  # to be involved in the Adagrad optimizer; currently this default value is used

epochs_per_step = 20
steps = 20
################################################

n_classes, n_train_samples_per_class, n_test_samples_per_class, X_train, y_train, X_test, y_test = utils.load_data()
n_train_samples_per_task = int(n_classes / n_tasks) * n_train_samples_per_class
n_test_samples_per_task = int(n_classes / n_tasks) * n_test_samples_per_class

mean_image, std_image = utils.compute_mean_and_std('ResNet50', X_train, data_shape)

for autoencoder_hidden_layer_size in autoencoder_hidden_layer_sizes:
    for weight_decay in weight_decays:
        for learning_rate in learning_rates:

            autoencoders = [None for _ in range(n_tasks)]
            encoders = [None for _ in range(n_tasks)]
            decoders = [None for _ in range(n_tasks)]

            print(
                "#######################################################################################################################\n"
                "HYPERPARAMETERS: {autoencoder_hidden_layer_size: %3d, weight_decay: %.4f, learning_rate: %.4f}" % (
                    autoencoder_hidden_layer_size, weight_decay, learning_rate))

            for step in range(steps):

                print()
                print("  STEP: %2d" % step)

                for task in range(n_tasks):
                    # create and compile autoencoder if it's the first step (the autoencoder does not exist yet)
                    if autoencoders[task] is None:
                        autoencoders[task] = utils.build_autoencoder(base_autoencoder_name, data_shape, mean_image,
                                                                     std_image,
                                                                     autoencoder_hidden_layer_size,
                                                                     int(n_classes / n_tasks),
                                                                     weight_decay)
                        autoencoders[task].compile(optimizer=keras.optimizers.Adagrad(lr=learning_rate),
                                                   loss='categorical_crossentropy',
                                                   metrics=['accuracy'])

                    # get train and test data for the current task
                    X_train_task, y_train_task, X_test_task, y_test_task = utils.get_task_data(task,
                                                                                               n_train_samples_per_task,
                                                                                               n_test_samples_per_task,
                                                                                               X_train,
                                                                                               y_train, X_test, y_test)

                    # train the autoencoder for the current task
                    autoencoders[task].fit(X_train_task, y_train_task, batch_size, epochs=epochs_per_step, shuffle=True,
                                           verbose=0,
                                           callbacks=[utils.FitCallback(step, task, epochs_per_step)])

                    encoders[task] = keras.Model(inputs=autoencoders[task].input,
                                                 outputs=autoencoders[task].get_layer('encoder').output)
                    decoders[task] = keras.Model(inputs=autoencoders[task].input,
                                                 outputs=autoencoders[task].get_layer('decoder').output)

                    # task cumulative test
                    if task > 0:
                        reconstruction_errors = np.zeros((len(y_test_task), task + 1))
                        probabilities = np.zeros((len(y_test_task), task + 1))
                        task_labels = [int(cl / n_tasks) for cl in
                                       np.argmax(y_test_task, axis=1)]  # compute the task each class label belongs to

                        for t in range(task + 1):
                            encoded_features = (encoders[t].predict(X_test_task, batch_size=256))[:, 0, 0, :]
                            decoded_features = (decoders[t].predict(X_test_task, batch_size=256))[:, 0, 0, :]
                            reconstruction_errors[:, t] = [euclidean(enc_feat, dec_feat) for enc_feat, dec_feat in
                                                           zip(encoded_features, decoded_features)]

                        for i in range(len(reconstruction_errors)):
                            denom = np.sum([math.exp(-err / 2) for err in reconstruction_errors[i]])
                            for t, err in enumerate(reconstruction_errors[i]):
                                probabilities[i, t] = math.exp(-err / 2) / denom

                        task_autoencoders_acc = np.sum(
                            [np.argmin(prob) == task_lab for prob, task_lab in zip(probabilities, task_labels)]) / len(
                            X_test_task)

                        print(" - Test autoencoders accuray %.5f" % task_autoencoders_acc)
                    else:
                        print()

                print()

            print()