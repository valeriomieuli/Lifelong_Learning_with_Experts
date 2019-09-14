import tensorflow as tf
import tensorflow.keras.applications as applications
import numpy as np

from warnings import filterwarnings
filterwarnings("ignore")


def load_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()

    n_classes = int(len(np.unique(y_train)))
    n_train_samples_per_class, n_test_samples_per_class = int(X_train.shape[0] / n_classes), int(
        X_test.shape[0] / n_classes)

    y_train, y_test = y_train[:, 0], y_test[:, 0]

    x_tmp, y_tmp = np.zeros((X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3])), np.zeros(
        X_train.shape[0])
    for i in range(n_classes):
        indexes = np.where(y_train == i)[0]
        x_tmp[i * n_train_samples_per_class:(i + 1) * n_train_samples_per_class, :, :, :] = X_train[indexes, :, :, :]
        y_tmp[i * n_train_samples_per_class:(i + 1) * n_train_samples_per_class] = y_train[indexes]
    X_train, y_train = x_tmp, y_tmp

    x_tmp, y_tmp = np.zeros((X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3])), np.zeros(
        X_test.shape[0])
    for i in np.unique(y_test):
        indexes = np.where(y_test == i)[0]
        x_tmp[i * n_test_samples_per_class:(i + 1) * n_test_samples_per_class, :, :, :] = X_test[indexes, :, :, :]
        y_tmp[i * n_test_samples_per_class:(i + 1) * n_test_samples_per_class] = y_test[indexes]
    X_test, y_test = x_tmp, y_tmp

    return n_classes, n_train_samples_per_class, n_test_samples_per_class, X_train, y_train, X_test, y_test


def get_task_data(task, n_train_samples_per_task, n_test_samples_per_task, X_train, y_train, X_test, y_test):
    from sklearn.utils import shuffle
    X_train_task = X_train[task * n_train_samples_per_task:(task + 1) * n_train_samples_per_task, :, :, :]
    y_train_task = tf.keras.utils.to_categorical(
        y_train[task * n_train_samples_per_task:(task + 1) * n_train_samples_per_task])[:, task * 10:(task + 1) * 10]
    X_train_task, y_train_task = shuffle(X_train_task, y_train_task)

    X_test_task = X_test[:(task + 1) * n_test_samples_per_task, :, :, :]
    y_test_task = y_test[:(task + 1) * n_test_samples_per_task]

    return X_train_task, y_train_task, X_test_task, tf.keras.utils.to_categorical(y_test_task)


def compute_mean_and_std(model_name, X, input_shape):
    if model_name == 'Xception':
        model = applications.Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'VGG16':
        model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'VGG19':
        model = applications.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'ResNet50':
        model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'InceptionResNetV2':
        model = applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'InceptionV3':
        model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'MobileNet':
        model = applications.MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'DenseNet121':
        model = applications.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'DenseNet169':
        model = applications.DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'DenseNet201':
        model = applications.DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'NASNetMobile':
        model = applications.NASNetMobile(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'NASNetLarge':
        model = applications.NASNetLarge(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        assert (False), "Specified base model is not available !"

    features = model.predict(X)[:, 0, 0, :]

    return features.mean(axis=0), features.std(axis=0)


def build_autoencoder(base_model_name, input_shape, imagenet_mean, imagenet_std, hidden_layer_size, n_classes,
                      weight_decay):
    if base_model_name == 'Xception':
        base_model = applications.Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'VGG16':
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'VGG19':
        base_model = applications.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'ResNet50':
        base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'InceptionResNetV2':
        base_model = applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'InceptionV3':
        base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'MobileNet':
        base_model = applications.MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'DenseNet121':
        base_model = applications.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'DenseNet169':
        base_model = applications.DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'DenseNet201':
        base_model = applications.DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'NASNetMobile':
        base_model = applications.NASNetMobile(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'NASNetLarge':
        base_model = applications.NASNetLarge(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        assert (False), "Specified base model is not available !"

    n_features = base_model.output.shape[-1]

    x = base_model.output
    x = tf.keras.layers.Lambda(lambda x: (x - imagenet_mean) / imagenet_std)(x)  # normalization
    x = tf.keras.layers.Activation(activation='sigmoid', name='encoder')(x)  # sigmoid
    x = tf.keras.layers.Dense(units=hidden_layer_size, activation=None)(x)  # encoding
    x = tf.keras.layers.Activation(activation='relu')(x)  # relu
    x = tf.keras.layers.Dense(units=n_features, activation=None, name='decoder')(x)  # decoding
    x = tf.keras.layers.Dense(units=n_classes, activation='sigmoid')(
        x)  # x = tf.keras.layers.Activation(activation='sigmoid')(x) # sigmoid

    model = tf.keras.Model(inputs=base_model.input, outputs=x[:, 0, 0, :])

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(tf.keras.regularizers.l2(weight_decay)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(tf.keras.regularizers.l2(weight_decay)(layer.bias))

    return model
