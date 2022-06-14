import tensorflow as tf
import numpy as np
from librosa.util import fix_length
import src.utils.path_utils as pu
import src.config as config
import src.constants as constants
import os
import src.utils.audio_utils as au
from tqdm import tqdm


def create_dataset_classification(sample_rate=None, train_test="train", batch_size=16, shuffle=True, shuffle_buffer=10000, num_workers=1, drop_remainder=True, max_samples=None) -> tf.data.Dataset:
    """ DEPRECATED
    Creates a dataset for classification.
    :param train_test: "train" or "test"
    :param batch_size: batch size
    :param shuffle: whether to shuffle the data
    :param shuffle_buffer: buffer size for shuffling
    :param num_workers: number of workers
    :param drop_remainder: whether to drop the last batch if it is not full
    :param max_samples: maximum number of samples to load
    :return: dataset
    """
    if train_test == "train":
        pos_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'positive_samples'), trainning=True)
        neg_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'negative_samples'), trainning=True)
        hard_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'hard_samples'), trainning=True)

    elif train_test == "test":
        pos_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'positive_samples'), trainning=False)
        neg_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'negative_samples'), trainning=False)
        hard_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'hard_samples'), trainning=False)

    else:
        raise ValueError("train_test must be either 'train' or 'test'")

    if sample_rate is None:
        sample_rate = au.load_audio_file(os.path.join(
            pos_path, os.listdir(pos_path)[10]))[1]
        print(sample_rate)

    pos_paths = [os.path.join(pos_path, file) for file in os.listdir(pos_path)]
    np.random.shuffle(pos_paths)
    neg_paths = [os.path.join(neg_path, file) for file in os.listdir(neg_path)]
    np.random.shuffle(neg_paths)
    hard_paths = [os.path.join(hard_path, file)
                  for file in os.listdir(hard_path)]
    np.random.shuffle(hard_paths)

    if max_samples is None:
        max_samples = len(os.listdir(pos_path))
    else:
        max_samples = min(max_samples, len(os.listdir(pos_path)))

    pos_samples = []
    neg_samples = []
    hard_samples = []
    for i in tqdm(range(max_samples)):
        pos_samples.append(au.compute_mel_spectrogram(au.load_audio_file(pos_paths[i], sr=sample_rate)[
            0], fs=sample_rate, n_fft=10, n_mels=2, hop_length=1))
        neg_samples.append(au.compute_mel_spectrogram(au.load_audio_file(neg_paths[i], sr=sample_rate)[
            0], fs=sample_rate, n_fft=10, n_mels=2, hop_length=1))
        hard_samples.append(au.compute_mel_spectrogram(au.load_audio_file(hard_paths[i], sr=sample_rate)[
            0], fs=sample_rate, n_fft=10, n_mels=2, hop_length=1))

    pos_dataset = tf.data.Dataset.from_tensor_slices(
        pos_samples).map(lambda x: [x, [1, 0]])
    neg_dataset = tf.data.Dataset.from_tensor_slices(
        neg_samples).map(lambda x: [x, [0, 1]])
    hard_dataset = tf.data.Dataset.from_tensor_slices(
        hard_samples).map(lambda x: [x, [0, 1]])

    dataset = pos_dataset.concatenate(neg_dataset).concatenate(hard_dataset)

    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset = dataset.prefetch(2)

    return dataset


def create_dataset_from_generator(sample_rate=None, fixed_size=None, train_test="train", batch_size=16, shuffle=True, shuffle_buffer=10000, num_workers=1, drop_remainder=True, max_samples=None) -> tf.data.Dataset:
    """
    Creates a dataset from a generator.
    :param generator: generator function
    :param batch_size: batch size
    :param shuffle: whether to shuffle the data
    :param shuffle_buffer: buffer size for shuffling
    :param num_workers: number of workers
    :param drop_remainder: whether to drop the last batch if it is not full
    :return: dataset
    """

    if train_test == "train":
        pos_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'positive_samples'), trainning=True)
        neg_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'negative_samples'), trainning=True)
        hard_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'hard_samples'), trainning=True)

    elif train_test == "test":
        pos_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'positive_samples'), trainning=False)
        neg_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'negative_samples'), trainning=False)
        hard_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'hard_samples'), trainning=False)

    else:
        raise ValueError("train_test must be either 'train' or 'test'")

    if sample_rate is None:
        sample_rate = au.load_audio_file(os.path.join(
            pos_path, os.listdir(pos_path)[10]))[1]
        print(sample_rate)

    if fixed_size is None:
        fixed_size = constants.mean_pos_size

    if max_samples is None:
        max_samples = len(os.listdir(pos_path))
    else:
        max_samples = min(max_samples, len(os.listdir(pos_path))-2)

    pos_paths = [os.path.join(pos_path, file) for file in os.listdir(pos_path)]
    np.random.shuffle(pos_paths)
    neg_paths = [os.path.join(neg_path, file) for file in os.listdir(neg_path)]
    np.random.shuffle(neg_paths)
    hard_paths = [os.path.join(hard_path, file)
                  for file in os.listdir(hard_path)]
    np.random.shuffle(hard_paths)

    index_generator = list(range(max_samples))

    pos_dataset = tf.data.Dataset.from_generator(
        lambda: index_generator, tf.uint16)
    neg_dataset = tf.data.Dataset.from_generator(
        lambda: index_generator, tf.uint16)
    hard_dataset = tf.data.Dataset.from_generator(
        lambda: index_generator, tf.uint16)

    def _load_mel_spectro(index, sample_type):
        if sample_type == "pos":
            return au.compute_mel_spectrogram(fix_length(au.load_audio_file(pos_paths[index], sr=sample_rate)[
                0], size=fixed_size, mode='symmetric'), fs=sample_rate, n_fft=10, n_mels=3, hop_length=1), [1, 0]
        elif sample_type == "neg":
            return au.compute_mel_spectrogram(fix_length(au.load_audio_file(neg_paths[index], sr=sample_rate)[
                0], size=fixed_size, mode='symmetric'), fs=sample_rate, n_fft=10, n_mels=3, hop_length=1), [0, 1]
        elif sample_type == "hard":
            return au.compute_mel_spectrogram(fix_length(au.load_audio_file(hard_paths[index], sr=sample_rate)[
                0], size=fixed_size, mode='symmetric'), fs=sample_rate, n_fft=10, n_mels=3, hop_length=1), [0, 1]
        else:
            raise ValueError(
                "sample_type must be either 'pos', 'neg' or 'hard'")

    pos_dataset = pos_dataset.map(lambda i: tf.py_function(func=_load_mel_spectro, inp=[i, "pos"],
                                                           Tout=[tf.float32, tf.float32]), num_parallel_calls=num_workers, deterministic=True)
    neg_dataset = neg_dataset.map(lambda i: tf.py_function(func=_load_mel_spectro, inp=[i, "neg"],
                                                           Tout=[tf.float32, tf.float32]), num_parallel_calls=num_workers, deterministic=True)
    hard_dataset = hard_dataset.map(lambda i: tf.py_function(func=_load_mel_spectro, inp=[i, "hard"],
                                                             Tout=[tf.float32, tf.float32]), num_parallel_calls=num_workers, deterministic=True)

    dataset = pos_dataset.concatenate(neg_dataset).concatenate(hard_dataset)

    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset = dataset.prefetch(2)
    return dataset


def create_dataset_birdnet(sample_rate=None, fixed_size=None, train_test="train", batch_size=16, shuffle=True, shuffle_buffer=10000, num_workers=1, drop_remainder=True, max_samples=None) -> tf.data.Dataset:
    """
    Create a dataloader for birdnet fine-tuning.
    :param sample_rate: sample rate of the audio
    :param fixed_size: fixed size of the audio
    :param train_test: train or test
    :param batch_size: batch size
    :param shuffle: whether to shuffle the dataset
    :param shuffle_buffer: buffer size for shuffling
    :param num_workers: number of workers
    :param drop_remainder: whether to drop the remainder
    :param max_samples: maximum number of samples   
    :return: dataset
    """

    if train_test == "train":
        pos_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'positive_samples'), trainning=True)
        neg_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'negative_samples'), trainning=True)
        hard_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'hard_samples'), trainning=True)

    elif train_test == "test":
        pos_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'positive_samples'), trainning=False)
        neg_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'negative_samples'), trainning=False)
        hard_path = pu.get_train_test_path(path=os.path.join(
            config.data_path, 'hard_samples'), trainning=False)

    else:
        raise ValueError("train_test must be either 'train' or 'test'")

    if sample_rate is None:
        sample_rate = au.load_audio_file(os.path.join(
            pos_path, os.listdir(pos_path)[10]))[1]
        print(sample_rate)

    if fixed_size is None:
        fixed_size = constants.mean_pos_size

    if max_samples is None:
        max_samples = len(os.listdir(pos_path))
    else:
        max_samples = min(max_samples, len(os.listdir(pos_path))-2)

    pos_paths = [os.path.join(pos_path, file) for file in os.listdir(pos_path)]
    np.random.shuffle(pos_paths)
    neg_paths = [os.path.join(neg_path, file) for file in os.listdir(neg_path)]
    np.random.shuffle(neg_paths)
    hard_paths = [os.path.join(hard_path, file)
                  for file in os.listdir(hard_path)]
    np.random.shuffle(hard_paths)

    index_generator = list(range(max_samples))

    pos_dataset = tf.data.Dataset.from_generator(
        lambda: index_generator, tf.uint16)
    neg_dataset = tf.data.Dataset.from_generator(
        lambda: index_generator, tf.uint16)
    hard_dataset = tf.data.Dataset.from_generator(
        lambda: index_generator, tf.uint16)

    def _load_audio(index, sample_type):
        if sample_type == "pos":
            return au.compute_mel_spectrogram(fix_length(au.load_audio_file(pos_paths[index], sr=sample_rate)[
                0], size=fixed_size, mode='symmetric'), fs=sample_rate, n_fft=10, n_mels=3, hop_length=1), [1, 0]
        elif sample_type == "neg":
            return au.compute_mel_spectrogram(fix_length(au.load_audio_file(neg_paths[index], sr=sample_rate)[
                0], size=fixed_size, mode='symmetric'), fs=sample_rate, n_fft=10, n_mels=3, hop_length=1), [0, 1]
        elif sample_type == "hard":
            return au.compute_mel_spectrogram(fix_length(au.load_audio_file(hard_paths[index], sr=sample_rate)[
                0], size=fixed_size, mode='symmetric'), fs=sample_rate, n_fft=10, n_mels=3, hop_length=1), [0, 1]
        else:
            raise ValueError(
                "sample_type must be either 'pos', 'neg' or 'hard'")

    pos_dataset = pos_dataset.map(lambda i: tf.py_function(func=_load_audio, inp=[i, "pos"],
                                                           Tout=[tf.float32, tf.float32]), num_parallel_calls=num_workers, deterministic=True)
    neg_dataset = neg_dataset.map(lambda i: tf.py_function(func=_load_audio, inp=[i, "neg"],
                                                           Tout=[tf.float32, tf.float32]), num_parallel_calls=num_workers, deterministic=True)
    hard_dataset = hard_dataset.map(lambda i: tf.py_function(func=_load_audio, inp=[i, "hard"],
                                                             Tout=[tf.float32, tf.float32]), num_parallel_calls=num_workers, deterministic=True)

    dataset = pos_dataset.concatenate(neg_dataset).concatenate(hard_dataset)

    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset = dataset.prefetch(2)
    return dataset


if __name__ == "__main__":
    dataset = create_dataset_from_generator(
        train_test="train", batch_size=4, max_samples=5)
    for X, y in dataset:
        print(10*"=")
        # print(_)
        print(X.shape)
        print(y.shape)
