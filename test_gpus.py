# Import TensorFlow and TensorFlow Datasets

import tensorflow_datasets as tfds
import tensorflow as tf

import os

if __name__ == '__main__':

    datasets, info = tfds.load(
        name='mnist', with_info=True, as_supervised=True)

    mnist_train, mnist_test = datasets['train'], datasets['test']

    strategy = tf.distribute.MirroredStrategy()

    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    num_train_examples = info.splits['train'].num_examples
    num_test_examples = info.splits['test'].num_examples

    BUFFER_SIZE = 10000

    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255

        return image, label

    train_dataset = mnist_train.map(scale).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu',
                                   input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])

    # Define the checkpoint directory to store the checkpoints

    checkpoint_dir = '../training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    # Function for decaying the learning rate.
    # You can define any decay function you need.
    def decay(epoch):
        if epoch < 3:
            return 1e-3
        elif epoch >= 3 and epoch < 7:
            return 1e-4
        else:
            return 1e-5

    model.fit(train_dataset, epochs=12)
