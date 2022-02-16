from tensorflow.keras.utils import get_file
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LeakyReLU, ReLU, Conv2D, Conv2DTranspose, \
    Concatenate, concatenate, BatchNormalization, Dropout, ZeroPadding2D, \
    Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from IPython import display
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os

URl = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz"

zip_path = get_file('maps.tar.gz',
                    origin=URl,
                    extract=True)

path = os.path.join(os.path.dirname(zip_path), "maps/")


def load_img(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)  # decode to jpg file

    width = tf.shape(image)[1]
    width = width // 2

    input_img = image[:, :width, :]  # [600, :600, 3]
    target_img = image[:, width:, :]  # [600, 600:, 3]

    input_img = tf.cast(input_img, tf.float32)
    target_img = tf.cast(target_img, tf.float32)

    return input_img, target_img


IMAGE_COLS = 256  # for resize
IMAGE_ROWS = 256  # for resize
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMAGE_CHANNELS = 3  # RGB
LAMBDA = 100


def resize(input_img, target_img, height, width):
    input_img = tf.image.resize(input_img, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    target_img = tf.image.resize(target_img, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_img, target_img


def random_crop(input_img, target_image):
    stacked_image = tf.stack([input_img, target_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image,
                                         size=[2, IMAGE_ROWS, IMAGE_COLS, 3])
    return cropped_image[0], cropped_image[1]


def normal(input_img, target_img):
    input_img = (input_img / 127.5) - 1  # [-1, 1]
    target_img = (target_img / 127.5) - 1  # [-1, 1]
    return input_img, target_img


def random_jitter(input_img, target_img):
    input_img, target_img = resize(input_img, target_img, 286, 286)

    input_img, target_img = random_crop(input_img, target_img)

    if tf.random.uniform(()) >= 0.5:  # random flip
        input_img = tf.image.flip_left_right(input_img)  # flip
        target_img = tf.image.flip_left_right(target_img)  # flip

    return input_img, target_img


def load_train(image_file):
    input_img, target_img = load_img(image_file)  # load image
    input_img, target_img = random_jitter(input_img, target_img)  # apply random jitter
    input_img, target_img = normal(input_img, target_img)  # normalizer image

    return input_img, target_img


def load_test(image_file):
    input_img, target_img = load_img(image_file)
    input_img, target_img = resize(input_img, target_img, IMAGE_ROWS,
                                   IMAGE_COLS)
    input_img, target_img = normal(input_img, target_img)

    return input_img, target_img


train_dataset = tf.data.Dataset.list_files(path + "train/*.jpg")
train_dataset = train_dataset.map(load_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)  # mapping dataset to load_test function
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(path + "val/*.jpg")
test_dataset = test_dataset.map(load_test)  # mapping dataset to load_test function
test_dataset = test_dataset.batch(BATCH_SIZE)


def downsample(filters, size, batch_norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)  # mean 0 and standard deviation 0.02

    model = Sequential()
    model.add(Conv2D(filters, size, strides=2, padding="same",
                     kernel_initializer=initializer))

    if batch_norm:
        model.add(BatchNormalization())

    model.add(LeakyReLU())  # f(x) = return x if x >= 0  
                            # f(x) = return alpha * x if x < 0

    return model


def upsample(filters, size, dropout=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    model = Sequential()

    model.add(Conv2DTranspose(filters, size, strides=2, padding="same",
                              kernel_initializer=initializer,
                              use_bias=False))

    model.add(BatchNormalization())

    if dropout:
        model.add(Dropout(0.5))  # add dropout to prevent over fitting

    model.add(ReLU())  # max(x, 0)

    return model


# U-NET
def Generator():
    inputs = Input(shape=[IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS])

    downs = [
        downsample(64, 4, batch_norm=False),  # (batch_size, 128, 128, 64)
        downsample(256, 4),  # (batch_size, 64, 64, 256)
        downsample(512, 4),  # (batch_size, 32, 23, 512)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    ups = [
        upsample(512, 4, dropout=True),  # (batch_size, 1, 1, 1024)
        upsample(512, 4, dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last_layer = Conv2DTranspose(IMAGE_CHANNELS,
                                 4,
                                 strides=2,
                                 padding="same",
                                 kernel_initializer=initializer,
                                 activation="tanh")  # (batch_size, 256, 256, 3)

    x = inputs

    skips = []
    for down in downs:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])  # E.g [1, 2, 3] ---> [3, 2]

    for up, skip in zip(ups, skips):
        x = up(x)
        x = Concatenate()([x, skip])  # skip connection

    x = last_layer(x)

    return Model(inputs=inputs, outputs=x)


generator = Generator()
# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

loss_object = BinaryCrossentropy(from_logits=True)


def generator_loss(disc_gen_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_gen_output), disc_gen_output) # sigmoid

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) # MAE(Mean Absolute Error)

    total_gen_loss = gan_loss + (LAMBDA * l1_loss) # total loss

    return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    input_image = Input(shape=[IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS], name="input_img")
    target_image = Input(shape=[IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS], name="target_image")

    x = concatenate([input_image, target_image])

    down1 = downsample(64, 4, batch_norm=False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = ZeroPadding2D()(down3)
    conv_layer = Conv2D(512, 4, strides=1,
                        kernel_initializer=initializer,
                        use_bias=False)(zero_pad1)
    batch_norm1 = BatchNormalization()(conv_layer)
    leaky_relu = LeakyReLU()(batch_norm1)
    zero_pad2 = ZeroPadding2D()(leaky_relu)
    last_layer = Conv2D(1, 4, strides=1,
                        kernel_initializer=initializer)(zero_pad2)

    return Model([input_image, target_image], last_layer)


discriminator = Discriminator()


# tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

def disc_loss(disc_real_out, disc_gen_out):
    real_loss = loss_object(tf.ones_like(disc_real_out), disc_real_out)
    fake_loss = loss_object(tf.zeros_like(disc_gen_out), disc_gen_out)
    return real_loss + fake_loss


generator_optim = Adam(2e-4, beta_1=0.5)
discriminator_optim = Adam(2e-4, beta_1=0.5)

checkpoint_path = './train_check'
checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optim=generator_optim,
                                 discriminator_optim=discriminator_optim,
                                 generator=generator,
                                 discriminator=discriminator)

def generate_images(model, test_image, target):
    prediction = model(test_image, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_image[0], target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)  # [0, 1]
        plt.axis('off')
    plt.show()


@tf.function
def train_step(input_image, target_image):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target_image], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target_image)
        dis_loss = disc_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(dis_loss,
                                                 discriminator.trainable_variables)

    generator_optim.apply_gradients(zip(generator_gradients,
                                        generator.trainable_variables))
    discriminator_optim.apply_gradients(zip(discriminator_gradients,
                                            discriminator.trainable_variables))

    return dis_loss, gen_total_loss


Epochs = 50


def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait=True)

        for example_input, example_target in test_ds.take(1):
            generate_images(generator, example_input, example_target)

        print("Epoch: ", epoch)

        for n, (input_image, target) in train_ds.enumerate():
            print('=', end='')
            dis_loss, gen_loss = train_step(input_image, target)
            if (n + 1) % 100 == 0:
                print(" Discriminator loss: {}, Generator Loss: {}".format(dis_loss, gen_loss))
                print()
        print()

        checkpoint.save(file_prefix=checkpoint_prefix) # saves data every epoch
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))

fit(train_dataset, Epoch, test_dataset)
