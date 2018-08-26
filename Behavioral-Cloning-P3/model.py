import math
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Cropping2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.regularizers import l2

flags = tf.app.flags
FLAGS = flags.FLAGS

#DEFINE FLAGS VARIABLES#
flags.DEFINE_float('steering_adjustment', 0.27, "Adjustment angle.")
flags.DEFINE_integer('epochs', 25, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")

image_load_data = []
with open('./sample_data/data_training_my/driving_log.csv') as f:
    reader = csv.reader(f)
    for csv_line in reader:
        image_load_data.append(csv_line)

image_load_data = shuffle(image_load_data)
train_load_data, validation_load_data = train_test_split(
    image_load_data, test_size=.2, random_state=42)


def data_gen(load_data, batch_size):

    while True:
        shuffle(load_data)
        for offset in range(0, len(load_data), batch_size):
            images = []
            steering_angles = []
            for line in load_data[offset:offset + batch_size]:
                center_img_path = line[0]
                center_file_name = center_img_path.split('\\')[-1]
                center_image = cv2.imread(
                    './sample_data/data_training_my/IMG/{}'.format(center_file_name))
                if center_image is not None:
                    images.append(center_image)
                    # images.append(np.fliplr(center_image))

                    center_steering_angle = float(line[3])
                    steering_angles.append(center_steering_angle)
                    # steering_angles.append(-center_steering_angle)
                left_img_path = line[1]
                left_file_name = left_img_path.split('\\')[-1]
                left_image = cv2.imread(
                    './sample_data/data_training_my/IMG/{}'.format(left_file_name))
                if left_image is not None:
                    images.append(left_image)
                    # images.append(np.fliplr(left_image))

                    left_steering_angle = float(
                        line[3]) + FLAGS.steering_adjustment
                    steering_angles.append(left_steering_angle)
                    # steering_angles.append(-left_steering_angle)
                right_img_path = line[2]
                right_file_name = right_img_path.split('\\')[-1]
                right_image = cv2.imread(
                    './sample_data/data_training_my/IMG/{}'.format(right_file_name))
                if right_image is not None:
                    images.append(right_image)
                    # images.append(np.fliplr(right_image))

                    right_steering_angle = float(
                        line[3]) - FLAGS.steering_adjustment
                    steering_angles.append(right_steering_angle)
                    # steering_angles.append(-right_steering_angle)
            if len(images) == 0:
                print("Didn't find any images - continuing")
                continue

            yield shuffle(np.array(images), np.array(steering_angles))


# Training Architecture: inspired by NVIDIA architecture #
model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(90, 320, 3)))
model.add(Convolution2D(24, 5, 5, border_mode='valid',
                        subsample=(2, 2), W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, border_mode='valid',
                        subsample=(2, 2), W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, border_mode='valid',
                        subsample=(2, 2), W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same',
                        subsample=(2, 2), W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid',
                        subsample=(2, 2), W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(80, W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(40, W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(16, W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(Dense(1, W_regularizer=l2(0.001)))
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
model.summary()

print('Training the model')

model.compile(optimizer='adam', loss='mse')
model.fit_generator(
    generator=data_gen(train_load_data, FLAGS.batch_size),
    samples_per_epoch=len(train_load_data) * 3,
    nb_epoch=FLAGS.epochs,
    validation_data=data_gen(validation_load_data, FLAGS.batch_size),
    nb_val_samples=len(validation_load_data) * 3,
    verbose=1
)

print('Saved the model')
model.save('model.h5')
