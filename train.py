from config import *
import numpy as np
import pandas as pd
import os
import gc
from utils import loses, generators
from config import TEST_DIR, TRAIN_DIR, BASE_DIR
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam


train = os.listdir(TRAIN_DIR)
test = os.listdir(TEST_DIR)


segmented_ships = pd.read_csv(os.path.join(BASE_DIR, 'train_ship_segmentations_v2.csv'))
pix = pd.notna(segmented_ships.EncodedPixels)
print(pix.sum(), 'segmented_ships from', segmented_ships[pix].ImageId.nunique(), 'images')
print((~pix).sum(), 'empty images from', segmented_ships.ImageId.nunique(), 'total images')
segmented_ships.head()

segmented_ships['ships'] = segmented_ships['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = segmented_ships.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
segmented_ships.drop(['ships'], axis=1, inplace=True)
print(unique_img_ids.loc[unique_img_ids.ships>=2].head())


SAMPLES_PER_GROUP = 4000
balanced_train_df = unique_img_ids.groupby('ships').apply(
    lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x
)


train_ids, valid_ids = train_test_split(balanced_train_df,
                                        test_size=0.2,
                                        stratify=balanced_train_df['ships'])
train_df = pd.merge(segmented_ships, train_ids)
valid_df = pd.merge(segmented_ships, valid_ids)
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')

train_gen = generators.make_image_gen(train_df)
train_x, train_y = next(train_gen)
valid_x, valid_y = next(generators.make_image_gen(valid_df, VALID_IMG_COUNT))

dg_args = dict(featurewise_center=False,
               samplewise_center=False,
               rotation_range=45,
               width_shift_range=0.1,
               height_shift_range=0.1,
               shear_range=0.01,
               zoom_range=[0.9, 1.25],
               horizontal_flip=True,
               vertical_flip=True,
               fill_mode='reflect',
               data_format='channels_last')

if AUGMENT_BRIGHTNESS:
    dg_args[' brightness_range'] = [0.5, 1.5]
image_gen = ImageDataGenerator(**dg_args)

if AUGMENT_BRIGHTNESS:
    dg_args.pop('brightness_range')
label_gen = ImageDataGenerator(**dg_args)

cur_gen = generators.create_aug_gen(train_gen) # for next package of augmented data
t_x, t_y = next(cur_gen)

gc.enable()
gc.collect() # In order to accommodate augmented data that may occupy more memory.


def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(strides):
    return layers.UpSampling2D(strides)


if UPSAMPLE_MODE == 'DECONV':
    upsample = upsample_conv
else:
    upsample = upsample_simple


def unet(pretrained_weights = None, input_size = (256, 256, 3)):
    inputs = layers.Input(input_size)

    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    d = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[d])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

seg_model = unet()

weight_path="weight_metrics_and_model/{}_weights.best.hdf5".format('seg_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1,
                             mode='max', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.2,
                                   patience=3,
                                   verbose=1, mode='max', min_delta=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_dice_coef",
                      mode="max",
                      patience=15)
callbacks_list = [checkpoint, early, reduceLROnPlat]


def fit():
    seg_model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=loses.FocalLoss,
                      metrics=[loses.dice_coef, 'binary_accuracy'])

    step_count = min(MAX_TRAIN_STEPS, train_df.shape[0] // BATCH_SIZE)
    aug_gen = generators.create_aug_gen(generators.make_image_gen(train_df))
    los_history = [seg_model.fit(aug_gen,
                                 steps_per_epoch=step_count,
                                 epochs=MAX_TRAIN_EPOCHS,
                                 validation_data=(valid_x, valid_y),
                                 callbacks=callbacks_list,
                                 workers=1
                                 )]
    return los_history


while True:
    loss_history = fit()
    if np.min([mh.history['val_dice_coef'] for mh in loss_history]) < -0.2:
        break

if IMG_SCALING is not None:
    trained_model = models.Sequential()
    trained_model.add(layers.AvgPool2D(IMG_SCALING, input_shape=(None, None, 3)))
    trained_model.add(seg_model)
    trained_model.add(layers.UpSampling2D(IMG_SCALING))
else:
    trained_model = seg_model
trained_model.save('weight_metrics_and_model/trained_model.h5')
