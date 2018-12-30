
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import numpy as np

from yolo import yolo_model, custom_loss
from preprocessing import parse_annotation, BatchGenerator




LABELS = ['crater']
IMAGE_H, IMAGE_W = 224, 224
GRID_H,  GRID_W  = 7 , 7
BOX              = 5
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD    = 0.3#0.5
NMS_THRESHOLD    = 0.3#0.45
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
CLASS_SCALE      = 1.0
TRUE_BOX_BUFFER  = 20
BATCH_SIZE       = 128







train_image_folder = "./images_train/"
train_annot_folder = "./annotations_train/"
test_image_folder = "./images_test/"
test_annot_folder = "./annotations_test/"

generator_config = {
    'IMAGE_H'         : IMAGE_H,
    'IMAGE_W'         : IMAGE_W,
    'GRID_H'          : GRID_H,
    'GRID_W'          : GRID_W,
    'BOX'             : BOX,
    'LABELS'          : LABELS,
    'CLASS'           : len(LABELS),
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
}

def normalize(image):
    return image / 255.

train_imgs, seen_train_labels = parse_annotation(train_annot_folder, train_image_folder, labels=LABELS)
train_batch = BatchGenerator(train_imgs, generator_config, norm=normalize)

test_imgs, seen_valid_labels = parse_annotation(test_annot_folder, test_image_folder, labels=LABELS)
test_batch = BatchGenerator(test_imgs, generator_config, norm=normalize, jitter=False)


early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0.001,
                           patience=3,
                           mode='min',
                           verbose=1)

checkpoint = ModelCheckpoint('weights_yolo_crater.h5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=1)



optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

yolo_model.compile(loss=custom_loss, optimizer=optimizer)

yolo_model.fit_generator(generator        = train_batch,
                    steps_per_epoch  = len(train_batch),
                    epochs           = 100,
                    verbose          = 1,
                    validation_data  = test_batch,
                    validation_steps = len(test_batch),
                    callbacks        = [early_stop, checkpoint],
                    max_queue_size   = 3)
