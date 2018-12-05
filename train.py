# _*_ coding:utf-8 -*-
from model import OpenNsfwModel,trainning,evaluation,log_loss
import os
import tensorflow as tf
import numpy as np
import pandas as pd


BATCH_SIZE = 24
IMAGE_SIZE = 448
testImg = []
LR = 0.001

logs_train_dir = 'train_path'
OUTPUT_PATH = 'output/output.csv' #output train log
input_path = 'input_path'

def prepare():
    global testImg
    withPath = lambda f: '{}/{}'.format(input_path, f)
    i = 0
    testImg = []
    for f in os.listdir(input_path):
        if os.path.isfile(withPath(f)):
                testImg.append(withPath(f))
                i = i + 1

def process(filename, label):
    image = tf.read_file(filename)
    ####
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32, saturate=True)
    image = tf.image.resize_images(image, (512, 512),
                                   method=tf.image.ResizeMethod.BILINEAR,
                                   align_corners=True)
    image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
    image = tf.image.encode_jpeg(image, format='', quality=75,
                                 progressive=False, optimize_size=False,
                                 chroma_downsampling=True,
                                 density_unit=None,
                                 x_density=None, y_density=None,
                                 xmp_metadata=None)
    image = tf.image.decode_jpeg(image, channels=3,
                                 fancy_upscaling=False,
                                 dct_method="INTEGER_ACCURATE")
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.crop_to_bounding_box(image, 16, 16, 448, 448)
    #####
    # image = tf.image.rot90(image, np.random.randint(-1, 2, size=1)[0], )
    # image = tf.image.random_flip_left_right(image)
    #####
    image = tf.reverse(image, axis=[2])
    VGG_MEAN = [104, 117, 123]
    image -= VGG_MEAN
    return image, filename, label


def training_preprocess(image, filename, label):
  flip_image = tf.image.random_flip_left_right(image)
  return flip_image, filename, label


def update_csv(loss,accuracy,epoch):
    global OUTPUT_PATH
    if os.path.exists(OUTPUT_PATH):
        upcsv = pd.read_csv(filepath_or_buffer=OUTPUT_PATH,
                            sep=',')
        get_loss = upcsv["loss"].values
        get_accuracy = upcsv["accuracy"].values
        get_epoch = upcsv["epoch"].values
    else:
        get_loss = []
        get_accuracy = []
        get_epoch = []
    get_loss = np.append(get_loss, loss)
    get_accuracy = np.append(get_accuracy, accuracy)
    get_epoch = np.append(get_epoch, epoch)
    df = pd.DataFrame(
        {'loss': get_loss, 'accuracy': get_accuracy, 'epoch': get_epoch})
    df.to_csv(OUTPUT_PATH, index=False, sep=',')


def run():
    global BATCH_SIZE
    global testImg
    global LR
    global logs_train_dir
    prepare()
    images = []
    labels = []
    for line in testImg:
        images.append(line)
        filename = os.path.splitext(os.path.split(line)[1])[0]
        true_index = 0
        if filename.split('Porn')[0] == 'v':
            true_index = 1
        labels.append(true_index)

    images = tf.constant(images)
    labels = tf.constant(labels)
    images = tf.random_shuffle(images, seed=0)
    labels = tf.random_shuffle(labels, seed=0)
    data = tf.data.Dataset.from_tensor_slices((images, labels))

    data = data.shuffle(len(testImg))
    data = data.map(process, num_parallel_calls=16)
    data = data.prefetch(buffer_size=BATCH_SIZE * 8)
    batched_data = data.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE)).repeat(40)#num of epoch


    iterator = tf.data.Iterator.from_structure(batched_data.output_types,
                                               batched_data.output_shapes)
    init_op = iterator.make_initializer(batched_data)


    Y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, ], name='Y')
    model = OpenNsfwModel()
    model.build(weights_path=None, batchsize=BATCH_SIZE, tag='')
    Y_pred = model.logits

    # sup1 = model.sup1
    # sup2 = model.sup2
    # sup3 = model.sup3
    # loss = supervision_loss(Y, Y_pred, sup1, sup2, sup3)
    loss = log_loss(Y,Y_pred)

    accuracy = evaluation(Y, Y_pred)
    train_op = trainning(loss, LR)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    step = 0
    epoch = 0
    aver_acc = 0


    with tf.Session(config=config) as sess:

        saver = tf.train.Saver(max_to_keep=0)
        sess.run(tf.global_variables_initializer())
        sess.run(init_op)

        images, filenames, labels = iterator.get_next()
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('global_step',global_step)
            if global_step != '0':
                step = int(global_step) + 1
            saver.restore(sess, ckpt.model_checkpoint_path)
        record_loss = 0
        record_acc = 0
        while True:
            try:
                print('step',step)
                name,label_Y,input = sess.run([filenames,labels,images])
                get_loss,get_op,get_acc = sess.run([loss,train_op,accuracy], feed_dict={model.input: input,Y:label_Y})
                print('loss',get_loss,'accuracy',get_acc)
                record_loss = record_loss + get_loss
                record_acc = record_acc + get_acc

                aver_acc = aver_acc + get_acc
                if (step + 1) % (len(testImg) / BATCH_SIZE) == 0:
                    epoch = epoch + 1
                    ###save record
                    record_loss = float(record_loss)/(len(testImg) / BATCH_SIZE)
                    record_acc = float(record_acc) / (len(testImg) / BATCH_SIZE)
                    update_csv(record_loss, record_acc, epoch)
                    ###
                    print('epoch', epoch)
                    print('train_learning_rate', LR)
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                step = step + 1
            except tf.errors.OutOfRangeError:
                sess.run(init_op)
                break
        print(float(aver_acc)/step)



