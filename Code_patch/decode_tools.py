import tensorflow as tf

width_patch = 256
height_patch = 180

def decode_from_tfrecords(filename, batch_size=1):
    filename_queue = tf.train.string_input_producer(filename, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                           'img_name' : tf.FixedLenFeature([],tf.string)
                                       })  
    
    image = tf.decode_raw(features['img_raw'],tf.uint8)
    image = tf.reshape(image, [height_patch, width_patch, 3])

    label = tf.cast(features['label'], tf.int64)
    image = tf.cast(image, tf.float32)#*(1./255)-0.5#
    #label = tf.cast(label, tf.float32)#*(1./255)-0.5#
    name = tf.cast(features['img_name'],tf.string)
    min_after_dequeue = 100
    capacity = min_after_dequeue+3*batch_size
    image, label = tf.train.shuffle_batch([image, label],
                                          batch_size=batch_size, 
                                          num_threads=3, 
                                          capacity=capacity,
                                          min_after_dequeue=min_after_dequeue)
    return image, label

def decode_from_tfrecords_linjian(filename, batch_size=1):
    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                           'img_name' : tf.FixedLenFeature([],tf.string)
                                       })  
    
    image = tf.decode_raw(features['img_raw'],tf.uint8)
    image = tf.reshape(image, [int(720/4),int(1280/4),3])

    label = tf.cast(features['label'], tf.int64)
    image = tf.cast(image, tf.float32)#*(1./255)-0.5#
    #label = tf.cast(label, tf.float32)#*(1./255)-0.5#
    name = tf.cast(features['img_name'],tf.string)
    min_after_dequeue = 10
    capacity = min_after_dequeue+3*batch_size
    image, label = tf.train.batch([image, label],
                                  batch_size=batch_size,
                                  num_threads=3,
                                  capacity=capacity)
    return image, label


def decode_from_tfrecords_eval(filename, batch_size=1):    
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                           'img_name' : tf.FixedLenFeature([],tf.string)
                                       })
    
    image = tf.decode_raw(features['img_raw'],tf.uint8)
    image = tf.reshape(image, [height_patch, width_patch, 3])
    label = tf.cast(features['label'], tf.int64)
    image = tf.cast(image, tf.float32)
    capacity = 10+3*batch_size
    image, label = tf.train.batch([image, label],
                                  batch_size=batch_size, 
                                  num_threads=3, 
                                  capacity=capacity)
    return image, label
