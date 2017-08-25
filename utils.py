import tensorflow as tf
import numpy as np
import scipy.misc
import os, sys, glob, argparse

def image_list(image_dir):
    if not os.path.exists(image_dir):
        print('Image directory %s not exists' % image_dir)
        return None
    file_type_extended = ['jpg','jpeg','png']
    file_list = list()
    for _path, _dir, _files in os.walk(image_dir):
        for f in _files:
            if f.split('.')[-1] in file_type_extended:
                file_list.append(os.path.join(os.path.abspath(_path), f))
    if len(file_list) == 0:
        print('No image files')
        return None
    # This is FIFO Queue and enqueue to capacity when reader dequeue an element
    # It needs a time to fill the queue
    file_list_queue = tf.train.string_input_producer(file_list, shuffle=True)
    return file_list_queue

def image_preprocess(file_queue, input_size, target_size):
    image_reader = tf.WholeFileReader()
    # Filereader read operation consumes filenames in filequeue and pass it to the decoder
    # Referenced by : https://www.tensorflow.org/programmers_guide/reading_data
    key, value = image_reader.read(file_queue)
    uint8_image = tf.image.decode_jpeg(value, channels=3)
    # Only 3-dim possible
    cropped_image = tf.cast(tf.image.crop_to_bounding_box(uint8_image, offset_height=50, offset_width=35, target_height=input_size, target_width=input_size), tf.float32)
    # Now expanding dimension including batch dimension
    cropped_image_4d = tf.expand_dims(cropped_image, 0)
    resized_image = tf.image.resize_bilinear(cropped_image_4d, size=[target_size, target_size])
    input_image = tf.squeeze(resized_image, 0)
    return input_image

def read_image(file_queue, args):
    input_img = image_preprocess(file_queue, args.input_size, args.target_size)
    num_preprocess_threads = 4
    min_queue_example = int(0.1*args.num_examples_per_epoch)
    # If number fo threads 1, batch will not be shuffled
    input_imgs = tf.train.batch([input_img], batch_size=args.batch_size, num_threads=num_preprocess_threads, capacity=min_queue_example+2*args.batch_size) 
    # Normalize between -1 and 1
    input_imgs = input_imgs/127.5 - 1
    return input_imgs

def save_image(images, size, save_path):
    print('Save images')
    height, width = images.shape[1], images.shape[2]
    merged_image = np.zeros([size[0]*height, size[1]*width, images.shape[3]])
    for index, img in enumerate(images):
        j = index % size[1]
        i = index // size[1]
        merged_image[i*height:i*height+height, j*width:j*width+width, :] = img
    merged_image += 1
    merged_image *= 127.5
    merged_image = np.clip(merged_image, 0, 255).astype(np.uint8)
    scipy.misc.imsave(save_path, merged_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=108)
    parser.add_argument('--target_size', type=int, default=64)
    parser.add_argument('--num_examples_per_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img_lists = image_list('../CelebA', 10000)
        imgs = read_image(img_lists, args)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(10):
            print(sess.run(img_lists.size()))
        coord.request_stop()
        coord.join(threads)




    
    
