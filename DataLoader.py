import struct
import logging

def read_images(images_file):
    # returns an array of flattened images

    # Description of image files according to MNIST database:
    # [offset] [type]          [value]          [description]
    # 0000     32 bit integer  0x00000803(2051) magic number
    # 0004     32 bit integer  60000            number of images
    # 0008     32 bit integer  28               number of rows
    # 0012     32 bit integer  28               number of columns
    # 0016     unsigned byte   ??               pixel
    # 0017     unsigned byte   ??               pixel
    # ........
    # xxxx     unsigned byte   ??               pixel
    # Pixels are organized row-wise. Pixel values are 0 to 255. 
    # 0 means background (white), 255 means foreground (black).

    f = open(images_file, 'rb')
    images = []

    # decoding the header of the file -- these are all 32 bit integers coded in these bytes
    magicWord_32bit = f.read(4) # read the first 4 bytes (8*4 = 32 bits) -- the magic word
    n_images_32bit = f.read(4)  # number of images
    n_rows_32bit = f.read(4)    # number of rows in each image
    n_cols_32bit = f.read(4)    # number of columns in each image

    # convert to integers (subscript [0] because struct.unpack always returns a tuple)
    magicWord = struct.unpack('>i', magicWord_32bit)[0]
    n_images = struct.unpack('>i', n_images_32bit)[0]
    n_rows = struct.unpack('>i', n_rows_32bit)[0]
    n_cols = struct.unpack('>i', n_cols_32bit)[0]

    try:
        for i in range(n_images):
            image = []
            for r in range(n_rows):
                for c in range(n_cols):
                    byte = f.read(1)
                    pixel = struct.unpack('B', byte)[0]
                    image.append(pixel)
            images.append(image)
    finally:
        f.close()
        return images

def read_labels(labels_file):
    # returns an array of labels, following same ideas as read_images()
    f = open(labels_file, 'rb')
    labels = []

    magicWord_32bit = f.read(4)
    n_labels_32bit = f.read(4)

    magicWord = struct.unpack('>i', magicWord_32bit)[0]
    n_labels = struct.unpack('>i', n_labels_32bit)[0]

    try:
        for l in range(n_labels):
            byte = f.read(1)
            label = struct.unpack('B', byte)[0]
            labels.append(label)
    finally:
        f.close()
        return labels

def read_dataset(images_file, labels_file, verbose=False):
    # read files containing images and labels and return an array of tuples of (flattened_image, label)
    images = read_images(images_file)
    labels = read_labels(labels_file)
    # logging (experimental, and not necessary)
    logging.info("Number of images: ", len(images), "\nNumber of labels: ", len(labels))
    if verbose:
        print("Number of images: ", len(images), "\nNumber of labels: ", len(labels), "\nDon't forget to convert the return data to a list before indexing into it!")
    assert len(images) == len(labels)
    return zip(images, labels)
