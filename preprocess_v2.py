import os
import shutil
import xml.etree.ElementTree as ET
# from pathlib import Path
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import csv

######################
## HELPER FUNCTIONS ##
######################

# Get files's name that match the given the file type containing in the given root directory
def filelist(root, f_type):
    # INPUT: 
    #   root (root directory), 
    #   f_type (type of file such as '.png', '.jpg', or '.xml')
    # OUTPUT: list of files' name under the root directory
    return [os.path.join(d_path, f) for d_path, d_name, files in os.walk(root) for f in files if f.endswith(f_type)]

# Retrieve images' info (file name, width, height, bounding box's location, and index) sorted by filename from .xml files
def get_image_info_xml(annotations_path):
    # INPUT: annotations_path (path to the directory that contains .xml annotation files)
    # OUTPUT: 2D NumPy array containing images' info extracted from .xml files
    annotations = filelist(annotations_path, '.xml')
    annotations_list = []
    for path in annotations:
        root = ET.parse(path).getroot()
        filename = int((root.find("./filename").text)[4:-4])
        width = int(root.find("./size/width").text)
        height = int(root.find("./size/height").text)
        x_min = int(root.find("./object/bndbox/xmin").text)
        y_min = int(root.find("./object/bndbox/ymin").text)
        x_max = int(root.find("./object/bndbox/xmax").text)
        y_max = int(root.find("./object/bndbox/ymax").text)
        annotation = [filename, width, height, x_min, y_min, x_max, y_max]
        annotations_list.append(annotation)
    annotations_list = np.array(annotations_list)
    sorted = annotations_list[np.argsort(annotations_list[:, 0])]
    sorted = np.hstack((sorted, np.arange(sorted.shape[0]).reshape(sorted.shape[0], 1)))
    return sorted

# Retrieve images' info (row number, image name, and bounding box's location) from .csv file
def get_image_info_csv(annotations_path):
    # INPUT: annotations_path (path to .csv annotation file)
    # OUTPUT: 2D NumPy array containing images' info extracted from .csv file
    data = []
    with open(annotations_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        # Skip the first row (which contains columns' names)
        next(reader)
        for row in reader:
            # r = []
            # for item in row:
            # r.append(item)
            data.append(row)
    return np.array(data)

# Read an image from the given file's location and convert into RGB values
def read_image(path):
    # INPUT: path (path to image file's location)
    # OUTPUT: 3D NumPy array of image data in RGB values
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Write an image to the given path and convert back into BGR values
def write_image(path, img):
    # INPUT: 
    #   path (path to image file's location), 
    #   img (3D NumPy array of image data in RGB values)
    # OUTPUT: -
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

# Create a mask from the given bounding box array of the same shape as image
def boundingbox_to_mask(bounding_box, img):
    # INPUT: 
    #   bounding_box (1D NumPy array with size = 4 containing bounding box's location [x_min y_min x_max y_max]),
    #   img (3D NumPy array of image data)
    # OUTPUT: 2D NumPy array containing 1's at the corresponding location of bounding box and 0's otherwise
    mask = np.zeros((img.shape[0], img.shape[1]))
    mask[bounding_box[1]:bounding_box[3]+1, bounding_box[0]:bounding_box[2]+1] = 1
    return mask

# Convert a mask to a bounding box array
def mask_to_boundingbox(mask):
    # INPUT: mask (2D NumPy array containing 1's if it is a bounding box's location, otherwise 0's)
    # OUTPUT: 1D NumPy array with size = 4 containing bounding box's location [x_min y_min x_max y_max]
    rows, cols = np.nonzero(mask)
    # If bounding box does not exist, return 1D NumPy array of 4 zeros
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros(4)
    return np.array([np.min(cols), np.min(rows), np.max(cols), np.max(rows)])

# Create a bounding box array from each image's info in 2D NumPy array images' info extracted from .xml files
def create_boundingbox_array_xml(row):
    # INPUT: row (1D NumPy array containing single image's info [filename width height x_min y_min x_max y_max index])
    # OUTPUT: 1D NumPy array with size = 4 containing bounding box's location [x_min y_min x_max y_max]
    return row[3:7]

# Create a bounding box array from each image's info in 2D NumPy array images' info extracted from .csv file(s)
def create_boundingbox_array_csv(row):
    # INPUT: row (1D NumPy array containing single image's info [row_num x_min y_min x_max y_max])
    # OUTPUT: 1D NumPy array with size = 4 containing bounding box's location [x_min y_min x_max y_max]
    return row[1:]

# Preprocess an image
def preprocess_image(img, write_directory, bounding_box, size, index):
    # INPUT: 
    #   img (3D NumPy array of image data), 
    #   write_directory (path to write a preprocessed image), 
    #   bounding_box (1D NumPy array with size = 4 containing bounding box's location [x_min y_min x_max y_max]), 
    #   size (size of preprocessed square image), 
    #   index (image's index that helps writing the image)
    # OUTPUT: 1D NumPy array with size = 4 containing relocated bounding box's location [x_min y_min x_max y_max]
    # Resize the image to be a square image with the given size
    resize_img = cv2.resize(img, (size, size))
    # Relocate the bounding box's location corresponding to the image resizing
    resized_mask = cv2.resize(boundingbox_to_mask(bounding_box, img), (size, size))
    relocated_boundingbox = mask_to_boundingbox(resized_mask)
    # Write the resized image to the given path
    write_path = write_directory + "/Cars" + str(index) + ".png"
    write_image(write_path, resize_img)
    return relocated_boundingbox

# Delete the directory if it is exist; otherwise, create a blank directory at the given path
def create_directory(path):
    # INPUT: path (path to directory)
    # OUTPUT: -
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def boundingbox_location_float_to_int(img, img_info):
    height, width, _ = img.shape
    new_img_info = [0, 0, 0, 0, 0]
    new_img_info[0] = int(img_info[0])
    new_img_info[1] = round(float(img_info[1]) * width)
    new_img_info[2] = round(float(img_info[2]) * height)
    new_img_info[3] = round(float(img_info[3]) * width)
    new_img_info[4] = round(float(img_info[4]) * height)
    return new_img_info

def find_csv_read_path(image_file, full_paths):
    for path in full_paths:
        if image_file in path:
            return True
    return False

def show_corner_bb(im, bb):
    def create_corner_rect(bb, color="red"):
        bb = np.array(bb, dtype=np.float32)
        return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color, fill=False, lw=4)
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))
    plt.show()

#######################
## DATA AUGMENTATION ##
#######################

def crop(img, r, c, target_r, target_c): 
    return img[r:r+target_r, c:c+target_c]

def random_crop(x, r_pix=8):
    r, c, _ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

def center_crop(x, r_pix=8):
    r, c, _ = x.shape
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)

def rotate_cv(im, deg, y=False, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
    if y:
        return cv2.warpAffine(im, M,(c,r), borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def random_cropXY(x, Y, r_pix=8):
    r, c, _ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    xx = crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)
    YY = crop(Y, start_r, start_c, r-2*r_pix, c-2*c_pix)
    return xx, YY

def transformsXY(path, bb, transforms):
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    Y = boundingbox_to_mask(bb, x)
    if transforms:
        rdeg = (np.random.random()-.50)*20
        x = rotate_cv(x, rdeg)
        Y = rotate_cv(Y, rdeg, y=True)
        if np.random.random() > 0.5: 
            x = np.fliplr(x).copy()
            Y = np.fliplr(Y).copy()
        x, Y = random_cropXY(x, Y)
    else:
        x, Y = center_crop(x), center_crop(Y)
    return x, mask_to_boundingbox(Y)


###################
## EXPORT RESULT ##
###################

# sample_img_num = 400
# read_path = Path(str(path) + '/Cars' + str(train_data[sample_img_num][0]) + '.png')
# im = cv2.imread(str(read_path))
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# show_corner_bb(im, train_data[sample_img_num][7:])

# im, bb = transformsXY(str(read_path), train_data[sample_img_num][7:], True)
# show_corner_bb(im, bb)

# sample_img_num = 200
# read_path = Path(find_csv_read_path(csv_train_data[sample_img_num][0], csv_images))
# im = cv2.imread(str(read_path))
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# show_corner_bb(im, csv_train_data[sample_img_num][9:])

# im, bb = transformsXY(str(read_path), csv_train_data[sample_img_num][9:], True)
# show_corner_bb(im, bb)

def preprocessing():
    size = 224

    # Input paths
    xml_images_read_path = "raw_data/dataset1/images"
    xml_annotations_read_path = "raw_data/dataset1/annotations"
    csv_images_read_path = "raw_data/dataset5/Cars"
    csv_annotations_read_path = "raw_data/dataset5/license.csv"

    # Output paths
    train_images_directory_path = "train_data"
    train_images_write_path = train_images_directory_path + "/data"
    create_directory(train_images_write_path)
    train_boundingboxes_write_path = train_images_directory_path + "/boundingbox.csv"
    test_images_directory_path = "test_data"
    test_images_write_path = test_images_directory_path + "/data"
    create_directory(test_images_write_path)
    test_boundingboxes_write_path = test_images_directory_path + "/boundingbox.csv"
    
    # Training Data
    csv_images_info = get_image_info_csv(csv_annotations_read_path)
    csv_images_info = np.delete(csv_images_info, 1, 1)
    index = 0
    train_boundingboxes_info = []
    csv_images_filelist = filelist(csv_images_read_path, ".jpg")
    for img_info in csv_images_info:
        isExist = find_csv_read_path("car" + str(img_info[0]) + ".jpg", csv_images_filelist)
        if isExist:
            read_path = csv_images_read_path + "/car" + str(img_info[0]) + ".jpg"
            img = read_image(read_path)
            new_img_info = boundingbox_location_float_to_int(img, img_info)
            bounding_box = create_boundingbox_array_csv(new_img_info)
            relocated_boundingbox = preprocess_image(img, train_images_write_path, bounding_box, size, index)
            boundingbox_info = [index] + list(relocated_boundingbox)
            train_boundingboxes_info.append(boundingbox_info)
            index += 1
    xml_images_info = get_image_info_xml(xml_annotations_read_path)[:100]
    for img_info in xml_images_info:
        read_path = xml_images_read_path + "/Cars" + str(img_info[0]) + ".png"
        img = read_image(read_path)
        bounding_box = create_boundingbox_array_xml(img_info)
        relocated_boundingbox = preprocess_image(img, train_images_write_path, bounding_box, size, index)
        boundingbox_info = [index] + list(relocated_boundingbox)
        train_boundingboxes_info.append(boundingbox_info)
        index += 1
    np.savetxt(train_boundingboxes_write_path, train_boundingboxes_info, delimiter=',', fmt='%d')
    
    # Testing Data

    

    # csv_images = filelist(Path("raw_data/dataset2"), '.jpg') + filelist(Path("raw_data/dataset3"), '.jpg') + filelist(Path("raw_data/dataset4"), '.jpg')
    # csv_annotations_paths = [Path("raw_data/dataset2/annotations.csv"), Path("raw_data/dataset3/annotations.csv"), Path("raw_data/dataset4/annotations.csv")]
    csv_images = filelist(Path("raw_data/dataset2"), '.jpg') + filelist(Path("raw_data/dataset4"), '.jpg')
    csv_annotations_paths = [Path("raw_data/dataset2/annotations.csv"), Path("raw_data/dataset4/annotations.csv")]
    
    csv_train_data = np.array(get_train_data_info_csv(csv_annotations_paths))

    # new_boundingboxes = []
    index = train_data.shape[0]
    modified_csv_train_data = []
    for img_info in csv_train_data:
        read_path = find_csv_read_path(img_info[0], csv_images)
        if read_path != None:
            # new_boundingbox = resize_image_boundingbox_csv(Path(read_path), resized_data_path, create_boundingbox_array_csv(img_info), 224, index)
            index += 1
            # new_boundingboxes.append(new_boundingbox)
            new_row = list(img_info) + [str(x) for x in new_boundingbox]
            modified_csv_train_data.append(new_row)
    # csv_train_data = np.hstack((csv_train_data, new_boundingboxes))
    csv_train_data = np.array(modified_csv_train_data)
    index_array = np.arange(train_data.shape[0], index).reshape(index - train_data.shape[0], 1)
    csv_train_data = np.hstack((index_array, csv_train_data))
    output = np.hstack((np.reshape(train_data[:, 7], (train_data.shape[0], 1)), train_data[:, 8:]))

    output2 = np.delete(csv_train_data, 1, 1)
    output2 = np.delete(output2, 3, 1)
    output2 = output2.astype(np.int)
    output2 = np.hstack((np.reshape(output2[:, 0], (output2.shape[0], 1)), output2[:, 7:]))

    # total_output = np.vstack((output, output2))
    total_output = output
    np.savetxt('boundingbox.csv', total_output, delimiter=',', fmt='%d')

if __name__ == "__main__":
    # preprocessing()
    # plt.imshow(im)
    # plt.show()
    # print(np.zeros(4))
    # print(np.zeros(4, dtype=np.float32))
    # bounding_box = [1, 2, 3, 5]
    # bounding_box = np.array(bounding_box)
    # mask = np.zeros((6, 6))
    # mask[bounding_box[1]:bounding_box[3]+1, bounding_box[0]:bounding_box[2]+1] = 1
    # print(mask)
    # rows, cols = np.nonzero(mask)
    # print("rows:", rows)
    # print("cols:", cols)
    # bb = np.array([np.min(cols), np.min(rows), np.max(cols), np.max(rows)])
    # print(bb)
    print("Begin the preprocessing process ...")
    preprocessing()
    