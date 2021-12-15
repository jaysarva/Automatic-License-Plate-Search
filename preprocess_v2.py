import os
import shutil
import xml.etree.ElementTree as ET
import numpy as np
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
    # OUTPUT: 1D NumPy array with size = 4 containing bounding box's location [y_min x_min y_max x_max]
    rows, cols = np.nonzero(mask)
    # If bounding box does not exist, return 1D NumPy array of 4 zeros
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros(4)
    return np.array([np.min(rows), np.min(cols), np.max(rows), np.max(cols)])

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
    # OUTPUT: 1D NumPy array with size = 4 containing relocated bounding box's location [y_min x_min y_max x_max]
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

# Convert the bounding box's location from ratio of height/width into integers
def boundingbox_location_float_to_int(img, img_info):
    # INPUT: 
    #   img (3D NumPy array of image data), 
    #   img_info (1D NumPy array containing single image's info [row_num x_min y_min x_max y_max]) in float values
    # OUTPUT: 1D NumPy array containing single image's info [row_num x_min y_min x_max y_max] in integer values
    height, width, _ = img.shape
    new_img_info = [0, 0, 0, 0, 0]
    new_img_info[0] = int(img_info[0])
    new_img_info[1] = round(float(img_info[1]) * width)
    new_img_info[2] = round(float(img_info[2]) * height)
    new_img_info[3] = round(float(img_info[3]) * width)
    new_img_info[4] = round(float(img_info[4]) * height)
    return new_img_info

# Find whether the image annotated with .csv file exists
def find_csv_read_path(image_file, full_paths):
    for path in full_paths:
        if image_file in path:
            return True
    return False

########################
## DATA VISUALIZATION ##
########################

# Visualizer
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

def rotate_image(img, deg, isMask, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    height, width, _ = img.shape
    rotationMatrix = cv2.getRotationMatrix2D((height/2, width/2), deg, 1)
    if isMask:
        return cv2.warpAffine(img, rotationMatrix, (height, width), borderMode=cv2.BORDER_CONSTANT)
    else:
        return cv2.warpAffine(img, rotationMatrix, (height, width), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def transformsXY(path, bounding_box):
    img = cv2.imread(str(path)).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
    mask = boundingbox_to_mask(bounding_box, img)
    rotation_deg = (np.random.random() - 0.5) * 20
    img = rotate_image(img, rotation_deg, False)
    mask = rotate_image(mask, rotation_deg, True)
    if np.random.random() > 0.5: 
        img = np.fliplr(img).copy()
        mask = np.fliplr(mask).copy()
    return img, mask_to_boundingbox(mask)


###################
## MAIN FUNCTION ##
###################

def preprocessing(output_image_size=224, training_size=0.85):
    print("Begin preprocessing ...")

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
    
    csv_images_info = get_image_info_csv(csv_annotations_read_path)
    csv_images_info = np.delete(csv_images_info, 1, 1)
    csv_images_filelist = filelist(csv_images_read_path, ".jpg")
    csv_cutoff_index = int(training_size * len(csv_images_info))
    xml_images_info = get_image_info_xml(xml_annotations_read_path)
    xml_cutoff_index = int(training_size * len(xml_images_info))

    # Training Data
    csv_train_images_info = csv_images_info[:csv_cutoff_index]
    index = 0
    train_boundingboxes_info = []
    for img_info in csv_train_images_info:
        isExist = find_csv_read_path("car" + str(img_info[0]) + ".jpg", csv_images_filelist)
        if isExist:
            read_path = csv_images_read_path + "/car" + str(img_info[0]) + ".jpg"
            img = read_image(read_path)
            new_img_info = boundingbox_location_float_to_int(img, img_info)
            bounding_box = create_boundingbox_array_csv(new_img_info)
            relocated_boundingbox = preprocess_image(img, train_images_write_path, bounding_box, output_image_size, index)
            boundingbox_info = [index] + list(relocated_boundingbox)
            train_boundingboxes_info.append(boundingbox_info)
            index += 1
    xml_train_images_info = xml_images_info[:xml_cutoff_index]
    for img_info in xml_train_images_info:
        read_path = xml_images_read_path + "/Cars" + str(img_info[0]) + ".png"
        img = read_image(read_path)
        bounding_box = create_boundingbox_array_xml(img_info)
        relocated_boundingbox = preprocess_image(img, train_images_write_path, bounding_box, output_image_size, index)
        boundingbox_info = [index] + list(relocated_boundingbox)
        train_boundingboxes_info.append(boundingbox_info)
        index += 1
    np.savetxt(train_boundingboxes_write_path, train_boundingboxes_info, delimiter=',', fmt='%d')
    
    # Testing Data
    csv_test_images_info = csv_images_info[csv_cutoff_index:]
    index = 0
    test_boundingboxes_info = []
    for img_info in csv_test_images_info:
        isExist = find_csv_read_path("car" + str(img_info[0]) + ".jpg", csv_images_filelist)
        if isExist:
            read_path = csv_images_read_path + "/car" + str(img_info[0]) + ".jpg"
            img = read_image(read_path)
            new_img_info = boundingbox_location_float_to_int(img, img_info)
            bounding_box = create_boundingbox_array_csv(new_img_info)
            relocated_boundingbox = preprocess_image(img, test_images_write_path, bounding_box, output_image_size, index)
            boundingbox_info = [index] + list(relocated_boundingbox)
            test_boundingboxes_info.append(boundingbox_info)
            index += 1
    xml_test_images_info = xml_images_info[xml_cutoff_index:]
    for img_info in xml_test_images_info:
        read_path = xml_images_read_path + "/Cars" + str(img_info[0]) + ".png"
        img = read_image(read_path)
        bounding_box = create_boundingbox_array_xml(img_info)
        relocated_boundingbox = preprocess_image(img, test_images_write_path, bounding_box, output_image_size, index)
        boundingbox_info = [index] + list(relocated_boundingbox)
        test_boundingboxes_info.append(boundingbox_info)
        index += 1
    np.savetxt(test_boundingboxes_write_path, test_boundingboxes_info, delimiter=',', fmt='%d')

    print("Preprocessing Done!")

if __name__ == "__main__":
    preprocessing()
    
    