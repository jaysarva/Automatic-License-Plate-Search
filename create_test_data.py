import os
import shutil
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv

def filelist(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name, 
            files in os.walk(root) for f in files if f.endswith(file_type)]

def get_test_data_info_csv(annotations_paths):
    data = []
    for path in annotations_paths:
        with open(path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                if row[3] == 'license-plate':
                    r = []
                    for item in row:
                        r.append(item)
                    data.append(r)
    return data

def read_img(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

def create_mask(bounding_box, img):
    height, width, _ = img.shape
    mask = np.zeros((height, width))
    mask[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]] = 1
    return mask

def mask_to_boundingbox(mask):
    cols, rows = np.nonzero(mask)
    if len(cols) == 0:
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row])

def create_boundingbox_array_csv(data_row):
    return np.array([int(data_row[5]), int(data_row[4]), int(data_row[7]), int(data_row[6])])

def resize_image_boundingbox_csv(read_path, write_path, bounding_box, size, index):
    img = read_img(read_path)
    resize_img = cv2.resize(img, (size, size))
    resized_mask = cv2.resize(create_mask(bounding_box, img), (size, size))
    new_path = str(write_path) + '/Cars' + str(index) + '.png'
    # new_path = str(write_path/read_path.parts[-1])
    cv2.imwrite(new_path, cv2.cvtColor(resize_img, cv2.COLOR_RGB2BGR))
    return mask_to_boundingbox(resized_mask)

def find_csv_read_path(image_file, full_paths):
    for path in full_paths:
        if image_file in path:
            return path
    return None

def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color,
                         fill=False, lw=3)

def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))
    plt.show()


###################
## EXPORT RESULT ##
###################

def create_test_data():
    path = "test_data/preprocessed_data"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)

    resized_data_path = Path(path)
    csv_images = filelist(Path("raw_data/dataset2"), '.jpg') + filelist(Path("raw_data/dataset4"), '.jpg')
    csv_annotations_paths = [Path("raw_data/dataset2/annotations.csv"), Path("raw_data/dataset4/annotations.csv")]
    
    csv_test_data = np.array(get_test_data_info_csv(csv_annotations_paths))

    index = 0
    modified_csv_test_data = []
    for img_info in csv_test_data:
        read_path = find_csv_read_path(img_info[0], csv_images)
        if read_path != None:
            new_boundingbox = resize_image_boundingbox_csv(Path(read_path), resized_data_path, create_boundingbox_array_csv(img_info), 224, index)
            index += 1
            new_row = list(img_info) + [str(x) for x in new_boundingbox]
            modified_csv_test_data.append(new_row)
    csv_test_data = np.array(modified_csv_test_data)
    num_img = csv_test_data.shape[0]
    csv_test_data = np.hstack((np.arange(num_img).reshape((num_img, 1)), csv_test_data))
    output2 = np.delete(csv_test_data, 1, 1)
    output2 = np.delete(output2, 3, 1)
    output2 = output2.astype(np.int)
    output2 = np.hstack((np.reshape(output2[:, 0], (output2.shape[0], 1)), output2[:, 7:]))

    # total_output = np.vstack((output, output2))
    total_output = output2
    np.savetxt('test_data/boundingbox.csv', total_output, delimiter=',', fmt='%d')

create_test_data()