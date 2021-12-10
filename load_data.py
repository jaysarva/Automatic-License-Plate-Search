import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import csv

def filelist(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name, 
            files in os.walk(root) for f in files if f.endswith(file_type)]

def get_train_data_info_xml(annotations_path):
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

def get_train_data_info_csv(annotations_paths):
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

def create_boundingbox_array(data_row):
    # [y_min y_max, x_min, x_max]
    return np.array([data_row[4], data_row[3], data_row[6], data_row[5]])

def create_boundingbox_array_csv(data_row):
    return np.array([int(data_row[5]), int(data_row[4]), int(data_row[7]), int(data_row[6])])

def resize_image_boundingbox(read_path, write_path, bounding_box, size, ind):
    img = read_img(read_path)
    resize_img = cv2.resize(img, (size, size))
    resized_mask = cv2.resize(create_mask(bounding_box, img), (size, size))
    # new_path = str(write_path/read_path.parts[-1])
    new_path = str(write_path) + '/Cars' + str(ind) + '.png'
    # cv2.imwrite(new_path, cv2.cvtColor(resize_img, cv2.COLOR_RGB2GRAY))
    cv2.imwrite(new_path, cv2.cvtColor(resize_img, cv2.COLOR_RGB2BGR))
    return mask_to_boundingbox(resized_mask)

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

#######################
## DATA AUGMENTATION ##
#######################

def crop(im, r, c, target_r, target_c): 
    return im[r:r+target_r, c:c+target_c]

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
    Y = create_mask(bb, x)
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
    images_path = Path("raw_data/dataset1/images")
    annotations_path = Path("raw_data/dataset1/annotations")
    
    train_data = get_train_data_info_xml(annotations_path)
    new_boundingboxes = []
    path = "preprocessed_data/resized_images"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)

    resized_data_path = Path(path)
    ind = 0
    for img_info in train_data:
        read_path = Path(str(images_path) + '/Cars' + str(img_info[0]) + '.png')
        new_boundingbox = resize_image_boundingbox(read_path, resized_data_path, create_boundingbox_array(img_info), 224, ind)
        ind += 1
        new_boundingboxes.append(new_boundingbox)
    train_data = np.hstack((train_data, new_boundingboxes))

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
            new_boundingbox = resize_image_boundingbox_csv(Path(read_path), resized_data_path, create_boundingbox_array_csv(img_info), 224, index)
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

    total_output = np.vstack((output, output2))
    np.savetxt('boundingbox.csv', total_output, delimiter=',', fmt='%d')

preprocessing()


