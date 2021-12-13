import cv2
import skimage.filters
import numpy as np
from skimage import color

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def segmentImage(image):

    # maybe need some preprocessing to rotate the image to be oriented horizontally... 
    image=image[2:image.shape[0] - 2, 2:image.shape[1] - 2]
    #grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale = color.rgb2gray(image)




    (thresh, binarized) = cv2.threshold(grayscale, 0.5, 0.9, cv2.THRESH_BINARY)
    
    #new_path = "binarized.png"
    #cv2.imwrite(new_path,image)

    blurred = skimage.filters.gaussian(binarized)
    #frame_normed = 255 * (blurred - blurred.min()) / (blurred.max() - blurred.min())
    #frame_normed = np.array(frame_normed, np.int)

    #cv2.imwrite("savedImage.png", frame_normed)
    # find histograms of horizontal lines, and if above certain threshold, determine upper/lower bounds of characters
    vertical_threshold = 0.5
    row_histograms = np.average(blurred,axis=1)
    top_border, bottom_border = np.nonzero(row_histograms > vertical_threshold)[0][[0,-1]]
    top_border -= 2
    if (top_border < 0):
        top_border = 0
    bottom_border += 2
    # find histograms of vertical lines, if above threshold you find regions separating individual characters. 
    
    horizontal_threshold = 0.45
    column_histograms = np.average(blurred, axis=0)
    
    vertical_boundaries = np.nonzero(column_histograms > horizontal_threshold)[0]
    # separate regions into individual sections and return. 
    subImages = []
    for i in range(len(vertical_boundaries) - 1):
        if (vertical_boundaries[i+1] - vertical_boundaries[i]) <= 5:
            continue
        print(vertical_boundaries[i], vertical_boundaries[i+1], top_border, bottom_border)
        subImage = blurred[:, vertical_boundaries[i]: vertical_boundaries[i+1]]
        subImages.append(subImage)


    #cur_img = subImages[3]
    #frame_normed = 255 * (cur_img - cur_img.min()) / (cur_img.max() - cur_img.min())
    #frame_normed = np.array(frame_normed, np.int)

    #cv2.imwrite("savedImage.png", frame_normed)

    return subImages





def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def detect(img_rgb):

    img = img_rgb.copy()
    input_height = img_rgb.shape[0]
    input_width = img_rgb.shape[1]
    hsv_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

    # yellow color
    low_yellow = np.array([20, 100, 100])
    high_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
    yellow = cv2.bitwise_and(yellow_mask, yellow_mask, mask=yellow_mask)

    cv2.imwrite("temp/steps/1_yellow_color_detection.png", yellow)
    # Close morph
    k = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, k)

    cv2.imwrite("temp/steps/2_closing_morphology.png", closing)
    # Detect yellow area
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # List of final crops
    crops = []
    print(1)
    # Loop over contours and find license plates
    for cnt in contours:
        print(2)
        x, y, w, h = cv2.boundingRect(cnt)

        # Conditions on crops dimensions and area
        if h*6 > w > 2 * h and h > 0.1 * w and w * h > input_height * input_width * 0.0001:

            # Make a crop from the RGB image, the crop is slided a bit at left to detect bleu area
            crop_img = img_rgb[y:y + h, x-round(w/10):x]
            crop_img = crop_img.astype('uint8')

            # Compute bleu color density at the left of the crop
            # Bleu color condition
            try:
                hsv_frame = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                low_bleu = np.array([100,150,0])
                high_bleu = np.array([140,255,255])
                bleu_mask = cv2.inRange(hsv_frame, low_bleu, high_bleu)
                bleu_summation = bleu_mask.sum()

            except:
                bleu_summation = 0

            # Condition on bleu color density at the left of the crop
            if bleu_summation > 550:

                # Compute yellow color density in the crop
                # Make a crop from the RGB image
                imgray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                crop_img_yellow = img_rgb[y:y + h, x:x+w]
                crop_img_yellow = crop_img_yellow.astype('uint8')

                # Detect yellow color
                hsv_frame = cv2.cvtColor(crop_img_yellow, cv2.COLOR_BGR2HSV)
                low_yellow = np.array([20, 100, 100])
                high_yellow = np.array([30, 255, 255])
                yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)

                # Compute yellow density
                yellow_summation = yellow_mask.sum()

                # Condition on yellow color density in the crop
                if yellow_summation > 255*crop_img.shape[0]*crop_img.shape[0]*0.4:

                    # Make a crop from the gray image
                    crop_gray = imgray[y:y + h, x:x + w]
                    crop_gray = crop_gray.astype('uint8')

                    # Detect chars inside yellow crop with specefic dimension and area
                    th = cv2.adaptiveThreshold(crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    contours2, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # Init number of chars
                    chars = 0
                    for c in contours2:
                        area2 = cv2.contourArea(c)
                        x2, y2, w2, h2 = cv2.boundingRect(c)
                        if w2 * h2 > h * w * 0.01 and h2 > w2 and area2 < h * w * 0.9:
                            chars += 1

                    # Condition on the number of chars
                    if 20 > chars > 4:
                        rect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        pts = np.array(box)
                        warped = four_point_transform(img, pts)
                        crops.append(warped)

                        # Using cv2.putText() method
                        img_rgb = cv2.putText(img_rgb, 'LP', (x, y), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)

                        cv2.drawContours(img_rgb, [box], 0, (0, 0, 255), 2)

    return crops
