import io
import os
import re
import xml.etree.ElementTree as ET
import cv2
import fitz
import numpy as np
from PIL import Image


def read_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    report_dict = {}
    for child in root:
        key = child.find("./key/string").text
        value = child.find("./value/string").text
        if key not in report_dict:
            report_dict.update({key: [value]})
    return report_dict


def MergeTwoDict(d1, d2):
    # d1 = {'a':[0,1], 'b':[1,2], 'd':[1,3]}
    # d2 = {'a':[0], 'c':[1]}
    key_set = set(list(d1.keys()) + list(d2.keys()))
    value = set(d1.keys() ^ d2.keys())  # find the differences between two dicts
    first_key = list(d1.keys())[0]
    len_key = len(d1[first_key])
    for key in value:
        if key not in d1:
            d1.update({key: ['na'] * len_key})

    for key in d1:
        if key in d2:
            d1[key].append(d2[key][0])
        else:
            d1[key].append('NA')
    return d1


def extract_images_from_pdf(file, name, pth):
    pdf_file = fitz.open(file)  # open the file
    page = pdf_file[0]  # only find the first page
    image_list = page.getImageList()
    # printing number of images found in this page
    if image_list:
        # print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
        if len(image_list) != 3:
            print('not enough images in pdf! path is:', file)
    else:
        print("[!] No images found on page", file)

    for image_index, img in enumerate(page.getImageList(), start=1):
        # get the XREF of the image
        xref = img[0]

        # extract the image bytes
        base_image = pdf_file.extractImage(xref)
        image_bytes = base_image["image"]

        # get the image extension
        image_ext = base_image["ext"]

        # load it to PIL
        image = Image.open(io.BytesIO(image_bytes))
        # print(image.size)
        if image.size[0] == 339 and image.size[1] == 346: continue
        # save it to local disk
        image_pth = os.path.join(pth, name)
        image.save(open(f"{image_pth}_{image_index}.{image_ext}", "wb"))


def check_empty_img(pth):
    image = Image.open(pth)
    flag = False
    if image is not None:
        flag = True
    return flag


def resize_donot_change_radio(region, x1, x2, y1, y2):
    # select the longest edge resized to 512
    w, h = x2 - x1, y2 - y1  # h, w = image.shape
    m = max(w, h)
    ratio = 512.0 / m
    new_w, new_h = int(ratio * w), int(ratio * h)
    assert new_w > 0 and new_h > 0
    resized = cv2.resize(region, (new_w, new_h))

    # padded the resized regoin to 512 and 512
    W, H = 512, 512
    top = (H - new_h) // 2
    bottom = (H - new_h) // 2
    if top + bottom + h < H:
        bottom += 1

    left = (W - new_w) // 2
    right = (W - new_w) // 2
    if left + right + w < W:
        right += 1
    pad_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return pad_image


def resize_donot_change_radio2(region):
    # select the longest edge resized to 512
    h, w, _ = region.shape
    m = max(w, h)
    ratio = 512.0 / m
    new_w, new_h = int(ratio * w), int(ratio * h)
    assert new_w > 0 and new_h > 0
    resized = cv2.resize(region, (new_w, new_h))

    # padded the resized regoin to 512 and 512
    W, H = 512, 512
    top = (H - new_h) // 2
    bottom = (H - new_h) // 2
    if top + bottom + h < H:
        bottom += 1

    left = (W - new_w) // 2
    right = (W - new_w) // 2
    if left + right + w < W:
        right += 1
    pad_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    pad_image = cv2.resize(pad_image, (512, 512))
    return pad_image


def detect_red_blue_green(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_red = np.array([-20, 100, 100])
    upper_red = np.array([13, 255, 255])

    # Here we are defining range of bluecolor in HSV
    # This creates a mask of blue coloured
    # objects found in the frame.
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = mask1 | mask2
    contours2, hierarchy2 = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours2)


def find_crop_box(pth):
    flag = False
    final_region = None
    if check_empty_img(pth):
        im = Image.open(pth)
        opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        # bgr->gray->threshold->erode->dilate
        gray = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 15, 255, 0)
        kernel = np.ones((7, 7), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=3)
        thresh = cv2.dilate(thresh, kernel, iterations=3)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_rect = None
        for index in range(len(contours)):
            contour = contours[index]
            area = cv2.contourArea(contour)
            rect = cv2.minAreaRect(contour)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
            if area > max_area:
                max_area = area
                max_rect = rect
        height, width, _ = opencvImage.shape
        if max_rect is None:
            center_area = [[int(height * 1 / 4), int(height * 3 / 4)], [int(width / 4), int(width * 3 / 4)]]
            crop_img = opencvImage[center_area[0][0]:center_area[0][1], center_area[1][0]:center_area[1][1]]
            # print('here')
            region = crop_img
        else:
            box = cv2.boxPoints(max_rect)
            box = np.int0(box)
            sort_box = box[box[:, 1].argsort()]
            # print(sort_box)
            upVertex = sort_box[:2]
            bottomVertex = sort_box[2:]
            upVertex_s = upVertex[upVertex[:, 0].argsort()]
            bottomVertex_s = bottomVertex[bottomVertex[:, 0].argsort()]
            y1, y2 = upVertex_s[0][1], bottomVertex_s[0][1]
            x1, x2 = upVertex_s[0][0], upVertex_s[1][0]
            # if prevent bot function give a negative value
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            region = opencvImage[y1:y2, x1:x2]
            # if sort box is too small

            if abs(x1 - x2) < (width * 0.5) or abs(y1 - y2) < (height * 0.5):
                center_area = [[int(height * 1 / 4), int(height * 3 / 4)], [int(width / 4), int(width * 3 / 4)]]
                crop_img = opencvImage[center_area[0][0]:center_area[0][1], center_area[1][0]:center_area[1][1]]
                # print('here')
                region = crop_img
            # get up two vertex and bottom vertex -> crop images and get the roi region
        # resize_region = resize_donot_change_radio(region, x1, x2, y1, y2)
        resize_region = resize_donot_change_radio2(region)
        # resize_region = cv2.resize(region, (512, 512))
        flag = True
        final_region = resize_region
    else:
        print('PTH is empty!')
    return flag, final_region


def hconcat_resize(img_list,
                   interpolation
                   =cv2.INTER_CUBIC):
    # take minimum hights
    h_min = min(img.shape[0]
                for img in img_list)

    # image resizing
    im_list_resize = [cv2.resize(img,
                                 (int(img.shape[1] * h_min / img.shape[0]),
                                  h_min), interpolation
                                 =interpolation)
                      for img in img_list]

    # return final image
    return cv2.hconcat(im_list_resize)


def clean_finding_Thyroid(fi):
    while fi.find('\n') != -1:
        fi = fi.replace("\n", "。")
    fi = re.sub('[\d.]', '', fi)
    fi = re.sub('、', '', fi)
    fi = re.sub(':', '', fi)
    fi = re.sub('%', '', fi)
    fi = re.sub('％', '', fi)
    fi = re.sub('cm×cm', 'x', fi)
    fi = re.sub('m×cm', 'x', fi)
    fi = re.sub('×cm', '', fi)
    fi = re.sub('cm', 'x', fi)
    fi = re.sub('xxxxx', 'x', fi)
    fi = re.sub('xxx', 'x', fi)
    fi = re.sub('xx', 'x', fi)
    fi = re.sub('ml', '', fi)
    fi = re.sub('mm×mm', '', fi)
    fi = re.sub('mm', '', fi)
    fi = re.sub('×', 'x', fi)
    fi = re.sub('xx', 'x', fi)
    fi = re.sub('>', '', fi)
    fi = re.sub('<', '', fi)
    fi = re.sub('=', '', fi)

    while fi.find('。。') != -1:
        fi = fi.replace("。。", "。")
    while fi.find(" ") != -1:
        fi = fi.replace(" ", "")
    while fi.find("；") != -1:
        fi = fi.replace("；", ",")
    while fi.find("，") != -1:
        fi = fi.replace("，", ",")
    while fi.find("。") != -1:
        fi = fi.replace("。", ",")
    while fi.find("：") != -1:
        fi = fi.replace("：", "")

    fi = re.sub(',,', '', fi)
    fi = fi[:-1] + '。'
    fi = re.sub(',。', '。', fi)

    return fi


def clean_finding_Mammary(fi):
    while fi.find('\n') != -1:
        fi = fi.replace("\n", "。")
    fi = re.sub('[\d.]', '', fi)
    fi = re.sub('、', '', fi)
    fi = re.sub(':', '', fi)
    fi = re.sub('%', '', fi)
    fi = re.sub('％', '', fi)
    fi = re.sub('cm×cm', 'x', fi)
    fi = re.sub('m×cm', 'x', fi)
    fi = re.sub('×cm', '', fi)
    fi = re.sub('cm', 'x', fi)
    fi = re.sub('xxxxx', 'x', fi)
    fi = re.sub('xxx', 'x', fi)
    fi = re.sub('xx', 'x', fi)
    fi = re.sub('ml', '', fi)
    fi = re.sub('mm×mm', '', fi)
    fi = re.sub('mm', '', fi)
    fi = re.sub('×', 'x', fi)
    fi = re.sub('xx', 'x', fi)
    fi = re.sub('>', '', fi)
    fi = re.sub('<', '', fi)
    fi = re.sub('=', '', fi)

    while fi.find('。。') != -1:
        fi = fi.replace("。。", "。")
    while fi.find(" ") != -1:
        fi = fi.replace(" ", "")
    while fi.find("；") != -1:
        fi = fi.replace("；", ",")
    while fi.find("，") != -1:
        fi = fi.replace("，", ",")
    while fi.find("。") != -1:
        fi = fi.replace("。", ",")
    while fi.find("：") != -1:
        fi = fi.replace("：", "")
    fi = re.sub('Vminx/s,', '', fi)
    fi = re.sub('Vmaxx/s,', '', fi)
    fi = re.sub(',,', '', fi)
    fi = fi[:-1] + '。'
    fi = re.sub(',。', '。', fi)

    return fi


def clean_finding_Liver(fi):
    while fi.find('\n') != -1:
        fi = fi.replace("\n", "。")
    fi = re.sub('[\d.]', '', fi)
    fi = re.sub('、', '', fi)
    fi = re.sub(':', '', fi)
    fi = re.sub('%', '', fi)
    fi = re.sub('％', '', fi)
    fi = re.sub('cm×cm', 'x', fi)
    fi = re.sub('m×cm', 'x', fi)
    fi = re.sub('×cm', '', fi)
    fi = re.sub('cm', 'x', fi)
    fi = re.sub('xxxxx', 'x', fi)
    fi = re.sub('xxx', 'x', fi)
    fi = re.sub('xx', 'x', fi)
    fi = re.sub('ml', '', fi)
    fi = re.sub('mm×mm', '', fi)
    fi = re.sub('mm', '', fi)
    fi = re.sub('×', 'x', fi)
    fi = re.sub('xx', 'x', fi)

    while fi.find('。。') != -1:
        fi = fi.replace("。。", "。")
    while fi.find(" ") != -1:
        fi = fi.replace(" ", "")
    while fi.find("；") != -1:
        fi = fi.replace("；", ",")
    while fi.find("，") != -1:
        fi = fi.replace("，", ",")
    while fi.find("。") != -1:
        fi = fi.replace("。", ",")
    while fi.find("：") != -1:
        fi = fi.replace("：", "")
    fi = re.sub('Vminx/s,', '', fi)
    fi = re.sub('Vmaxx/s,', '', fi)
    fi = fi[:-1] + '。'
    fi = re.sub(',。', '。', fi)
    fi = re.sub('xx', 'x', fi)
    fi = re.sub(',。', '。', fi)
    fi = re.sub(',,', ',', fi)

    return fi
