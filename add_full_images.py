import os
import shutil
from PIL import Image
import pandas as pd
from tqdm import tqdm

from tools import find_crop_box
import cv2

# 医院直接导出的文件夹
Thyroid_path = 'D:/RIO/All_Datastes/甲状腺超声数据集'
Mammary_path = 'D:/RIO/All_Datastes/乳腺/乳腺'
Liver_path = 'D:/RIO/All_Datastes/脂肪肝/脂肪肝'

# 整理后的文件架
ThyroidRawCsv = 'D:/RIO/All_Datastes/整理好的超声数据集/Ultrasonic_datasets/Throid_dataset/Thyroid_rawXML.csv'
MammaryRawCsv = 'D:/RIO/All_Datastes/整理好的超声数据集/Ultrasonic_datasets/Mammary_dataset/Mammary_rawXML.csv'
LiverRawCsv = 'D:/RIO/All_Datastes/整理好的超声数据集/Ultrasonic_datasets/Liver_dataset/Liver_rawXML.csv'

# 搬到新的文件夹
new_Thyroid_path = 'D:/RIO/All_Datastes/整理好的超声数据集/RawFull_Images/Raw/Thyroid'
new_Mammary_path = 'D:/RIO/All_Datastes/整理好的超声数据集/RawFull_Images/Raw/Mammary'
new_Liver_path = 'D:/RIO/All_Datastes/整理好的超声数据集/RawFull_Images/Raw/Liver'


def crop_save_images(read_path, save_path):
    flag, region = find_crop_box(read_path)
    if flag:
        new_img = region
    if new_img is not None:
        img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        im_pil.save(save_path)


def move_file_and_add_to_csv(csv_file, raw_path, des_path):
    data = pd.read_csv(csv_file)
    # print(data.head(5))
    data = pd.concat([data, pd.DataFrame(columns=['原始图合集路径', '图像数量'])])
    for index, item in tqdm(data.iterrows()):
        uid = str(int(item['uid']))
        name = item['name']
        file = '00' + uid + '(' + name + ')'
        folder = os.path.join(raw_path, file)
        files = os.listdir(folder)
        images = [s for s in files if s.endswith(".jpg")]
        images_str = ",".join(images)

        # move to new folder
        for item in images:
            new_path = os.path.join(des_path, item)
            if os.path.exists(new_path): continue
            source_pth = os.path.join(folder, item)
            if os.path.exists(source_pth):
                shutil.copyfile(source_pth, new_path)

        # save to csv
        data['原始图合集路径'][index] = images_str
        data['图像数量'][index] = len(images)
    data.to_csv(csv_file, encoding="utf_8_sig", index=False)


# move_file_and_add_to_csv(ThyroidRawCsv, Thyroid_path, new_Thyroid_path)
# move_file_and_add_to_csv(MammaryRawCsv, Mammary_path, new_Mammary_path)
# move_file_and_add_to_csv(LiverRawCsv, Liver_path, new_Liver_path)


# 截图，存储新路径
def cut_save(csv_file, raw_path, des_path):
    data = pd.read_csv(csv_file)
    # print(data.head(5))
    data = pd.concat([data, pd.DataFrame(columns=['剪裁图合集路径'])])
    for index, item in tqdm(data.iterrows()):
        image_nums = int(item['图像数量'])
        if image_nums != 0:
            image_list = item['原始图合集路径'].split(',')
            uid = str(int(item['uid']))
            cut_list = []
            for i in range(len(image_list)):
                file = image_list[i]
                file_path = os.path.join(raw_path, file)
                output = os.path.join(des_path, uid+'_'+str(i)+'.jpg')
                # print(output)
                if not os.path.exists(output):
                    crop_save_images(file_path, output)
                cut_list.append(uid+'_'+str(i)+'.jpg')

            cut_str = ",".join(cut_list)
            data['剪裁图合集路径'][index] = cut_str
    data.to_csv(csv_file, encoding="utf_8_sig", index=False)

# 搬到新的文件夹
crop_Thyroid_path = 'D:/RIO/All_Datastes/整理好的超声数据集/RawFull_Images/Crop/Thyroid'
crop_Mammary_path = 'D:/RIO/All_Datastes/整理好的超声数据集/RawFull_Images/Crop/Mammary'
crop_Liver_path = 'D:/RIO/All_Datastes/整理好的超声数据集/RawFull_Images/Crop/Liver'
# cut_save(ThyroidRawCsv, new_Thyroid_path, crop_Thyroid_path)
# cut_save(MammaryRawCsv, new_Mammary_path, crop_Mammary_path)
# cut_save(LiverRawCsv, new_Liver_path, crop_Liver_path)

