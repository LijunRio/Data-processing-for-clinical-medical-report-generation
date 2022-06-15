import ast
import json
import os
from os.path import exists
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm
from tools import clean_finding_Thyroid, clean_finding_Liver, clean_finding_Mammary
from tools import read_xml, MergeTwoDict, extract_images_from_pdf, find_crop_box


def read_all_xml(root_path):
    folder_list = os.listdir(root_path)
    total_dict = {}

    for item in tqdm(folder_list):
        item_list = item.split('(')
        uid = item_list[0]
        name = item_list[1][:-1]
        xml_file = os.path.join(root_path, item + '\\info.xml')
        report_dict = {'uid': [uid], 'name': [name]}
        report_dict.update(read_xml(xml_file))
        if len(total_dict) == 0:
            total_dict = report_dict
        else:
            total_dict = MergeTwoDict(total_dict, report_dict)
    return total_dict


def save_pdf_images(root_pth, save_pth):
    folder_list = os.listdir(root_pth)
    for item in tqdm(folder_list):
        name = item.split('(')[0][2:]
        pth = os.path.join(root_pth, item)
        pth_list = os.listdir(pth)
        pdf = [s for s in pth_list if '.pdf' in s]
        if len(pdf) == 0: print('error!')
        pdf = pth + '/' + pdf[0]
        extract_images_from_pdf(pdf, name, save_pth)


def crop_save_images(read_path, save_path):
    image_list = os.listdir(read_path)
    for i in tqdm(range(len(image_list))):
        item = image_list[i]
        pth = os.path.join(read_path, item)
        ## directly crop
        # im = Image.open(pth)
        # opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        # height, width, _ = opencvImage.shape
        # center_area = [[int(height * 1 / 4), int(height * 3 / 4)], [int(width / 4), int(width * 3 / 4)]]
        # crop_img = opencvImage[center_area[0][0]:center_area[0][1], center_area[1][0]:center_area[1][1]]
        # pad = resize_donot_change_radio2(crop_img)
        # img_h_resize = hconcat_resize([pad, opencvImage])
        # cv2.imshow('man_image.jpeg', img_h_resize)
        # cv2.waitKey(0)

        ## detect and crop
        flag, region = find_crop_box(pth)
        if flag:
            new_img = region
        else:
            continue
        if new_img is not None:
            ## visualize
            # orim = Image.open(pth)
            # origin_img = cv2.cvtColor(np.array(orim), cv2.COLOR_RGB2BGR)
            # img_h_resize = hconcat_resize([new_img, origin_img])
            # cv2.imshow('man_image.jpeg', img_h_resize)
            # cv2.waitKey(0)
            img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            new_pth = os.path.join(save_path, item)
            im_pil.save(new_pth)


def CleanReport_AddImages(file_pth, image_path, ReportType):
    data = pd.read_csv(file_pth)
    data = pd.concat([data, pd.DataFrame(columns=['images', 'finding'])])
    for index, item in tqdm(data.iterrows()):
        uid = str(int(item['uid']))
        finding = item['检查所见']
        if ReportType == 'Thyroid':
            fi = clean_finding_Thyroid(finding)
        elif ReportType == 'Mammary':
            fi = clean_finding_Mammary(finding)
        elif ReportType == 'Liver':
            fi = clean_finding_Liver(finding)
        else:
            print('error!')
            break
        print(fi)
        data['finding'][index] = fi
        img1 = os.path.join(image_path, uid + '_1.jpeg')
        img2 = os.path.join(image_path, uid + '_2.jpeg')
        if exists(img1) and exists(img2):
            data['images'][index] = [uid + '_1.jpeg', uid + '_2.jpeg']
        else:
            data['images'][index] = [uid + '_1.jpeg', uid + '_1.jpeg']
    data.to_csv(file_pth, encoding="utf_8_sig")


def csv_to_json(file, output):
    data = pd.read_csv(file, encoding="utf_8_sig")
    data = data.sample(frac=1).reset_index(drop=True)  # shuffle
    train = []
    test = []
    val = []
    length = len(data)
    train_data = data[:int(length * 0.7)]
    test_data = data[int(length * 0.7):int(length * 0.8)]
    val_data = data[int(length * 0.8):]
    dataset = [train_data, test_data, val_data]
    dataset_list = [train, test, val]
    split_list = ['train', 'test', 'val']
    for i in tqdm(range(len(dataset))):
        cur_set = dataset[i]
        tmp = []
        for index, item in cur_set.iterrows():
            uid = item['uid']
            finding = item['finding']
            split = split_list[i]
            pth = ast.literal_eval(item['images'])
            tmp.append({'uid': int(uid), 'finding': finding, 'image_path': pth, 'split': split})
        dataset_list[i] = tmp
    final_save = {}
    for i in range(len(dataset_list)):
        final_save.update({split_list[i]: dataset_list[i]})
    for split in final_save:
        for sample in final_save[split]:
            print(split, sample['finding'])
    with open(output + '_annotation.json', 'w', encoding="utf_8_sig") as f:
        json.dump(final_save, f, ensure_ascii=False)


"""
step 1, read xml and save to csv
        we have directly delete some columns by excel, and save the file as my_report.csv.
"""
Thyroid_path = 'D:/RIO/All_Datastes/甲状腺超声数据集'
Mammary_path = 'D:/RIO/All_Datastes/乳腺/乳腺'
Liver_path = 'D:/RIO/All_Datastes/脂肪肝/脂肪肝'
# thyroid_dict = read_all_xml(Thyroid_path)
# mammary_dict = read_all_xml(Mammary_path)
# liver_dict = read_all_xml(Liver_path)
# df1 = pd.DataFrame.from_dict(thyroid_dict)
# df2 = pd.DataFrame.from_dict(mammary_dict)
# df3 = pd.DataFrame.from_dict(liver_dict)
# df1.to_csv('D:/RIO/All_Datastes/整理好的超声数据集/Thyroid_rawXML.csv', encoding="utf_8_sig")
# df2.to_csv('D:/RIO/All_Datastes/整理好的超声数据集/Mammary_rawXML.csv', encoding="utf_8_sig")
# df3.to_csv('D:/RIO/All_Datastes/整理好的超声数据集/liver_rawXML.csv', encoding="utf_8_sig")

"""
step 2, Manually select useful columns
"""

"""
step 3, Find the two images in the corresponding PDF, 
        save them as JPG, and add the corresponding path to the CSV file
"""
ThyroidSave = 'D:/RIO/All_Datastes/整理好的超声数据集/rawThyroid_images'
MammarySave = 'D:/RIO/All_Datastes/整理好的超声数据集/rawMammary_images'
LiverSave = 'D:/RIO/All_Datastes/整理好的超声数据集/rawLiver_images'
# save_pdf_images(Thyroid_path, ThyroidSave)
# save_pdf_images(Mammary_path, MammarySave)
# save_pdf_images(Liver_path, LiverSave)

"""
step 4: Crop the images
"""
newThyroidPath = 'D:/RIO/All_Datastes/整理好的超声数据集/Thyroid_images'
newMammaryPath = 'D:/RIO/All_Datastes/整理好的超声数据集/Mammary_images'
newLiverPath = 'D:/RIO/All_Datastes/整理好的超声数据集/Liver_images'
# crop_save_images(ThyroidSave, newThyroidPath)
# crop_save_images(MammarySave, newMammaryPath)
# crop_save_images(LiverSave, newLiverPath)

"""
step 5: Process report text, remove redundant numbers and special symbols
        find corresponding image and aid path
"""
ThyroidRawCsv = 'D:/RIO/All_Datastes/整理好的超声数据集/Thyroid_rawXML.csv'
MammaryRawCsv = 'D:/RIO/All_Datastes/整理好的超声数据集/Mammary_rawXML.csv'
LiverRawCsv = 'D:/RIO/All_Datastes/整理好的超声数据集/Liver_rawXML.csv'
# CleanReport_AddImages(ThyroidRawCsv, newThyroidPath, 'Thyroid')
# CleanReport_AddImages(MammaryRawCsv, newMammaryPath, 'Mammary')
# CleanReport_AddImages(LiverRawCsv, newLiverPath, 'Liver')


"""
step 5: manually delete the samples that are too long
"""

"""
step 6: csv to json
"""
# csv_to_json(ThyroidRawCsv, 'D:/RIO/All_Datastes/整理好的超声数据集/Thyroid')
# csv_to_json(MammaryRawCsv, 'D:/RIO/All_Datastes/整理好的超声数据集/Mammary')
# csv_to_json(LiverRawCsv, 'D:/RIO/All_Datastes/整理好的超声数据集/Liver')
