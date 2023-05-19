import numpy as np
import cv2
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO


import os
import cv2
import random
import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
import colorsys

def generate_colors(num_colors):
    # 计算每个颜色的色相值
    hue_values = [i / num_colors for i in range(num_colors)]
    # 随机生成亮度值和饱和度值
    saturation = 1.0
    brightness = 1.0
    # 生成K种不同颜色
    colors = []
    for hue in hue_values:
        # 将HSV颜色空间转换为RGB颜色空间
        rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)
        # 将RGB值转换为整数值（0-255）
        rgb = [int(255 * c) for c in rgb]
        # 添加该颜色到列表中
        colors.append(rgb)
    return colors

def visualize_segmentation(json_path,dataset_dir="dootrim", save_dir="results"):
    # 加载COCO数据集标注
    coco = COCO(json_path)
    #查看coco中有多少种类
    colors=generate_colors(len(coco.dataset['categories']))
    # 获取所有图像的ID
    img_ids = coco.getImgIds()
    # 遍历每个图像ID
    for img_id in img_ids:
        try:
            # 加载图像的元数据
            img_data = coco.loadImgs(img_id)[0]
            # 加载图像的标注数据
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            # 创建与图像大小相同的空白图像
            #img = np.zeros((img_data['height'], img_data['width'], 3), np.uint8)
            #读取图像
            print(os.path.join(dataset_dir,img_data['file_name']))
            img = cv2.imread(os.path.join(dataset_dir,img_data['file_name']))
            # 遍历每个标注
            if len(anns) == 0:
                continue
            for ann in anns:
                # 从标注中获取掩膜数据
                if ann['iscrowd']:
                    mask = maskUtils.decode(ann['segmentation'])
                else:
                    if type(ann['segmentation']) == list:
                        mask = coco.annToMask(ann)
                    else:
                        rle = ann['segmentation']
                        mask = maskUtils.decode(rle)
                print(mask.mean())
                # 为该标注随机生成半透明的颜色
                color = colors[ann['category_id'] - 1]
                alpha = 0.5
                print(color)
                # 将该标注的掩膜数据可视化到空白图像上
                img[mask > 0] = alpha * np.array(color) + (1 - alpha) * img[mask > 0]
                #画出bbox
                bbox=ann['bbox']
                cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])),color,2)
            # 保存可视化后的图像
            save_path = os.path.join(save_dir, f"{img_data['file_name']}")
            print(save_path)
            cv2.imwrite(save_path, img)
        except:
            continue
if __name__ == "__main__":
    visualize_segmentation('new_annotations.json', dataset_dir="/home/wangrui/wr-data/ceramicTile/cropTile_ccd/20230508", save_dir="results")