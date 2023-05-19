from segment_anything import sam_model_registry, SamPredictor
import tqdm
from pycocotools.coco import COCO
import numpy as np
import cv2
import os
import json
import logging
import glob
from pycocotools import mask as maskUtils

logging.basicConfig(level=logging.INFO)

class JsonEncoder(json.JSONEncoder):
    """Convert numpy classes to JSON serializable objects."""
 
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)

class det2seg():
    def __init__(self, 
                 coco_json_path = "F:/salt/doortrim/annotations.json",
                  dataset_path = "F:/salt/doortrim",
                 sam_checkpoint = "F:/salt/sam_vit_b_01ec64.pth",
                 model_type = "vit_b",
                 device = "cuda",
                 RLEmode = True
                 ):
        self.coco_json_path = coco_json_path
        self.dataset_path = dataset_path
        self.coco = COCO(coco_json_path)
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        
    def transfer(self):
        #输出coco中有多少张图片
        print("There are {} images in the dataset".format(len(self.coco.dataset['images'])))
        for i in tqdm.tqdm(range(len(self.coco.dataset['images']))):
            image_id = self.coco.dataset['images'][i]['id']
            image_info = self.coco.loadImgs(image_id)[0]
            image_path = os.path.join(self.dataset_path, image_info['file_name'])
            image = cv2.imread(image_path)
            #如果image为空，则跳过进入下一个循环
            if image is None:
                continue
            self.predictor.set_image(image)
            # Get all annotations for the image
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                bbox0 = ann['bbox'].copy()
                bbox = ann['bbox']
                #bbox 由[x,y,w,h]转化为[x1,y1,x2,y2]
                bbox[2] = bbox[0] + bbox[2]
                bbox[3] = bbox[1] + bbox[3]
                bbox = np.array(bbox)
                masks, _, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=bbox,    
                        multimask_output=False,
                            )
                            #把masks按RLE格式保存
                if self.RLEmode:
                    mask = masks[0]
                    rle_mask =  maskUtils.encode(np.asfortranarray(mask))
                    print(rle_mask)
                    #rle=maskUtils.frPyObjects(rle_mask,image_info['height'],image_info['width'])
                    #将rle_mask中的counts转化为字符串
                    rle_mask['counts'] = rle_mask['counts'].decode('utf-8')
                    ann['segmentation'] = rle_mask
                    #更新信息到coco同一个id的annotation中
                else:
                #把mask通过approxPolyDP转化为分割轮廓标注
                    mask = masks[0]
                    mask = mask.astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        points = contours[0]
                        epsilon = 0.01 * cv2.arcLength(points, True)
                        approx = cv2.approxPolyDP(points, epsilon, True)
                        #转为一维列表
                        segmentation = approx.flatten().tolist()
                        ann['segmentation'] = [segmentation]
                        #更新信息到coco同一个id的annotation中
                        #self.coco.dataset['annotations'][ann['id']] = ann
                    else:
                        ann['segmentation'] = [[bbox[0],bbox[1],bbox[2],bbox[1],bbox[2],bbox[3],bbox[0],bbox[3]]]
                ann['bbox']=bbox0
                try:
                    self.coco.dataset['annotations'][ann['id']] = ann
                except:
                    continue
    def save(self,path="new_annotations.json"):
        with open(path, "w") as f:
            json.dump(self.coco.dataset, f,ensure_ascii=False, cls=JsonEncoder)

if __name__ == "__main__":
    sam_checkpoint = "/home/fangzhenghao/workspace/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    coco_json_path = "/home/wangrui/ceramicTile/high_precision/cis/json/20230417/val.json"
    dataset_path = "/home/wangrui/wr-data/ceramicTile/cropTile_pic/20230417"
    #dataset_path = "/home/wangrui/wr-data/ceramicTile/cropTile_ccd/20230508"
    det2seg = det2seg(coco_json_path=coco_json_path,dataset_path=dataset_path,sam_checkpoint=sam_checkpoint,model_type=model_type,device=device)
    det2seg.transfer()
    det2seg.save()
    print("done")