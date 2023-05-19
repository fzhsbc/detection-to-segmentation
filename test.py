from det2segv2 import det2seg
from drawseg import visualize_segmentation

if __name__ == "__main__":
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    coco_json_path = "val.json"
    dataset_path = "cropTile_pic/"
    det2seg = det2seg(coco_json_path=coco_json_path,dataset_path=dataset_path,sam_checkpoint=sam_checkpoint,model_type=model_type,device=device)
    det2seg.transfer()
    det2seg.save()
    print("done")
    visualize_segmentation('new_annotations.json', dataset_dir="images", save_dir="results")