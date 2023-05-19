# detection-to-segmentation
use SAM(segment anything) to transfer detection label to segmentation
利用segment anything将coco格式的检测标签转换为分割标签
# 先安装SAM
pip install -r requirements.txt
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
# 使用方法
```
python test.py
```
TODO
 other dataset surpport， labeling tool。
