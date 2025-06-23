import os
import torch
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from main import build_model_main
from util.slconfig import SLConfig
from datasets.coco import make_coco_transforms

# 修改为你自己的配置路径和权重路径
config_path = "config/DINO/DINO_4scale.py"
model_path = "/data_lg/keru/project/DINO/outputs/dino_acne/checkpoint_best_regular.pth"
image_dir = "/data_lg/keru/project/DINO/datasets/coco/val2017"  # 你的测试图片路径
output_dir = "./detection_visualizations"
os.makedirs(output_dir, exist_ok=True)

# 加载模型
args = SLConfig.fromfile(config_path)
args.device = 'cuda'
model, _, postprocessors = build_model_main(args)
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval().cuda()

# 类别名称
CLASSES = ['Acne', 'nodules and cysts', 'papules', 'pustules', 'whitehead and blackhead']
transform = make_coco_transforms("val")

# 阈值设置
CONF_THRESH = 0.3

def detect_and_visualize(image_path):
    img = Image.open(image_path).convert("RGB")
    img_transformed, _ = transform(img, target={})
    img_tensor = img_transformed.unsqueeze(0).cuda()

    with torch.no_grad():
        outputs = model(img_tensor)
        results = postprocessors['bbox'](outputs, target_sizes=torch.tensor([[img.height, img.width]]).cuda())[0]

    scores = results['scores'].cpu().numpy()
    boxes = results['boxes'].cpu().numpy()
    labels = results['labels'].cpu().numpy()

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    for box, score, label in zip(boxes, scores, labels):
        if score < CONF_THRESH:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{CLASSES[label]}: {score:.2f}"
        cv2.putText(img_cv, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)

    save_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, img_cv)
    print(f"Saved: {save_path}")

# 可视化前10张图片
image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))][:10]
for file in image_files:
    detect_and_visualize(os.path.join(image_dir, file))
