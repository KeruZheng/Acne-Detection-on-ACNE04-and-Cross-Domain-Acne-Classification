import os
import random
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from frcnn import FRCNN

def draw_boxes(image, boxes, labels, scores, class_names, color="green"):
    """
    在图片上绘制bbox和标签
    boxes: [[xmin, ymin, xmax, ymax], ...]
    labels: [label_idx, ...]
    scores: [score, ...]
    color: 框颜色，"green" 或 "red"
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = map(int, box)
        class_name = class_names[label]
        text = f"{class_name} {score:.2f}"

        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        text_size = draw.textsize(text, font=font)
        draw.rectangle([xmin, ymin - text_size[1], xmin + text_size[0], ymin], fill=color)
        draw.text((xmin, ymin - text_size[1]), text, fill="white", font=font)

    return image

def draw_single_box(draw, box, label, color, text=None, font=None):
    xmin, ymin, xmax, ymax = map(int, box)
    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
    if text and font:
        text_size = draw.textsize(text, font=font)
        draw.rectangle([xmin, ymin - text_size[1], xmin + text_size[0], ymin], fill=color)
        draw.text((xmin, ymin - text_size[1]), text, fill="white", font=font)

def parse_prediction_txt(pred_txt_path):
    """
    解析frcnn.get_map_txt生成的预测txt文件
    格式示例：
    class_name score xmin ymin xmax ymax
    """
    boxes = []
    labels = []
    scores = []
    with open(pred_txt_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            if len(line) != 6:
                continue
            cls_name = line[0]
            score = float(line[1])
            xmin, ymin, xmax, ymax = map(float, line[2:6])
            boxes.append([xmin, ymin, xmax, ymax])
            scores.append(score)
            labels.append(cls_name)  # 这里先保存类名，后面映射索引时用
    return boxes, labels, scores

def parse_groundtruth_txt(gt_txt_path):
    """
    解析ground-truth txt文件
    格式：
    class_name xmin ymin xmax ymax [difficult]
    """
    boxes = []
    labels = []
    difficults = []
    with open(gt_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_name = parts[0]
            xmin, ymin, xmax, ymax = map(float, parts[1:5])
            difficult = (len(parts) == 6 and parts[5] == 'difficult')
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(cls_name)
            difficults.append(difficult)
    return boxes, labels, difficults

def voc_iou(boxA, boxB):
    """
    计算两个bbox的IoU，bbox格式为[xmin, ymin, xmax, ymax]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_sample(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thresh=0.5):
    """
    简单评估该图预测的准确度：
    - 匹配的正确预测数
    - 召回率 recall = 匹配正确的 / gt总数
    - 准确率 precision = 匹配正确的 / 预测总数
    """
    matched = 0
    gt_matched = [False] * len(gt_boxes)
    for pb, pl in zip(pred_boxes, pred_labels):
        for i, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
            if not gt_matched[i] and pl == gl:
                if voc_iou(pb, gb) >= iou_thresh:
                    matched += 1
                    gt_matched[i] = True
                    break
    recall = matched / len(gt_boxes) if gt_boxes else 0
    precision = matched / len(pred_boxes) if pred_boxes else 0
    return matched, precision, recall

if __name__ == "__main__":
    import sys

    map_mode        = 0
    classes_path    = './faster-rcnn-pytorch-master/model_data/voc_classes.txt'
    MINOVERLAP      = 0.5
    confidence      = 0.02
    nms_iou         = 0.5
    score_threhold  = 0.5
    map_vis         = True
    map_out_path    = '/data_lg/keru/project/Faster-RCNN-Pytorch/map_out'

    # 你的测试集txt路径
    image_id_path = '/data_lg/keru/project/Faster-RCNN-Pytorch/faster-rcnn-pytorch-master/map_out/2007_val.txt'

    # 读取图片路径和对应的id (不带后缀的basename)
    image_ids = []
    with open(image_id_path, 'r') as f:
        for line in f:
            path = line.strip().split()[0]
            basename = os.path.splitext(os.path.basename(path))[0]
            image_ids.append((path, basename))

    # 创建输出目录
    for subfolder in ['ground-truth', 'detection-results', 'images-optional', 'correct', 'wrong']:
        os.makedirs(os.path.join(map_out_path, subfolder), exist_ok=True)

    class_names, _ = get_classes(classes_path)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # 加载模型
    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        frcnn = FRCNN(confidence=confidence, nms_iou=nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_path, image_id in tqdm(image_ids):
            if not os.path.exists(image_path):
                print(f"[Warning] Image not found: {image_path}")
                continue
            image = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional", image_id + ".jpg"))
            frcnn.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    # 获取真实框
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_path, image_id in tqdm(image_ids):
            xml_path = os.path.splitext(image_path)[0] + ".xml"
            if not os.path.exists(xml_path):
                print(f"[Warning] XML not found: {xml_path}")
                continue

            with open(os.path.join(map_out_path, "ground-truth", image_id + ".txt"), "w") as new_f:
                root = ET.parse(xml_path).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') is not None:
                        if int(obj.find('difficult').text) == 1:
                            difficult_flag = True

                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left   = bndbox.find('xmin').text
                    top    = bndbox.find('ymin').text
                    right  = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write(f"{obj_name} {left} {top} {right} {bottom} difficult\n")
                    else:
                        new_f.write(f"{obj_name} {left} {top} {right} {bottom}\n")
        print("Get ground truth result done.")

    # 计算 VOC mAP
    if map_mode == 0 or map_mode == 3:
        print("Get mAP.")
        get_map(MINOVERLAP, draw_plot=True, score_threhold=score_threhold, path=map_out_path)

        # 自动读取并输出指标
        results_file = os.path.join(map_out_path, "results.txt")
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                lines = f.readlines()
                ap_lines = [line.strip() for line in lines if "AP for" in line]
                mAP_line = next((line.strip() for line in lines if "mAP =" in line), None)

                print("\n===== Evaluation Metrics =====")
                if mAP_line:
                    print(f"✅ Mean Average Precision (mAP): {mAP_line.split('=')[-1].strip()}")
                else:
                    print("❌ mAP not found in results.txt")

                print("\n🎯 Per-class AP:")
                for ap_line in ap_lines:
                    print(f"  {ap_line}")

                print(f"\n📏 IoU Threshold used: {MINOVERLAP}")
                print(f"📊 Confidence threshold for PR: {score_threhold}")
        else:
            print("[Warning] results.txt not found. Cannot display metrics.")

        print("Get mAP done.")
    
    # 额外：保存随机的正确和错误检测图片
    if map_mode == 0:
        print("Evaluating samples and saving visualization...")

        sample_results = []

        for image_path, image_id in tqdm(image_ids):
            gt_txt_path = os.path.join(map_out_path, "ground-truth", image_id + ".txt")
            pred_txt_path = os.path.join(map_out_path, "detection-results", image_id + ".txt")

            if not os.path.exists(gt_txt_path) or not os.path.exists(pred_txt_path):
                continue

            gt_boxes, gt_labels, _ = parse_groundtruth_txt(gt_txt_path)
            pred_boxes, pred_labels_str, pred_scores = parse_prediction_txt(pred_txt_path)

            # 把pred_labels_str转换成索引
            pred_labels = [class_to_idx.get(cls_name, -1) for cls_name in pred_labels_str]

            # 过滤无效预测类别
            filtered = [(b, l, s) for b, l, s in zip(pred_boxes, pred_labels, pred_scores) if l >= 0]
            if not filtered:
                continue
            pred_boxes, pred_labels, pred_scores = zip(*filtered)

            matched, precision, recall = evaluate_sample(
                pred_boxes, pred_labels, pred_scores, gt_boxes, [class_to_idx[l] for l in gt_labels]
            )

            sample_results.append({
                "image_path": image_path,
                "image_id": image_id,
                "matched": matched,
                "total_gt": len(gt_boxes),
                "precision": precision,
                "recall": recall,
                "pred_boxes": pred_boxes,
                "pred_labels": pred_labels,
                "pred_scores": pred_scores,
                "gt_boxes": gt_boxes,
                "gt_labels": gt_labels
            })

        # 定义正确/错误的判定标准
        correct_samples = [x for x in sample_results if x["total_gt"] > 0 and x["matched"] >= 0.7 * x["total_gt"]]
        wrong_samples = [x for x in sample_results if x["total_gt"] > 0 and x["matched"] <= 0.3 * x["total_gt"]]

        # 随机抽样，最多5张
        correct_samples = random.sample(correct_samples, min(5, len(correct_samples)))
        wrong_samples = random.sample(wrong_samples, min(5, len(wrong_samples)))

        correct_dir = os.path.join(map_out_path, "correct")
        wrong_dir = os.path.join(map_out_path, "wrong")

        for sample in correct_samples:
            img = Image.open(sample["image_path"]).convert("RGB")
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            # 画预测框(绿色+置信度)
            for box, label_idx, score in zip(sample["pred_boxes"], sample["pred_labels"], sample["pred_scores"]):
                label = class_names[label_idx]
                text = f"{label} {score:.2f}"
                draw_single_box(draw, box, label, "green", text=text, font=font)

            # 画真实框(红色+类名)
            for box, label in zip(sample["gt_boxes"], sample["gt_labels"]):
                draw_single_box(draw, box, label, "red", text=label, font=font)

            img.save(os.path.join(correct_dir, sample["image_id"] + ".jpg"))

        for sample in wrong_samples:
            img = Image.open(sample["image_path"]).convert("RGB")
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            for box, label_idx, score in zip(sample["pred_boxes"], sample["pred_labels"], sample["pred_scores"]):
                label = class_names[label_idx]
                text = f"{label} {score:.2f}"
                draw_single_box(draw, box, label, "green", text=text, font=font)

            for box, label in zip(sample["gt_boxes"], sample["gt_labels"]):
                draw_single_box(draw, box, label, "red", text=label, font=font)

            img.save(os.path.join(wrong_dir, sample["image_id"] + ".jpg"))

        print(f"Saved {len(correct_samples)} correct and {len(wrong_samples)} wrong detection samples.")

