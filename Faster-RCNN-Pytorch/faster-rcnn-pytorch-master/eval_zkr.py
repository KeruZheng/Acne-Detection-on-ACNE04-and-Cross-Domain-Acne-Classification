import os
import json
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from collections import defaultdict
import re
# -------------------------- 1. 配置参数 --------------------------
# 路径配置
DETECTION_DIR = "/data_lg/keru/project/Faster-RCNN-Pytorch/map_out/detection-results"
GROUND_TRUTH_DIR = "/data_lg/keru/project/Faster-RCNN-Pytorch/map_out/ground-truth"
OUTPUT_DIR = "/data_lg/keru/project/Faster-RCNN-Pytorch/map_out/evaluation_results"
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualization")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# 评估参数
IOU_THRESHOLD = 0.5  # mAP 计算的 IoU 阈值
CONFIDENCE_THRESHOLD = 0.5  # 过滤低置信度预测
SAMPLE_COUNT = 5  # 正负样本各抽取数量

# 类别列表（需与训练时一致）
CLASSES = ["whitehead", "blackhead"]


# -------------------------- 2. 工具函数 --------------------------
def parse_detection_file(file_path):
    """解析检测结果文件（处理混合类别名，如 'whitehead and blackhead'）"""
    detections = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 正则匹配：字母部分（类别名，可能含空格）、数字部分（置信度、坐标）
            # 模式说明：
            # - [a-zA-Z\s]+：匹配字母和空格（类别名，如 'whitehead and blackhead'）
            # - \d+\.?\d*：匹配数字（置信度、坐标，如 0.0232、65）
            parts = re.findall(r"([a-zA-Z\s]+)|(\d+\.?\d*)", line)
            
            # 分离类别名、数字部分
            class_parts = []
            num_parts = []
            for p in parts:
                if p[0]:  # 字母部分（类别名）
                    class_parts.append(p[0].strip())
                elif p[1]:  # 数字部分
                    num_parts.append(p[1])
            
            # 类别名拼接（可能有多个单词，如 'whitehead and blackhead'）
            class_name = " ".join(class_parts)
            # 数字部分需至少 5 个（置信度 + 4 个坐标）
            if len(num_parts) < 5:
                print(f"警告: 解析错误，跳过行: {line}，文件: {file_path}")
                continue
            
            # 提取置信度和坐标
            try:
                score = float(num_parts[0])
                bbox = [float(x) for x in num_parts[1:5]]
            except ValueError:
                print(f"警告: 数字转换失败，跳过行: {line}，文件: {file_path}")
                continue
            
            detections.append({
                "class": class_name,
                "score": score,
                "bbox": bbox
            })
    return detections


def parse_ground_truth_file(file_path):
    """解析真实标签文件（处理混合类别名，如 'whitehead and blackhead'）"""
    ground_truths = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 正则匹配：字母部分（类别名，可能含空格）、数字部分（坐标）、difficult 标记
            parts = re.findall(r"([a-zA-Z\s]+)|(\d+\.?\d*)|(difficult)", line)
            
            # 分离类别名、数字部分、difficult 标记
            class_parts = []
            num_parts = []
            difficult = False
            for p in parts:
                if p[0]:  # 字母部分（类别名）
                    class_parts.append(p[0].strip())
                elif p[1]:  # 数字部分
                    num_parts.append(p[1])
                elif p[2]:  # difficult 标记
                    difficult = True
            
            # 类别名拼接
            class_name = " ".join(class_parts)
            # 数字部分需至少 4 个（坐标）
            if len(num_parts) < 4:
                print(f"警告: 解析错误，跳过行: {line}，文件: {file_path}")
                continue
            
            # 提取坐标
            try:
                bbox = [float(x) for x in num_parts[:4]]
            except ValueError:
                print(f"警告: 数字转换失败，跳过行: {line}，文件: {file_path}")
                continue
            
            ground_truths.append({
                "class": class_name,
                "bbox": bbox,
                "difficult": difficult
            })
    return ground_truths



def calculate_iou(box1, box2):
    """计算两个边界框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def draw_boxes_on_image(image_path, detections, ground_truths, output_path):
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        
        # 绘制检测框（绿色）
        for det in detections:
            if det["score"] < CONFIDENCE_THRESHOLD:
                continue
            bbox = det["bbox"]
            class_name = det["class"]
            score = det["score"]
            draw.rectangle(bbox, outline="green", width=2)
            text = f"{class_name}: {score:.2f}"
            draw.text((bbox[0], bbox[1] - 20), text, fill="green", font=font)
            draw.rectangle([bbox[0], bbox[1] - 20, bbox[0] + draw.textlength(text, font=font), bbox[1]], fill="white")
        
        # 绘制真实框（红色）
        for gt in ground_truths:
            if gt["difficult"]:
                continue
            bbox = gt["bbox"]
            class_name = gt["class"]
            draw.rectangle(bbox, outline="red", width=2)
            text = class_name
            draw.text((bbox[0], bbox[1] - 20), text, fill="red", font=font)
            draw.rectangle([bbox[0], bbox[1] - 20, bbox[0] + draw.textlength(text, font=font), bbox[1]], fill="white")
        
        # 添加图例说明
        legend_text = " green = detect_result, red = ground_truth"
        draw.text((10, 10), legend_text, fill="black", font=font)
        draw.rectangle([10, 10, 10 + draw.textlength(legend_text, font=font), 30], fill="white", outline="black")
        
        image.save(output_path)
        return True
    except Exception as e:
        print(f"图像绘制失败 {image_path}: {e}")
        return False


# -------------------------- 3. 指标计算函数 --------------------------
def calculate_precision_recall(detections, ground_truths, iou_threshold=0.5):
    """计算 Precision 和 Recall"""
    # 按类别统计
    class_detections = defaultdict(list)
    class_ground_truths = defaultdict(list)
    
    for det in detections:
        class_detections[det["class"]].append(det)
    
    for gt in ground_truths:
        if not gt["difficult"]:  # 只考虑非困难样本
            class_ground_truths[gt["class"]].append(gt)
    
    # 初始化统计量
    true_positives = 0
    false_positives = 0
    total_ground_truth = 0
    
    # 按类别计算
    for class_name in CLASSES:
        dets = class_detections.get(class_name, [])
        gts = class_ground_truths.get(class_name, [])
        
        # 按置信度排序
        dets.sort(key=lambda x: x["score"], reverse=True)
        
        # 标记已匹配的真实框
        gt_matched = [False] * len(gts)
        
        for det in dets:
            best_iou = -1
            best_idx = -1
            
            # 寻找最佳匹配的真实框
            for i, gt in enumerate(gts):
                iou = calculate_iou(det["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            # 判断是否为 TP
            if best_iou >= iou_threshold and not gt_matched[best_idx]:
                true_positives += 1
                gt_matched[best_idx] = True
            else:
                false_positives += 1
        
        total_ground_truth += len(gts)
    
    # 计算 Precision 和 Recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / total_ground_truth if total_ground_truth > 0 else 0
    
    return precision, recall

def calculate_ap(detections, ground_truths, iou_threshold=0.5):
    """计算单个类别的 AP (Average Precision)"""
    class_detections = []
    class_ground_truths = []
    
    # 收集特定类别的检测和真实框
    for det in detections:
        class_detections.append(det)
    
    for gt in ground_truths:
        if not gt["difficult"]:
            class_ground_truths.append(gt)
    
    # 按置信度排序
    class_detections.sort(key=lambda x: x["score"], reverse=True)
    
    # 标记已匹配的真实框
    gt_matched = [False] * len(class_ground_truths)
    
    # 计算 TP 和 FP
    tp = np.zeros(len(class_detections))
    fp = np.zeros(len(class_detections))
    
    for d, det in enumerate(class_detections):
        best_iou = -1
        best_idx = -1
        
        # 寻找最佳匹配的真实框
        for i, gt in enumerate(class_ground_truths):
            iou = calculate_iou(det["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        
        # 判断是否为 TP
        if best_iou >= iou_threshold and not gt_matched[best_idx]:
            tp[d] = 1
            gt_matched[best_idx] = True
        else:
            fp[d] = 1
    
    # 计算累积 TP 和 FP
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    
    # 计算 Precision 和 Recall
    precision = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp).any() else np.zeros_like(cum_tp)
    recall = cum_tp / len(class_ground_truths) if len(class_ground_truths) > 0 else np.zeros_like(cum_tp)
    
    # 使用 11 点插值法计算 AP
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        valid_precision = precision[recall >= t]
        if valid_precision.size > 0:  # 检查数组是否非空
            p = np.max(valid_precision)
        else:
            p = 0
        ap += p / 11
    
    return ap

def calculate_mAP(all_detections, all_ground_truths, iou_threshold=0.5):
    """计算所有类别的 mAP"""
    aps = []
    
    for class_name in CLASSES:
        # 收集特定类别的所有检测和真实框
        detections = []
        ground_truths = []
        
        for img_id in all_detections:
            for det in all_detections[img_id]:
                if det["class"] == class_name:
                    detections.append(det)
        
        for img_id in all_ground_truths:
            for gt in all_ground_truths[img_id]:
                if gt["class"] == class_name and not gt["difficult"]:
                    ground_truths.append(gt)
        
        # 计算该类别的 AP
        ap = calculate_ap(detections, ground_truths, iou_threshold)
        aps.append(ap)
    
    # 计算 mAP
    mAP = np.mean(aps) if aps else 0
    return mAP, aps


def calculate_average_iou(detections, ground_truths, iou_threshold=0.5):
    """计算所有匹配框的平均 IoU"""
    ious = []
    
    for img_id in detections:
        img_detections = detections[img_id]
        img_ground_truths = ground_truths.get(img_id, [])
        
        # 过滤困难样本
        img_ground_truths = [gt for gt in img_ground_truths if not gt["difficult"]]
        
        # 对每个检测框，寻找最佳匹配的真实框
        for det in img_detections:
            if det["score"] < CONFIDENCE_THRESHOLD:
                continue
                
            best_iou = 0
            for gt in img_ground_truths:
                if gt["class"] != det["class"]:
                    continue
                iou = calculate_iou(det["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
            
            # 只考虑 IoU 大于阈值的匹配
            if best_iou >= iou_threshold:
                ious.append(best_iou)
    
    # 计算平均 IoU
    return np.mean(ious) if ious else 0


# -------------------------- 4. 主程序 --------------------------
def main():
    # 读取所有检测结果和真实标签
    all_detections = {}
    all_ground_truths = {}
    
    # 假设图像路径与检测/真实标签文件同名，仅扩展名不同
    image_dir = "/data_lg/keru/project/Faster-RCNN-Pytorch/map_out/images-optional"
    
    print("读取检测结果和真实标签...")
    for filename in tqdm(os.listdir(DETECTION_DIR)):
        if not filename.endswith(".txt"):
            continue
        
        img_id = os.path.splitext(filename)[0]
        det_path = os.path.join(DETECTION_DIR, filename)
        gt_path = os.path.join(GROUND_TRUTH_DIR, filename)
        
        # 解析检测结果和真实标签
        detections = parse_detection_file(det_path)
        ground_truths = parse_ground_truth_file(gt_path)
        
        all_detections[img_id] = detections
        all_ground_truths[img_id] = ground_truths
    
    # 计算评估指标
    print("计算评估指标...")
    mAP, aps = calculate_mAP(all_detections, all_ground_truths, IOU_THRESHOLD)
    precision, recall = calculate_precision_recall(
        [det for img_dets in all_detections.values() for det in img_dets],
        [gt for img_gts in all_ground_truths.values() for gt in img_gts],
        IOU_THRESHOLD
    )
    avg_iou = calculate_average_iou(all_detections, all_ground_truths, IOU_THRESHOLD)
    
    # 准备指标数据
    metrics = {
        "mAP": mAP,
        "AP_per_class": {cls: ap for cls, ap in zip(CLASSES, aps)},
        "precision": precision,
        "recall": recall,
        "f1_score": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
        "average_iou": avg_iou,
        "iou_threshold": IOU_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"指标已保存到: {METRICS_PATH}")
    
    print("评估样本质量...")
    sample_quality = []
    for img_id in all_detections:
        dets = all_detections[img_id]
        gts = [gt for gt in all_ground_truths.get(img_id, []) if not gt["difficult"]]
        if not gts:
            continue
        matched_gts = set()
        for det in dets:
            if det["score"] < CONFIDENCE_THRESHOLD:
                continue
            best_iou, best_idx = -1, -1
            for i, gt in enumerate(gts):
                if gt["class"] == det["class"]:
                    iou = calculate_iou(det["bbox"], gt["bbox"])
                    if iou > best_iou and iou >= IOU_THRESHOLD:
                        best_iou, best_idx = iou, i
            if best_idx != -1:
                matched_gts.add(best_idx)
        match_rate = len(matched_gts) / len(gts)
        sample_quality.append({
            "img_id": img_id,
            "match_rate": match_rate,
            "num_dets": len(dets),
            "num_gts": len(gts)
        })
    
    sample_quality.sort(key=lambda x: x["match_rate"], reverse=True)
    positive_samples = [s["img_id"] for s in sample_quality if s["match_rate"] >= 0.7]
    negative_samples = [s["img_id"] for s in reversed(sample_quality) if s["match_rate"] <= 0.3]
    
    # 确保正负样本各有至少5个
    positive_samples = positive_samples[:SAMPLE_COUNT]
    negative_samples = negative_samples[:SAMPLE_COUNT]
    
    # 补充样本数量不足的情况
    if len(positive_samples) < SAMPLE_COUNT:
        # 从中间样本补充
        mid_samples = [s["img_id"] for s in sample_quality 
                      if 0.3 < s["match_rate"] < 0.7 and s["img_id"] not in positive_samples]
        positive_samples += mid_samples[:SAMPLE_COUNT - len(positive_samples)]
    
    if len(negative_samples) < SAMPLE_COUNT:
        mid_samples = [s["img_id"] for s in sample_quality 
                      if 0.3 < s["match_rate"] < 0.7 and s["img_id"] not in negative_samples]
        negative_samples += mid_samples[:SAMPLE_COUNT - len(negative_samples)]
    
    print(f"正样本数量: {len(positive_samples)}, 负样本数量: {len(negative_samples)}")
    
    print("可视化并保存样本...")
    for img_id in positive_samples + negative_samples:
        sample_type = "positive" if img_id in positive_samples else "negative"
        dets = all_detections.get(img_id, [])
        gts = all_ground_truths.get(img_id, [])
        
        image_path = os.path.join(image_dir, f"{img_id}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(image_dir, f"{img_id}.png")
            if not os.path.exists(image_path):
                print(f"警告: 图像不存在 {img_id}")
                continue
        
        output_path = os.path.join(VISUALIZATION_DIR, f"{sample_type}_{img_id}.jpg")
        draw_boxes_on_image(image_path, dets, gts, output_path)
    
    print(f"可视化结果已保存到: {VISUALIZATION_DIR}")
    print("评估完成!")


if __name__ == "__main__":
    main()