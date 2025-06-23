import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

def generate_ground_truth_txt(image_list_txt, output_dir):
    """
    根据图片路径列表文件生成对应的ground-truth txt
    image_list_txt: txt文件路径，里面每行是图片全路径，比如：
      /data_lg/keru/project/Faster-RCNN-Pytorch/dataset/test/levle0_508_jpg.rf.375bd87e515c2242195f669e55e9d1e7.jpg
    output_dir: 生成txt的目录
    """

    os.makedirs(output_dir, exist_ok=True)

    with open(image_list_txt, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        img_path = line.strip().split()[0]
        basename = os.path.splitext(os.path.basename(img_path))[0]  # 包含hash部分完整文件名
        xml_path = os.path.splitext(img_path)[0] + ".xml"
        if not os.path.exists(xml_path):
            print(f"[Warning] XML not found for image {img_path}, expected at {xml_path}")
            continue

        output_txt_path = os.path.join(output_dir, basename + ".txt")
        with open(output_txt_path, "w") as out_f:
            root = ET.parse(xml_path).getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult') is not None and obj.find('difficult').text == '1':
                    difficult_flag = True

                obj_name = obj.find('name').text
                bndbox = obj.find('bndbox')
                left = bndbox.find('xmin').text
                top = bndbox.find('ymin').text
                right = bndbox.find('xmax').text
                bottom = bndbox.find('ymax').text

                if difficult_flag:
                    out_f.write(f"{obj_name} {left} {top} {right} {bottom} difficult\n")
                else:
                    out_f.write(f"{obj_name} {left} {top} {right} {bottom}\n")

    print(f"Ground truth txt files generated at: {output_dir}")

if __name__ == "__main__":
    image_list_txt = "/data_lg/keru/project/Faster-RCNN-Pytorch/faster-rcnn-pytorch-master/map_out/2007_val.txt"  # 你的图片路径列表文件
    output_dir = "./map_out/ground-truth"  # 你想存放ground-truth txt的目录
    generate_ground_truth_txt(image_list_txt, output_dir)
