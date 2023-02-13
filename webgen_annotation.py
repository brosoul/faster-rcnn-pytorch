import json
import os
# 指定比例
trainval_percent    = 0.9
train_percent       = 0.9


output_path  = './webgen/output'
classes_path        = f'{output_path}/cls.txt'
images_path = f'{output_path}/images/'
annatations_path = f'{output_path}/web_gen_annotations.json'


def read_cls(cls_path):
    """
    读取类别文件，获取类别列表
    """

    with open(cls_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

ALL_CLS, _ = read_cls(classes_path)

def get_filename(key):
    file_name = key.split(".")[0] 
    file_abs_path = f"{images_path}{file_name}.png"
    return file_name


ALL_TYPES = set()
def get_annatations_line(key, vals):
    res = {}
    filename = get_filename(key)
    res = {"filename": filename, "object": []}
    for obj in vals:
        region_attributes = obj["region_attributes"]
        shape_attributes= obj["shape_attributes"]
        cls_name = region_attributes["type"]
        ALL_TYPES.add(cls_name)
        left = shape_attributes["x"]
        top = shape_attributes["y"]
        bottom = top + shape_attributes["height"]
        right = left + shape_attributes["width"]
        cls_idx = ALL_CLS.index(cls_name) if cls_name in ALL_CLS else -1
        border_desc = map(str, (left, top, right, bottom, cls_idx))
        res["object"].append(",".join(border_desc))
    return res


def write_set_info(data_set, filename):
    with open(file=filename, mode="w") as f:
        for data in data_set: 
            f.write(data["filename"])
            f.write("\n")


def write_anna_info(data_set, filename):
    with open(file=filename, mode="w") as f:
        for data in data_set:
            abspath = os.path.abspath(f"{images_path}{data['filename']}.png")
            line = f"{abspath} {' '.join(data['object'])}"
            f.write(line)
            f.write("\n")

if __name__ == "__main__":
    annatations_dict = json.load(open(annatations_path))

    all_annatations = []

    for key, vals in annatations_dict.items():
        """
        构造出对应的 绝对路径
        """
        annatations_line = get_annatations_line(key, vals)
        all_annatations.append(annatations_line)

    all_size = len(all_annatations)
    # 根据比例，拆分数据集
    trainval_size = int(all_size * trainval_percent)
    train_size = int(trainval_size * train_percent)
    train_set = all_annatations[:train_size]
    val_set = all_annatations[train_size:trainval_size]
    test_set = all_annatations[trainval_size:]
    
    write_set_info(train_set, filename="./train.txt")
    write_set_info(val_set, filename="./val.txt")
    write_set_info(test_set, filename="./test.txt")
    write_anna_info(train_set, filename=f"{output_path}/train_anno.txt")
    write_anna_info(val_set, filename=f"{output_path}/val_anno.txt")
    write_anna_info(test_set, filename=f"{output_path}/test_anno.txt")
    print(ALL_TYPES)


    # with open("./classes.txt", mode="w") as f:
    #     for kind in ALL_TYPES:
    #         f.write(f"{kind}\n")

    print("debug")

