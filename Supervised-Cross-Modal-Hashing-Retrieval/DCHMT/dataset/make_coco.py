import os
import numpy as np


def make_index(jsonData: dict, indexDict: dict):
    """
    use coco dict data as orignial data.
    indexDict: {jsonData's key: [index_key, index_value]}
    """
    result = []
    for name in indexDict:
        data = jsonData[name]
        middle_dict = {}
        for item in data:
            if item[indexDict[name][0]] not in middle_dict:
                middle_dict.update({item[indexDict[name][0]]: [item[indexDict[name][1]]]})
            else:
                middle_dict[item[indexDict[name][0]]].append(item[indexDict[name][1]])
        result.append(middle_dict)

    return result

def check_file_exist(indexDict: dict, file_path: str):
    keys = list(indexDict.keys())
    for item in keys:
        # print(indexDict[item])
        if not os.path.exists(os.path.join(file_path, indexDict[item][0])):
            print(item, indexDict[item])
            indexDict.pop(item)
        indexDict[item] = os.path.join(file_path, indexDict[item][0])
    return indexDict

def chage_categories2numpy(category_ids: dict, data: dict):
    
    for item in data:
        class_item = [0] * len(category_ids)
        for class_id in data[item]:
            class_item[category_ids[class_id]] = 1
        data[item] = np.asarray(class_item)

    return data

def get_all_use_key(categoryDict: dict):
    return list(categoryDict.keys())

def remove_not_use(data: dict, used_key: list):

    keys = list(data.keys())
    for item in keys:
        if item not in used_key:
            # print("remove:", item, indexDict[item])
            data.pop(item)
    # print(len(category_list))
    return data

def merge_to_list(data: dict):

    result = []
    key_sort = list(data.keys())
    key_sort.sort()
    # print(key_sort)
    # print(key_sort.index(91654))

    for item in key_sort:
        result.append(data[item])

    return result


if __name__ == "__main__":
    import json
    import scipy.io as scio
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-dir", default="./", type=str, help="the coco dataset dir")
    parser.add_argument("--save-dir", default="./", type=str, help="mat file saved dir")
    args = parser.parse_args()
    
    
    PATH = args.coco_dir
    jsonFile = os.path.join(PATH, "annotations", "captions_train2017.json")
    with open(jsonFile, "r") as f:
        jsonData = json.load(f)
    indexDict = {"images": ["id", "file_name"], "annotations": ["image_id", "caption"]}
    result = make_index(jsonData, indexDict)
    indexDict_, captionDict = result
    indexDict_ = check_file_exist(indexDict_, os.path.join(PATH, "train2017"))
    print("caption:", len(indexDict_), len(captionDict))
    # print_result = list(indexDict.keys())
    # print_result.sort()
    # print(print_result)
    # indexList = merge_to_list(indexDict_)
    # captionList = merge_to_list(captionDict)
    # print(indexDict[565962], indexList[4864])
    # print(captionDict[565962], captionList[4864])
    # print(result)
    jsonFile = os.path.join(PATH, "annotations", "instances_train2017.json")
    with open(jsonFile, "r") as f:
        jsonData = json.load(f)
    categroy_ids = {}
    for i, item in enumerate(jsonData['categories']):
        categroy_ids.update({item['id']: i})
    indexDict = {"annotations": ["image_id", "category_id"], "images": ["id", "file_name"]}
    result = make_index(jsonData, indexDict)
    categoryDict = result[0]
    cateIndexDict = result[1]
    # cateIndexList = merge_to_list(cateIndexDict)
    # print(categoryDict[91654])
    categoryDict = chage_categories2numpy(categroy_ids, categoryDict)
    # print(categoryDict[91654])
    # categoryList = merge_to_list(categoryDict)
    # print(categoryDict[91654], categoryList[780])
    # print(indexList[100], cateIndexList[100])
    # print("category:", len(categoryDict), len(cateIndexList))
    used_key = get_all_use_key(categoryDict)
    # 统一index
    indexDict_ = remove_not_use(indexDict_, used_key)
    captionDict = remove_not_use(captionDict, used_key)
    categoryIndexDict = remove_not_use(cateIndexDict, used_key)
    categoryDict = remove_not_use(categoryDict, used_key)
    # 转变为list
    indexList = merge_to_list(indexDict_)
    captionList = merge_to_list(captionDict)
    categoryIndexList = merge_to_list(categoryIndexDict)
    categoryList = merge_to_list(categoryDict)
    print("result", len(indexDict_), len(categoryDict))
    print("category:", len(categoryDict), len(categoryIndexList))
    for i in range(len(indexList)):
        if indexList[i] != categoryIndexList[i]:
            print("Not the same:", i, indexList[i], categoryIndexList[i])
    
    val_jsonFile = os.path.join(PATH, "annotations", "captions_val2017.json")
    with open(val_jsonFile, "r") as f:
         jsonData = json.load(f)
    indexDict = {"images": ["id", "file_name"], "annotations": ["image_id", "caption"]}
    result = make_index(jsonData, indexDict)
    val_indexDict = result[0]
    val_captionDict = result[1]
    val_indexDict = check_file_exist(val_indexDict, os.path.join(PATH, "val2017"))
    jsonFile = os.path.join(PATH, "annotations", "instances_val2017.json")
    with open(jsonFile, "r") as f:
        jsonData = json.load(f)
    categroy_ids = {}
    for i, item in enumerate(jsonData['categories']):
        categroy_ids.update({item['id']: i})
    indexDict = {"annotations": ["image_id", "category_id"], "images": ["id", "file_name"]}
    result = make_index(jsonData, indexDict)
    val_categoryDict = result[0]
    val_categoryIndexDict = result[1]
    val_categoryDict = chage_categories2numpy(categroy_ids, val_categoryDict)
    used_key = get_all_use_key(val_categoryDict)
    val_indexDict = remove_not_use(val_indexDict, used_key)
    val_captionDict = remove_not_use(val_captionDict, used_key)
    val_categoryIndexDict = remove_not_use(val_categoryIndexDict, used_key)
    val_categoryDict = remove_not_use(val_categoryDict, used_key)
    
    val_indexList = merge_to_list(val_indexDict)
    val_captionList = merge_to_list(val_captionDict)
    val_categoryIndexList = merge_to_list(val_categoryIndexDict)
    val_categoryList = merge_to_list(val_categoryDict)

    indexList.extend(val_indexList)
    captionList.extend(val_captionList)
    categoryIndexList.extend(val_categoryIndexList)
    categoryList.extend(val_categoryList)

    print(len(indexList), len(captionList), len(categoryIndexList))
    indexs = {"index": indexList}
    captions = {"caption": captionList}
    categorys = {"category": categoryList}

    scio.savemat(os.path.join(args.save_dir, "index.mat"), indexs)
    scio.savemat(os.path.join(args.save_dir, "caption.mat"), captions)
    scio.savemat(os.path.join(args.save_dir, "label.mat"), categorys)



