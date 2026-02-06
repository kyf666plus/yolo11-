# debug_json_detailed.py
import json
from pathlib import Path


def debug_json_detailed():
    """详细调试JSON文件结构"""

    json_dir = Path(r"D:\数据集标注\下井动作_yolo\annotations\train")
    json_files = list(json_dir.glob("*.json"))

    if not json_files:
        print("❌ 没有找到JSON文件")
        return

    # 检查第一个JSON文件
    json_file = json_files[0]
    print(f"🔍 检查文件: {json_file.name}\n")

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📊 JSON是一个列表，包含 {len(data)} 个元素\n")

    # 查看第一个元素
    first_item = data[0]
    print(f"第一个元素的键: {list(first_item.keys())}\n")

    print(f"file_upload: {first_item.get('file_upload')}")
    print(f"id: {first_item.get('id')}")

    # 查看annotations
    if 'annotations' in first_item and first_item['annotations']:
        print(f"\nannotations 长度: {len(first_item['annotations'])}")
        annotation = first_item['annotations'][0]
        print(f"annotation 的键: {list(annotation.keys())}")

        if 'result' in annotation and annotation['result']:
            print(f"\nresult 长度: {len(annotation['result'])}")
            result = annotation['result'][0]
            print(f"result[0] 的键: {list(result.keys())}")

            if 'value' in result:
                value = result['value']
                print(f"\nvalue 的键: {list(value.keys())}")

                if 'sequence' in value:
                    sequence = value['sequence']
                    print(f"\n✅ 找到 sequence！长度: {len(sequence)}")
                    if sequence:
                        print(f"\nsequence[0] 的键: {list(sequence[0].keys())}")
                        print(f"第一帧示例:")
                        for key, val in sequence[0].items():
                            print(f"  {key}: {val}")

    # 查找当前JSON文件对应的视频
    print(f"\n\n🔍 查找文件名匹配...")
    json_stem = json_file.stem
    print(f"JSON文件名: {json_stem}")

    # 在所有元素中查找匹配的file_upload
    for i, item in enumerate(data):
        file_upload = item.get('file_upload', '')
        if json_stem in file_upload or file_upload.replace('.mp4', '') == json_stem:
            print(f"\n✅ 找到匹配！索引: {i}")
            print(f"   file_upload: {file_upload}")
            print(f"   id: {item.get('id')}")

            if 'annotations' in item and item['annotations']:
                ann = item['annotations'][0]
                if 'result' in ann and ann['result']:
                    res = ann['result'][0]
                    if 'value' in res and 'sequence' in res['value']:
                        seq = res['value']['sequence']
                        print(f"   sequence长度: {len(seq)}")
                        break


if __name__ == "__main__":
    debug_json_detailed()