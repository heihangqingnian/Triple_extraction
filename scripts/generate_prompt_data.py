import json
import os

# 从 infer.py 中提取的三种 Prompt 类型定义
SUBJECT_TYPES = "人物、电视综艺、娱乐人物、影视作品、企业/品牌、歌曲、图书作品、学科专业、机构、行政区、企业、文学作品、学校、国家、历史人物、景点、地点"
OBJECT_TYPES = "学校、人物、歌曲、音乐专辑、Date、Text、Number、气候、城市、地点、奖项、作品、语言、影视作品、企业、国家"
SCHEMA_RELATIONS = "导演、出生地、毕业院校、嘉宾、配音、主题曲、代言人、所属专辑、父亲、作者、上映时间、母亲、专业代码、占地面积、邮政编码、票房、注册资本、主角、妻子、编剧、气候、歌手、获奖、校长、创始人、首都、丈夫、朝代、饰演、面积、总部地点、祖籍、人口数量、制片人、修业年限、所在城市、董事长、作词、改编自、出品公司、作曲、主演、主持人、成立日期、简称、海拔、号、国籍、官方语言"


def build_base_instruction():
    """Prompt A (Base): 直接给出句子，要求输出三元组"""
    return "请从以下文本中抽取事实三元组。输出格式为：[(\"主体\", \"关系\", \"客体\"), ...]，如果没有符合的三元组则输出 []。"


def build_schema_instruction():
    """Prompt B (添加 Schema 约束): 在 Prompt 中明确告诉模型 DUIE 包含哪些关系类型和实体类型"""
    return (
        f"你是一个信息抽取专家。请从以下文本中抽取事实三元组。\n"
        f"主体实体类型只能从以下列表中选择：{SUBJECT_TYPES}\n"
        f"对象实体类型只能从以下列表中选择：{OBJECT_TYPES}\n"
        f"关系类型只能从以下列表中选择：{SCHEMA_RELATIONS}\n"
        '输出格式为：[("主体", "关系", "客体"), ...]，如果没有符合的三元组则输出 []。'
    )


def build_cot_instruction():
    """Prompt C (添加 CoT / 思想链): 提示模型先找出实体，再判断关系，最后输出，同时指定 Schema 约束"""
    return (
        "你是一个信息抽取专家。请按照以下步骤从文本中抽取事实三元组：\n"
        f"主体实体类型只能从以下列表中选择：{SUBJECT_TYPES}\n"
        f"对象实体类型只能从以下列表中选择：{OBJECT_TYPES}\n"
        f"关系类型只能从以下列表中选择：{SCHEMA_RELATIONS}\n"
        "1. 首先识别文本中出现的所有符合上述类型的实体；\n"
        "2. 然后根据给定的关系类型列表，判断这些实体之间可能存在的关系；\n"
        "3. 最后将符合条件的三元组以列表形式输出。\n"
        '输出格式为：[("主体", "关系", "客体"), ...]，如果没有符合的三元组则输出 []。'
    )


def extract_text_from_input(input_str):
    """从输入中提取原始文本（去除 Schema 部分）"""
    if "### 文本\n" in input_str:
        return input_str.split("### 文本\n")[-1].strip()
    return input_str.strip()


def process_file(input_path, output_path, instruction_func):
    """处理单个文件，生成对应 Prompt 类型的数据"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    new_data = []
    for sample in data:
        text = extract_text_from_input(sample["input"])
        new_sample = {
            "instruction": instruction_func(),
            "input": text,
            "output": sample["output"]
        }
        new_data.append(new_sample)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    print(f"已生成: {output_path}，共 {len(new_data)} 条数据")


def main():
    base_dir = "e:/Research/Research1/data/processed/llm"
    output_dir = "e:/Research/Research1/data/processed/llm"
    
    os.makedirs(output_dir, exist_ok=True)
    
    file_types = ["train", "dev", "test"]
    prompt_types = {
        "base": build_base_instruction,
        "schema": build_schema_instruction,
        "cot": build_cot_instruction
    }
    
    for file_type in file_types:
        input_path = os.path.join(base_dir, f"{file_type}.json")
        if not os.path.exists(input_path):
            print(f"警告：输入文件不存在: {input_path}")
            continue
        
        for prompt_name, instruction_func in prompt_types.items():
            output_path = os.path.join(output_dir, f"{file_type}_{prompt_name}.json")
            process_file(input_path, output_path, instruction_func)


if __name__ == "__main__":
    main()