import json
import os

SUBJECT_TYPES = "人物、电视综艺、娱乐人物、影视作品、企业/品牌、歌曲、图书作品、学科专业、机构、行政区、企业、文学作品、学校、国家、历史人物、景点、地点"
OBJECT_TYPES = "学校、人物、歌曲、音乐专辑、Date、Text、Number、气候、城市、地点、奖项、作品、语言、影视作品、企业、国家"
SCHEMA_RELATIONS = "导演、出生地、毕业院校、嘉宾、配音、主题曲、代言人、所属专辑、父亲、作者、上映时间、母亲、专业代码、占地面积、邮政编码、票房、注册资本、主角、妻子、编剧、气候、歌手、获奖、校长、创始人、首都、丈夫、朝代、饰演、面积、总部地点、祖籍、人口数量、制片人、修业年限、所在城市、董事长、作词、改编自、出品公司、作曲、主演、主持人、成立日期、简称、海拔、号、国籍、官方语言"

def build_base_instruction():
    """Prompt A (Base)"""
    return (
        "请从以下文本中抽取事实三元组。\n"
        "【重要格式要求】\n"
        "你只能输出一个合法的 Python 列表结构，形如：[(\"主体\", \"关系\", \"客体\"), ...]\n"
        "如果没有符合的三元组则输出 []。\n"
        "绝对不要输出任何前言、后语、解释或 Markdown 代码块标记（如 ```python）。你的回复必须直接以 '[' 开头，以 ']' 结尾。"
    )

def build_schema_instruction():
    """Prompt B"""
    return (
        "你是一个信息抽取专家。请从以下文本中抽取事实三元组。\n"
        f"【Schema约束】\n"
        f"1. 主体只能是：{SUBJECT_TYPES}\n"
        f"2. 客体只能是：{OBJECT_TYPES}\n"
        f"3. 关系只能是：{SCHEMA_RELATIONS}\n"
        "【重要格式要求】\n"
        "你只能输出一个合法的 Python 列表结构，形如：[(\"主体\", \"关系\", \"客体\"), ...]\n"
        "如果没有符合的三元组则输出 []。\n"
        "不要捏造列表以外的关系。绝对不要输出任何解释性文本、废话或 Markdown 代码块标记（如 ```python）。你的回复必须直接以 '[' 开头，以 ']' 结尾。"
    )

def build_cot_instruction():
    """Prompt C (CoT)"""
    return (
        "你是一个信息抽取专家。请按照以下步骤从文本中抽取事实三元组：\n"
        f"主体可选类型：{SUBJECT_TYPES}\n"
        f"客体可选类型：{OBJECT_TYPES}\n"
        f"关系可选类型：{SCHEMA_RELATIONS}\n\n"
        "步骤 1: 识别文本中出现的符合类型的主体和客体实体。\n"
        "步骤 2: 根据限定的关系列表，判断实体之间存在的关系。\n"
        "步骤 3: 严格按格式输出最终结果。\n\n"
        "【重要格式要求】\n"
        "你只能输出一个合法的 Python 列表结构，形如：[(\"主体\", \"关系\", \"客体\"), ...]\n"
        "如果没有符合的三元组则输出 []。\n"
        "不要捏造列表以外的关系。绝对不要输出任何解释性文本、废话或 Markdown 代码块标记（如 ```python）。你的回复必须直接以 '[' 开头，以 ']' 结尾。"
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