import pandas as pd
import re
import string
import pickle
import numpy as np
from typing import Generator, Tuple, Any


def extract_multiple_choice_answer(llm_output: str, options: str = "ABCD") -> str | None:
    """
    从大模型的输出文本中提取多项选择题的答案 (A, B, C, D 等)。

    考虑到 LLM 可能的输出格式，例如：
    - "A"
    - "B."
    - "答案是 C"
    - "选择 D"
    - "The correct option is A."
    - "I choose B"
    - "**C**"
    - "(D)"
    - "A. ..." (只取 A)
    - "我觉得是 B"
    - "选项 C 是正确的"
    - 等等

    Args:
        llm_output (str): 大模型生成的包含答案的文本。
        options (str): 可能的选项字母字符串，默认为 "ABCD"。

    Returns:
        str | None: 提取到的选项字母 (大写)，如果未能明确提取则返回 None。
    """
    if not llm_output:
        return None

    # 0. 预处理：去除首尾空格，统一处理大小写可能有助于匹配，但选项本身需要区分大小写，所以暂时只strip
    text = llm_output.strip()
    if not text:
        return None

    # 1. 优先匹配明确的答案指示词 + 选项 (修正版)
    #    例如："答案是 A", "选择 B", "Option: C", "选 D", "Answer is D"
    #    (?:...) 表示非捕获组
    #    \s*[:：]?\s* 匹配可选的空格、可选的冒号（中英文）、可选的空格
    #    (?:是|is)?\s* 匹配可选的 "是" 或 "is"、可选的空格
    #    ([{options}]) 捕获选项字母
    #    \b 确保选项字母是独立的单词边界（防止匹配单词内部的字母）
    pattern_explicit = rf"(?:答案|选项|选择|answer|option|choice|choose|select)\s*[:：]?\s*(?:是|is)?\s*([{options}])\b"
    match = re.search(pattern_explicit, text, re.IGNORECASE)
    if match:
        # 提取捕获组 1 (选项字母) 并转为大写
        return match.group(1).upper()

    # 2. 匹配被括号、星号等包围的单个选项字母
    #    例如："(A)", "[B]", "**C**", "*D*", "{A}"
    #    (?<!\w) 和 (?!\w) 用于确保选项字母前后不是其他字母数字字符（比 \b 更灵活）
    #    (?:...) 包裹各种可能的包围符号
    pattern_bracketed = rf"(?:[\[({{*<])\s*([{options}])\s*(?:[\])}}*>.])"
    match = re.search(pattern_bracketed, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 尝试宽松一点，匹配前后有非字母数字字符或边界的单个选项
    # 改进：使用更严格的边界检查，避免匹配 "Grade A"
    # 要求前面是开头、空格或特定标点，后面是结尾、空格或特定标点
    pattern_isolated_boundary = rf"(?:^|\s|[:：(\[{{*<])\s*([{options}])\s*(?:$|\s|[.,!?)\]}}*>:])"
    match = re.search(pattern_isolated_boundary, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 3. 匹配以选项字母 + 点/括号/空格 开头的字符串
    #    例如："A. xxx", "B) yyy", "C zzz"
    #    ^ 表示字符串开头
    #    \s* 匹配开头的可选空格
    #    使用 \b 确保选项字母后不是其他字母 (防止匹配 "Apple")
    pattern_start = rf"^\s*([{options}])\b(?:\.|\)|\s|:|$)"
    match = re.match(pattern_start, text, re.IGNORECASE) # 使用 match 确保从头开始
    if match:
        return match.group(1).upper()

    # 4. 匹配句子中 "is A", "是 B" 这样的结构 (优先级较低)
    #    确保 "is" 或 "是" 前面是一个单词边界
    pattern_is_option = rf"\b(?:is|是)\s+([{options}])\b"
    match = re.search(pattern_is_option, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 5. 最后手段：检查整个字符串是否仅仅是选项字母（可能带标点）
    #    去除所有标点和空格后，看是否只剩下一个在 options 中的字母
    cleaned_text = ''.join(c for c in text if c not in string.punctuation and not c.isspace())
    if len(cleaned_text) == 1 and cleaned_text.upper() in options.upper():
         return cleaned_text.upper()

    # 6. 如果以上策略都失败，则无法确定答案
    return None


def excel_column_generator(file_path: str, duration="all") -> Generator[Tuple[Any, Any], None, None]:
    """
    Generator that reads an xlsx file and yields values from columns K and L,
    skipping the first row (header).
    
    Args:
        file_path (str): Path to the xlsx file
        
    Yields:
        Tuple[Any, Any]: A tuple containing values from column K and L for each row
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Get columns K and L (index 10 and 11, since pandas uses 0-based indexing)
        # Column K = index 10, Column L = index 11
        if df.shape[1] < 12:  # Check if file has at least 12 columns (A-L)
            raise ValueError(f"File must have at least 12 columns (A-L). Found {df.shape[1]} columns.")
        
        # Skip header (first row) and iterate through remaining rows
        for index in range(1, len(df)):
            if duration != "all":
                if duration != df.iloc[index, 3]:
                    continue
            answer = df.iloc[index, 10]  # Column K (11th column, 0-indexed)
            prediction = df.iloc[index, 11]  # Column L (12th column, 0-indexed)
            yield (index, answer, prediction)
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return


result_paths = [
    "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250613-llavavideo-videomme-frame-num_forward_1/refine_llava/refine_llava_Video-MME.xlsx",
    "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250613-llavavideo-videomme-frame-num_forward_2/refine_llava/refine_llava_Video-MME.xlsx",
]


for duration in ["short", "medium", "long", "all"]:
    cnt1 = cnt2 = cnt12 = total = 0
    path = "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250615-llavavideo-videomme-confidence-frame-num_forward_2/refine_llava/T20250616_G/refine_llava_Video-MME.xlsx"
    extra_path = "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250615-llavavideo-videomme-confidence-frame-num_forward_2/refine_llava/T20250616_G/extra.pkl"
    extra_result = pickle.load(open(extra_path, "rb"))
    for index, answer, pred in excel_column_generator(path, duration):
        option_logits = extra_result[index]["option_logits"]
        top = np.argsort(option_logits)[::-1]
        pred1 = chr(ord('A') + top[0])
        pred2 = chr(ord('A') + top[1])
        if pred1 == answer:
            cnt1 += 1
        if pred2 == answer:
            cnt2 += 1
        if pred1 == answer or pred2 == answer:
            cnt12 += 1
        total += 1

    print(duration, cnt1 / total, cnt2 / total, cnt12 / total)


# extra_result_paths = [
#     "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250614-llavavideo-videomme-confidence-frame-num_forward_1/refine_llava/T20250614_G/extra.pkl",
#     "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250614-llavavideo-videomme-smallest-confidence-frame-num_forward_2/refine_llava/T20250614_G/extra.pkl",
# ]
# extra_results = [
#     pickle.load(open(extra_result_path, 'rb'))
#     for extra_result_path in extra_result_paths
# ]

# num_different_answers_values = []
# for duration in ["short", "medium", "long", "all"]:
#     result_generators = [excel_column_generator(path, duration) for path in result_paths]
#     correct = total = 0
#     while True:
#         try:
#             max_log_prob = float('-inf')
#             max_answer = None
#             for i in range(len(result_generators)):
#                 answer, pred = next(result_generators[i])
#                 pred = extract_multiple_choice_answer(pred)
#                 log_prob = extra_results[i][total]["log_prob"]
#                 if log_prob > max_log_prob:
#                     max_log_prob = log_prob
#                     max_answer = pred
#             if max_answer == answer:
#                 correct += 1
#             total += 1
#         except StopIteration:
#             break
#     print(duration, correct / total)


# for num_forward, path in enumerate(result_paths, start=1):
#     for duration in ["short", "medium", "long", "all"]:
#         total = correct = 0
#         for answer, pred in excel_column_generator(path, duration):
#             total += 1
#             pred = extract_multiple_choice_answer(pred)
#             correct += int(answer == pred)
#         print('###', num_forward, duration, correct / total)


# num_different_answers_values = []
# for duration in ["short", "medium", "long", "all"]:
#     result_generators = [excel_column_generator(path, duration) for path in result_paths]
#     correct = total = 0
#     while True:
#         try:
#             vote = {key: 0 for key in "ABCD"}
#             for i in range(len(result_generators)):
#                 answer, pred = next(result_generators[i])
#                 pred = extract_multiple_choice_answer(pred)
#                 vote[pred] += 1
#             if vote[answer] > 0:
#                 correct += 1
#             total += 1
#         except StopIteration:
#             break
#     print(duration, correct / total)

# baseline_wrong = num_exclude = num_correct = icl_wrong = 0
# correct = total = 0
# while True:
#     try:
#         vote = {key: 0 for key in "ABCD"}
#         for i in range(len(result_generators)):
#             answer, pred = next(result_generators[i])
#             pred = extract_multiple_choice_answer(pred)
#             vote[pred] += 1
#         if vote[answer] > 0:
#             correct += 1
#         total += 1
        # answer, pred = next(baseline_result_generator)
        # baseline_pred = extract_multiple_choice_answer(pred)
        # answer, pred = next(icl_result_generator)
        # icl_pred = extract_multiple_choice_answer(pred)
        # if baseline_pred != answer:
        #     baseline_wrong += 1
        # if icl_pred != answer:
        #     icl_wrong += 1
        # if baseline_pred != answer and vote[baseline_pred] == 0:
        #     num_exclude += 1
        # if baseline_pred != answer and vote[baseline_pred] == 0 and icl_pred == answer:
        #     num_correct += 1
    # except StopIteration:
    #     break

# print(baseline_wrong)
# print(icl_wrong)
# print(num_exclude)
# print(num_correct)
# print(correct / total)
