import json
import pickle
import os
import numpy as np
import pandas as pd
from collections import defaultdict


def show_VideoMME(root):
    for name in os.listdir(root):
        if name.endswith("Video-MME_extra.pkl"):
            extra_path = os.path.join(root, name)
        elif name.endswith("Video-MME_rating.json"):
            rating_path = os.path.join(root, name)
        elif name.endswith("Video-MME.xlsx"):
            excel_path = os.path.join(root, name)

    with open(rating_path) as f:
        rating = json.load(f)
    result = {
        'short': rating['short']['overall'],
        'medium': rating['medium']['overall'],
        'long': rating['long']['overall'],
        'overall': rating['overall']['overall'],
    }

    with open(extra_path, 'rb') as f:
        extra = pickle.load(f)
    total_time = {"short": 0, "medium": 0, "long": 0}
    peak_mem = {"short": 0, "medium": 0, "long": 0}
    cnt = {"short": 0, "medium": 0, "long": 0}
    excel = pd.read_excel(excel_path)
    details = []
    for _, row in excel.iterrows():
        index = row['index']
        duration = row['duration']
        detail_item = {
            'split': duration,
            "gt": ord(row['answer']) - ord('A'),
        }
        detail_item.update(extra[index])
        details.append(detail_item)
        cnt[duration] += 1
        peak_mem[duration] += extra[index]['peak_mem']
        total_time[duration] += extra[index]['time']

    total_time["overall"] = sum(total_time.values())
    peak_mem["overall"] = max(peak_mem.values())

    for duration in cnt.keys():
        total_time[duration] /= cnt[duration]
        peak_mem[duration] /= cnt[duration] * 2**30

    extra = {
        "total_time": total_time['overall'],
        "peak_mem": peak_mem['overall']
    }

    # extra = details = None
    return result, extra, details


def show_merge(root1, idx1, root2, idx2, dataset):
    _, _, detail1 = show(root1, dataset)
    _, _, detail2 = show(root2, dataset)
    total = defaultdict(int)
    correct = defaultdict(int)
    for it1, it2 in zip(detail1, detail2):
        first = it1['first_option_logits'] if idx1 == 1 else it1['second_option_logits']
        second = it2['first_option_logits'] if idx2 == 1 else it2['second_option_logits']
        merged = first + second
        answer = np.argmax(merged)
        # top2 = np.argsort(first)[-2:]
        # answer = top2[0] if second[top2[0]] >= second[top2[1]] else top2[1]
        total[it1['split']] += 1
        correct[it1['split']] += int(answer == it1['gt'])
    for key in total.keys():
        print(key, correct[key] / total[key])
    print('overall', sum(correct.values()) / sum(total.values()))


def show_LongVideoBench(root):
    for name in os.listdir(root):
        if name.endswith("LongVideoBench_extra.pkl"):
            extra_path = os.path.join(root, name)
        elif name.endswith("LongVideoBench_rating.json"):
            rating_path = os.path.join(root, name)
        elif name.endswith("LongVideoBench_score.xlsx"):
            score_path = os.path.join(root, name)

    with open(rating_path) as f:
        rating = json.load(f)
    result = {
        '15': rating['15']['overall'],
        '60': rating['60']['overall'],
        '600': rating['600']['overall'],
        '3600': rating['3600']['overall'],
        'overall': rating['overall']['overall'],
    }

    with open(extra_path, 'rb') as f:
        extra = pickle.load(f)

    df = pd.read_excel(score_path)
    details = []
    for index, row in df.iterrows():
        detail_item = {
            "split": str(row['duration_group']),
            "gt": row['correct_choice'],
        }
        detail_item.update(extra[index])
        details.append(detail_item)

    total_time = sum(it['time'] for it in extra.values()) / len(extra)
    peak_mem = max(it['peak_mem'] for it in extra.values())

    extra = {
        "total_time": total_time,
        "peak_mem": peak_mem
    }

    # extra = details = None
    return result, extra, details


def show_MLVU(root):
    for name in os.listdir(root):
        if name.endswith("MLVU_MCQ_extra.pkl"):
            extra_path = os.path.join(root, name)
        elif name.endswith("MLVU_MCQ_score.xlsx"):
            score_path = os.path.join(root, name)

    result = defaultdict(int)
    total = defaultdict(int)
    df = pd.read_excel(score_path)
    for _, row in df.iterrows():
        task_type = row['task_type']
        score = row['score']
        result[task_type] += score
        total[task_type] += 1
    # for task_type in result.keys():
    #     result[task_type] /= total[task_type]
    # m_avg = sum(result.values()) / len(result)
    m_avg = sum(result.values()) / sum(total.values())
    for task_type in result.keys():
        result[task_type] /= total[task_type]
    result['M-Avg'] = m_avg

    with open(extra_path, 'rb') as f:
        extra = pickle.load(f)
    total_time = sum(it['time'] for it in extra.values()) / len(extra)
    peak_mem = max(it['peak_mem'] for it in extra.values())

    details = []
    for index, row in df.iterrows():
        if not isinstance(row['answer'], str):
            row['answer'] = ''
        detail_item = {
            "split": row['task_type'],
            "gt": eval(row['candidates']).index(row['answer']),
        }
        if index not in extra:
            continue
        detail_item.update(extra[index])
        details.append(detail_item)

    extra = {
        "total_time": total_time,
        "peak_mem": peak_mem
    }

    # extra = details = None
    return result, extra, details


def show(root, dataset):
    root = next(os.path.join(root, item) for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)))
    cache_folder = ''
    for subfolder in os.listdir(root):
        if not os.path.isdir(os.path.join(root, subfolder)):
            continue
        if subfolder.startswith('T2025') and subfolder > cache_folder:
            cache_folder = subfolder
    # root = os.path.join(root, cache_folder)

    if dataset == "VideoMME":
        return show_VideoMME(root)
    if dataset == "LongVideoBench":
        return show_LongVideoBench(root)
    if dataset == "MLVU":
        return show_MLVU(root)


def stable_softmax(x, temperature=1.0):
    x = np.asarray(x) / temperature
    x_max = np.max(x, axis=-1, keepdims=True)
    x_exp = np.exp(x - x_max)
    return x_exp / np.sum(x_exp, axis=-1, keepdims=True)


def produce_of_experts(logits_list):
    logits_concat = np.stack(logits_list)
    prob_concat = stable_softmax(logits_concat)
    unscaled_prob = np.prod(prob_concat, axis=0)
    normalization_factor = np.sum(unscaled_prob, axis=-1, keepdims=True)
    merged_prob = unscaled_prob / (normalization_factor + 1e-8)
    return merged_prob


def show_poe_metrics():
    for dataset in [
        # "VideoMME",
        # "LongVideoBench",
        "MLVU"
    ]:
        # root = "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250625-qwenvl-prodecode-nframe256-alpha1_alpha2_0.5_0.6"
        # result, extra, details = show(root, dataset)
        # print('### baseline', dataset)
        # print(result)
        # print(extra)
        # print()

        root1 = "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250714-qwenvl-prodecode_randframe_token_logits-nframe256-alpha1_alpha2_0.5_0.3"
        result1, extra1, details1 = show(root1, dataset)
        correct = defaultdict(int)
        total = defaultdict(int)
        print('###', dataset)
        for detail in details1:
            logits1 = detail['first_option_logits']
            logits2 = detail['second_option_logits']
            merged_prob = produce_of_experts([logits2, logits2])
            answer = np.argmax(merged_prob)
            if detail['gt'] == answer:
                correct[detail['split']] += 1
                correct['overall'] += 1
            total[detail['split']] += 1
            total['overall'] += 1
        for key in correct.keys():
            print(key, correct[key] / total[key])
        print()


def show_avg_metrics():
    for dataset in [
        # "VideoMME",
        # "LongVideoBench",
        "MLVU"
    ]:
        root = "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250714-qwenvl-prodecode_randframe_token_logits-nframe256-alpha1_alpha2_0.5_0.3"
        result, extra, details = show(root, dataset)
        correct = defaultdict(int)
        total = defaultdict(int)
        print('###', dataset)
        for detail in details:
            logits1 = detail['first_option_logits']
            logits2 = detail['second_option_logits']
            merged_prob = (logits1 + logits2) / 2
            # merged_prob = logits1
            answer = np.argmax(merged_prob)
            if detail['gt'] == answer:
                correct[detail['split']] += 1
                correct['overall'] += 1
            total[detail['split']] += 1
            total['overall'] += 1
        for key in correct.keys():
            print(key, correct[key] / total[key])


def entropy(p):
    """Calculates the Shannon entropy of a probability distribution."""
    # Filter out zero probabilities to avoid log(0) which results in NaN
    return -np.sum(np.log2(np.maximum(p, 0) + 1e-6))


def show_entropy_metrics():
    for dataset in [
        "VideoMME",
        "LongVideoBench",
        "MLVU"
    ]:
        root = "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250714-qwenvl-prodecode_randframe_token_logits-nframe256-alpha1_alpha2_0.5_0.3"
        result, extra, details = show(root, dataset)
        correct = defaultdict(int)
        total = defaultdict(int)
        print('###', dataset)
        for detail in details:
            logits1 = detail['first_option_logits']
            logits2 = detail['second_option_logits']
            prob1 = stable_softmax(logits1)
            prob2 = stable_softmax(logits2)

            entropy1 = entropy(prob1)
            entropy2 = entropy(prob2)
            try:
                weight1 = 1 / (entropy1 + 1e-6)
                weight2 = 1 / (entropy2 + 1e-6)
            except RuntimeWarning as e:
                print('######', entropy1, entropy2)
                breakpoint()
            total_weight = weight1 + weight2
            norm_weight1 = weight1 / total_weight
            norm_weight2 = weight2 / total_weight

            merged_prob = norm_weight1 * prob1 + norm_weight2 * prob2
            answer = np.argmax(merged_prob)
            if detail['gt'] == answer:
                correct[detail['split']] += 1
                correct['overall'] += 1
            total[detail['split']] += 1
            total['overall'] += 1
        for key in correct.keys():
            print(key, correct[key] / total[key])


def show_ablation_bsz():
    root1 = "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250717-qwenvl-prodecode_randframe_token_logits-nframe256-part1-05_05"
    root2 = "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250717-qwenvl-prodecode_randframe_token_logits-nframe256-part2-05_05"
    for dataset in [
        # "VideoMME",
        "LongVideoBench",
        # "MLVU"
    ]:
        _, _, details1 = show(root1, dataset)
        _, _, details2 = show(root2, dataset)
        for B in range(1, 5):
            B =  4
            print()
            print('###', dataset, B)
            for perm in range(4**4):
                correct = defaultdict(int)
                total = defaultdict(int)
                for it1, it2 in zip(details1, details2):
                    logits = []
                    for i in range(4):
                       j = (perm >> (i * 2)) & 3
                       logits.append([
                            it1['first_option_logits'],
                            it1['second_option_logits'],
                            it2['second_option_logits'],
                            it2['first_option_logits'],
                        ][j])
                    logits = np.stack(logits)
                    # merged_prob = np.sum(logits[:B], axis=0) / B
                    merged_prob = np.prod(stable_softmax(logits, 100)[:B], axis=0)
                    answer = np.argmax(merged_prob)
                    if it1['gt'] == answer:
                        correct[it1['split']] += 1
                        correct['overall'] += 1
                    total[it1['split']] += 1
                    total['overall'] += 1

                for key in ['overall']:
                    print(key, correct[key] / total[key])
            break


# baseline_result, baseline_total_time, baseline_peak_mem = show(
#     "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250621-llavavideo-videomme-baseline/baseline_llava/T20250622_G/baseline_llava_Video-MME_rating.json",
#     "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250621-llavavideo-videomme-baseline/baseline_llava/T20250622_G/extra.pkl",
#     "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250621-llavavideo-videomme-baseline/baseline_llava/T20250622_G/baseline_llava_Video-MME.xlsx"
# )
# baseline_total_time["overall"] = sum(baseline_total_time.values())
# baseline_peak_mem["overall"] = max(baseline_peak_mem.values())
# print('### baseline')
# print(baseline_result)
# print(baseline_total_time)
# print(baseline_result)
# print()


def show_20250625_qwenvl_prodecode_nframe256_alpha1_alpha2():
    for alpha1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for alpha2 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for dataset in ["VideoMME", "LongVideoBench", "MLVU"]:
                try:
                    result, extra, details = show(
                        f"/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250625-qwenvl-prodecode-nframe256-alpha1_alpha2_{alpha1}_{alpha2}",
                        dataset
                    )
                except:
                    continue
                
                print('###', dataset, alpha1, alpha2)
                if 'M-Avg' in result:
                    print(round(result['M-Avg'], 3))
                else:
                    print(result['overall'])
            print()


def show_20250628_qwenvl_prodecode_nframe128x2_alpha1_alpha2():
    for dataset in ["MLVU"]:
        baseline_result, baseline_extra, baseline_details = show(
            "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250625-qwenvl-baseline_logits-nframe256",
            dataset
        )
        for alpha1 in [0.5, 0.6, 0.7, 0.8, 0.9]:
            alpha2 = alpha1
            try:
                result, extra, details = show(
                    f"/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250628-qwenvl-prodecode-nframe128x2-alpha1_alpha2_{alpha1}_{alpha2}",
                    dataset
                )
            except:
                continue
            
            print('###', dataset, alpha1, alpha2)
            print(result)
            print((baseline_extra['total_time'] - extra['total_time']) / baseline_extra['total_time'])
            print((baseline_extra['peak_mem'] - extra['peak_mem']) / baseline_extra['peak_mem'])


def grid_search():
    # root_tmpl = "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250726-oryx-fixframe-token-nframe256-alpha1_alpha2_{}_{}"
    # root_tmpl = "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250726-oryx-fixframe-token-nframe256-mme-alpha1_alpha2_{}_{}"
    root_tmpl = "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250725-oryx-randframe-token-nframe256-alpha1_alpha2_{}_{}"
    
    logits = {}
    gt = []
    alpha_space = ['0.2', '0.3', '0.4', '0.5', '0.6']
    for alpha in alpha_space:
        print(alpha)
        root = root_tmpl.format(alpha, alpha)
        # result, extra, details = show(root, "VideoMME")
        # result, extra, details = show(root, "LongVideoBench")
        result, extra, details = show(root, "MLVU")
        logits[alpha] = []
        for detail in details:
            logits[alpha].append([detail['first_option_logits'], detail['second_option_logits']])
            if len(gt) < len(logits[alpha]):
                gt.append(detail['gt'])
    best_acc = 0
    best_alpha1 = best_alpha2 = 0
    for i in range(len(alpha_space) * 2):
        for j in range(len(alpha_space) * 2):
            if i == j:
                continue
            alpha1 = str(alpha_space[i // 2])
            alpha2 = str(alpha_space[j // 2])
            logits1 = logits[alpha1]
            logits2 = logits[alpha2]
            total = correct = 0
            for idx, (first, second) in enumerate(zip(logits1, logits2)):
                first = first[i%2]
                second = second[i%2]
                merged = first + second
                # merged = stable_softmax(first) + stable_softmax(second)
                # merged = stable_softmax(first) * stable_softmax(second)
                # merged = first
                answer = np.argmax(merged)
                # top2 = np.argsort(first)[-2:]
                # answer = top2[0] if second[top2[0]] >= second[top2[1]] else top2[1]
                total += 1
                correct += int(answer == gt[idx])
            acc = correct / total
            # if acc > 0.570:
            print('#', acc, alpha1, alpha2)
            if acc > best_acc or acc == best_acc and float(alpha1) + float(alpha2) < best_alpha1 + best_alpha2:
                best_acc = acc
                best_alpha1 = float(alpha1)
                best_alpha2 = float(alpha2)
    print(best_acc, best_alpha1, best_alpha2)


def show_single():
    root = "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20251116-vtw_qwen3vl30b-nframe128-K24/vtw_qwen3vl30b"
    for dataset in [
            "VideoMME",
            "LongVideoBench",
            "MLVU",
        ]:
        print('###', dataset)
        result, extra, details = show(root, dataset)
        print(result)

        # baseline_result, baseline_extra, baseline_details = show(
        #     "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20251031-qwenvl-baseline_logits-nframe256/baseline_qwen2vl_logits",
        #     # "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250704-llavavideo-baseline_logits-nframe128/baseline_llava_logits",
        #     # "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250708-oryx-baseline_logits-nframe256/baseline_oryx_logits",
        #     dataset
        # )
        # print(details[0], baseline_details[0])
        # print(baseline_extra['total_time'] / extra['total_time'])
        # # print((baseline_extra['peak_mem'] - extra['peak_mem']) / baseline_extra['peak_mem'])
        # print()


def grid_search_qwen3vl():
    root_tmpl = "/mnt/afs/wangkaibin/VLMEvalKit/outputs/20251117-prodecode_qwen3vl30b_randframe_token_logits-nframe64-alpha1_alpha2_{}_{}"
    # dataset = "VideoMME"
    # dataset = "LongVideoBench"
    dataset = "MLVU"

    logits = {}
    gt = []
    alpha_space = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6']
    for alpha in alpha_space:
        print(alpha)
        root = root_tmpl.format(alpha, alpha)
        result, extra, details = show(root, dataset)
        logits[alpha] = []
        for detail in details:
            logits[alpha].append([detail['first_option_logits'], detail['second_option_logits']])
            if len(gt) < len(logits[alpha]):
                gt.append(detail['gt'])

    # logits['0.6'] = []
    # result, extra, details1 = show("/mnt/afs/wangkaibin/VLMEvalKit/outputs/20251109-prodecode_qwen3vl_randframe_token_logits-nframe256-alpha1_alpha2_0.1_0.6", dataset)
    # result, extra, details2 = show("/mnt/afs/wangkaibin/VLMEvalKit/outputs/20251109-prodecode_qwen3vl_randframe_token_logits-nframe256-alpha1_alpha2_0.6_0.1", dataset)
    # for detail1, detail2 in zip(details1, details2):
    #     logits['0.6'].append([detail2['first_option_logits'], detail1['second_option_logits']])
    #     if len(gt) < len(logits[alpha]):
    #         gt.append(detail['gt'])
    # alpha_space.append('0.6')

    best_acc = 0
    best_alpha1 = best_alpha2 = 0
    for i in range(len(alpha_space) * 2):
        for j in range(len(alpha_space) * 2):
            if i == j:
                continue
            alpha1 = str(alpha_space[i // 2])
            alpha2 = str(alpha_space[j // 2])
            logits1 = logits[alpha1]
            logits2 = logits[alpha2]
            print(alpha1, alpha2, len(logits1), len(logits2))
            total = correct = 0
            for idx, (first, second) in enumerate(zip(logits1, logits2)):
                first = first[i%2]
                second = second[i%2]
                merged = first + second
                # merged = stable_softmax(first) + stable_softmax(second)
                # merged = stable_softmax(first) * stable_softmax(second)
                # merged = first
                answer = np.argmax(merged)
                # top2 = np.argsort(first)[-2:]
                # answer = top2[0] if second[top2[0]] >= second[top2[1]] else top2[1]
                total += 1
                correct += int(answer == gt[idx])
            acc = correct / total
            # if acc > 0.570:
            print('#', acc, alpha1, alpha2)
            if acc > best_acc or acc == best_acc and float(alpha1) + float(alpha2) < best_alpha1 + best_alpha2:
                best_acc = acc
                best_alpha1 = float(alpha1)
                best_alpha2 = float(alpha2)
    print(best_acc, best_alpha1, best_alpha2)


if __name__ == "__main__":
    # show_single()
    # show_avg_metrics()
    # show_poe_metrics()
    # show_entropy_metrics()
    # show_20250625_qwenvl_prodecode_nframe256_alpha1_alpha2()
    # show_20250628_qwenvl_prodecode_nframe128x2_alpha1_alpha2()
    # grid_search()
    show_merge(
        '/mnt/afs/wangkaibin/VLMEvalKit/outputs/20251117-prodecode_qwen3vl30b_randframe_token_logits-nframe64-alpha1_alpha2_0.5_0.5', 2,
        '/mnt/afs/wangkaibin/VLMEvalKit/outputs/20251117-prodecode_qwen3vl30b_randframe_token_logits-nframe64-alpha1_alpha2_0.6_0.6', 2,
        "VideoMME",
        # "LongVideoBench",
        # 'MLVU',
    )
    # show_ablation_bsz()
    # grid_search_qwen3vl()
