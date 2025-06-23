import json
import pickle
import os.path as osp
import numpy as np
import pandas as pd


def show(result_dir, model_name):
    rating_path = osp.join(result_dir, f"{model_name}/{model_name}_Video-MME_rating.json")
    extra_path = osp.join(result_dir, f"{model_name}/T20250622_G/extra.pkl")
    excel_path = osp.join(result_dir, f"{model_name}/{model_name}_Video-MME.xlsx")

    if not osp.isfile(rating_path) or not osp.isfile(extra_path) or not osp.isfile(excel_path):
        return

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
    for _, row in excel.iterrows():
        index = row['index']
        duration = row['duration']
        cnt[duration] += 1
        peak_mem[duration] += extra[index]['peak_mem']
        total_time[duration] += extra[index]['time']

    for duration in cnt.keys():
        total_time[duration] /= cnt[duration]
        peak_mem[duration] /= cnt[duration] * 2**30

    return result, total_time, peak_mem


baseline_result, baseline_total_time, baseline_peak_mem = show("/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250621-llavavideo-baseline", "baseline_llava")
baseline_total_time["overall"] = sum(baseline_total_time.values())
baseline_peak_mem["overall"] = max(baseline_peak_mem.values())
print('### baseline')
print(baseline_result)
print(baseline_total_time)
print(baseline_result)
print()

for alpha1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for alpha2 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        result_dir = f"/mnt/afs/wangkaibin/VLMEvalKit/outputs/20250621-llavavideo-videomme-prodecode-alpha1_alpha2_{alpha1}_{alpha2}"
        try:
            result, total_time, peak_mem = show(result_dir, "prodecode_llava")
        except:
            continue
        print('###', alpha1, alpha2)
        print(result)
        total_time["overall"] = sum(total_time.values())
        peak_mem["overall"] = max(peak_mem.values())
        for key in total_time.keys():
            total_time[key] = (baseline_total_time[key] - total_time[key]) / baseline_total_time[key]
            peak_mem[key] = (baseline_peak_mem[key] - peak_mem[key]) / baseline_peak_mem[key]
        print(total_time)
        print(peak_mem)
        print()
