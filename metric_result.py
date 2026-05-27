import json
import numpy as np
from utils import *
import os

def calculate_metrics(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"=== Number: {len(data)} ===\n")
    na_types = [
        'object_counting', 
        'object_abs_distance',
        'object_size_estimation', 
        'room_size_estimation', 
    ]
    mca_types = [
        'object_rel_distance', 
        'object_rel_direction',
        'route_planning',
        'obj_appearance_order', 
    ]

    scores = {q_type: [] for q_type in na_types + mca_types}
    thetas = [round(0.5 + i * 0.05, 2) for i in range(10)]

    for item in data:
        q_type = item.get('question_type')
        pred = item.get('pred')
        gt = item.get('ground_truth')

        if q_type in ['object_rel_direction_hard', 'object_rel_direction_medium', 'object_rel_direction_easy']:
            mapped_type = 'object_rel_direction'
        else:
            mapped_type = q_type

        if mapped_type not in scores:
            continue

        if mapped_type in mca_types:
            is_correct = 1.0 if str(pred[0]).strip() == str(gt).strip() else 0.0
            scores[mapped_type].append(is_correct)
        elif mapped_type in na_types:
            try:
                y_hat = float(pred)
                y = float(gt)
                if y == 0:
                    if y_hat == 0:
                        rel_error = 0.0
                    else:
                        rel_error = float('inf')
                else:
                    rel_error = abs(y_hat - y) / y

                mra_score = 0
                for theta in thetas:
                    if rel_error < (1.0 - theta):
                        mra_score += 1
                mra_score = mra_score / 10.0
                scores[mapped_type].append(mra_score)

            except (ValueError, TypeError):
                scores[mapped_type].append(0.0)

    results_per_type = {}
    for q_type, type_scores in scores.items():
        if len(type_scores) > 0:
            results_per_type[q_type] = sum(type_scores) / len(type_scores)
        else:
            results_per_type[q_type] = 0.0 

    average_all_8_types = sum(results_per_type.values()) / 8.0

    return {
        "metrics_per_type": results_per_type,
        "average_all_8_types": average_all_8_types
    }

if __name__ == "__main__":
    json_path = 'result.json' 
    final_results = calculate_metrics(json_path)
    print("=== Question_type Acc ===")
    for q_type, acc in final_results["metrics_per_type"].items():
        print(f"{q_type}: {acc:.4f}")
    print("\n=== Average Acc ===")
    print(f"Average: {final_results['average_all_8_types']:.4f}")
