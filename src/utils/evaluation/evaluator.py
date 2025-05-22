# import json
# from sklearn.metrics import f1_score
# from sentence_transformers import SentenceTransformer, util
# from typing import List, Dict
# import numpy as np

# class Evaluator:
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
#         self.embedder = SentenceTransformer(model_name)

#     def load_ground_truth(self, file_path: str) -> List[Dict]:
#         with open(file_path, "r") as f:
#             return json.load(f)

#     def exact_match(self, predicted: str, ground_truth: str) -> bool:
#         return predicted.strip().lower() == ground_truth.strip().lower()

#     def f1(self, predicted: str, ground_truth: str) -> float:
#         pred_tokens = predicted.lower().split()
#         gold_tokens = ground_truth.lower().split()
#         common = set(pred_tokens) & set(gold_tokens)
#         if not common:
#             return 0.0
#         precision = len(common) / len(pred_tokens)
#         recall = len(common) / len(gold_tokens)
#         if precision + recall == 0:
#             return 0.0
#         return 2 * (precision * recall) / (precision + recall)

#     def semantic_similarity(self, predicted: str, ground_truth: str) -> float:
#         emb1 = self.embedder.encode(predicted, convert_to_tensor=True)
#         emb2 = self.embedder.encode(ground_truth, convert_to_tensor=True)
#         return float(util.pytorch_cos_sim(emb1, emb2)[0][0])

#     def evaluate(self, qa_pairs: List[Dict], answer_fn) -> Dict[str, float]:
#         results = []
#         for pair in qa_pairs:
#             question = pair["question"]
#             expected = pair["answer"]
#             generated = answer_fn(question)

#             exact = self.exact_match(generated, expected)
#             f1_val = self.f1(generated, expected)
#             sem_sim = self.semantic_similarity(generated, expected)

#             results.append({
#                 "question": question,
#                 "generated": generated,
#                 "expected": expected,
#                 "exact_match": exact,
#                 "f1_score": f1_val,
#                 "semantic_sim": sem_sim
#             })

#         # Aggregate results
#         exact_total = sum(r["exact_match"] for r in results) / len(results)
#         avg_f1 = np.mean([r["f1_score"] for r in results])
#         avg_sem_sim = np.mean([r["semantic_sim"] for r in results])

#         return {
#             "exact_match_accuracy": exact_total,
#             "average_f1": avg_f1,
#             "average_semantic_similarity": avg_sem_sim
#         }



import json
import re
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer, util
from .text_normalizer import normalize_text  # Adjust import if needed

# Load model once at module level to avoid overhead
_semantic_model = SentenceTransformer('all-MiniLM-L6-v2')


class Evaluator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)

    def load_ground_truth(self, file_path: str) -> List[Dict]:
        with open(file_path, "r") as f:
            return json.load(f)

    def exact_match(self, predicted: str, ground_truth: str) -> bool:
        pred_norm = normalize_text(predicted)
        gt_norm = normalize_text(ground_truth)

        pred_tokens = set(pred_norm.split())
        gt_tokens = set(gt_norm.split())

        # Strict equality or subset match (lenient)
        return pred_tokens == gt_tokens or pred_tokens.issubset(gt_tokens) or gt_tokens.issubset(pred_tokens)

    def f1(self, predicted: str, ground_truth: str) -> float:
        pred_norm = normalize_text(predicted)
        gt_norm = normalize_text(ground_truth)
        pred_tokens = pred_norm.split()
        gt_tokens = gt_norm.split()
        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def semantic_similarity(self, predicted: str, ground_truth: str) -> float:
        emb1 = self.embedder.encode(predicted, convert_to_tensor=True)
        emb2 = self.embedder.encode(ground_truth, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(emb1, emb2)[0][0])


    def evaluate(self, qa_pairs: List[Dict], answer_fn) -> Dict[str, float]:
        results = []
        for pair in qa_pairs:
            question = pair["question"]
            expected = pair["answer"]
            generated = answer_fn(question)

            exact = self.exact_match(generated, expected)
            f1_val = self.f1(generated, expected)
            sem_sim = self.semantic_similarity(generated, expected)

            results.append({
                "question": question,
                "generated": generated,
                "expected": expected,
                "exact_match": exact,
                "f1_score": f1_val,
                "semantic_sim": sem_sim
            })

        exact_total = sum(r["exact_match"] for r in results) / len(results)
        avg_f1 = np.mean([r["f1_score"] for r in results])
        avg_sem_sim = np.mean([r["semantic_sim"] for r in results])

        return {
            "exact_match_accuracy": exact_total,
            "average_f1": avg_f1,
            "average_semantic_similarity": avg_sem_sim
        }
