# import json
# import re
# from typing import List, Dict
# import numpy as np
# from sentence_transformers import SentenceTransformer, util
# from .text_normalizer import normalize_text  # Adjust import if needed
# import nltk
# from rouge_score import rouge_scorer

# # Download required NLTK data (only once)
# nltk.download('punkt_tab')

# # Load model once at module level to avoid overhead
# _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')


# class Evaluator:
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
#         self.embedder = SentenceTransformer(model_name)
#         self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

#     def load_ground_truth(self, file_path: str) -> List[Dict]:
#         with open(file_path, "r") as f:
#             return json.load(f)

#     def exact_match(self, predicted: str, ground_truth: str) -> bool:
#         pred_norm = normalize_text(predicted)
#         gt_norm = normalize_text(ground_truth)

#         pred_tokens = set(pred_norm.split())
#         gt_tokens = set(gt_norm.split())

#         # Strict equality or subset match (lenient)
#         return pred_tokens == gt_tokens or pred_tokens.issubset(gt_tokens) or gt_tokens.issubset(pred_tokens)

#     def f1(self, predicted: str, ground_truth: str) -> float:
#         pred_norm = normalize_text(predicted)
#         gt_norm = normalize_text(ground_truth)
#         pred_tokens = pred_norm.split()
#         gt_tokens = gt_norm.split()
#         common = set(pred_tokens) & set(gt_tokens)
#         if not common:
#             return 0.0
#         precision = len(common) / len(pred_tokens)
#         recall = len(common) / len(gt_tokens)
#         if precision + recall == 0:
#             return 0.0
#         return 2 * (precision * recall) / (precision + recall)

#     def semantic_similarity(self, predicted: str, ground_truth: str) -> float:
#         emb1 = self.embedder.encode(predicted, convert_to_tensor=True)
#         emb2 = self.embedder.encode(ground_truth, convert_to_tensor=True)
#         return float(util.pytorch_cos_sim(emb1, emb2)[0][0])
    
#     def semantic_match(self, predicted: str, ground_truth: str, threshold: float = 0.8) -> bool:
#         sim = self.semantic_similarity(predicted, ground_truth)
#         return sim >= threshold

#     def compute_bleu(self, predicted: str, ground_truth: str) -> float:
#         # Tokenize sentences into words
#         pred_tokens = nltk.word_tokenize(predicted.lower())
#         gt_tokens = nltk.word_tokenize(ground_truth.lower())
#         # BLEU expects list of references, each reference is a list of tokens
#         return nltk.translate.bleu_score.sentence_bleu([gt_tokens], pred_tokens)

#     def compute_rouge(self, predicted: str, ground_truth: str) -> Dict[str, float]:
#         scores = self.rouge_scorer.score(ground_truth, predicted)
#         # Extract F1 scores of ROUGE-1, ROUGE-2, ROUGE-L
#         return {
#             'rouge1_f1': scores['rouge1'].fmeasure,
#             'rouge2_f1': scores['rouge2'].fmeasure,
#             'rougeL_f1': scores['rougeL'].fmeasure,
#         }

#     def evaluate(self, qa_pairs: List[Dict], answer_fn) -> Dict[str, float]:
#         results = []
#         for pair in qa_pairs:
#             question = pair["question"]
#             expected = pair["answer"]
#             generated = answer_fn(question)

#             exact = self.exact_match(generated, expected)
#             f1_val = self.f1(generated, expected)
#             sem_sim = self.semantic_similarity(generated, expected)
#             relaxed_match = self.semantic_match(generated, expected, threshold=0.8)
#             bleu_score = self.compute_bleu(generated, expected)
#             rouge_scores = self.compute_rouge(generated, expected)

#             results.append({
#                 "question": question,
#                 "generated": generated,
#                 "expected": expected,
#                 "exact_match": exact,
#                 "relaxed_match": relaxed_match,
#                 "f1_score": f1_val,
#                 "semantic_sim": sem_sim,
#                 "bleu": bleu_score,
#                 **rouge_scores,
#             })

#         return {
#             "exact_match_accuracy": sum(r["exact_match"] for r in results) / len(results),
#             "relaxed_match_accuracy": sum(r["relaxed_match"] for r in results) / len(results),
#             "average_f1": np.mean([r["f1_score"] for r in results]),
#             "average_semantic_similarity": np.mean([r["semantic_sim"] for r in results]),
#             "average_bleu": np.mean([r["bleu"] for r in results]),
#             "average_rouge1_f1": np.mean([r["rouge1_f1"] for r in results]),
#             "average_rouge2_f1": np.mean([r["rouge2_f1"] for r in results]),
#             "average_rougeL_f1": np.mean([r["rougeL_f1"] for r in results]),
#         }


import json
import numpy as np
from typing import List, Dict, Callable
from sentence_transformers import SentenceTransformer, util
from .text_normalizer import normalize_text
import nltk
from rouge_score import rouge_scorer

# Ensure tokenizer resources are downloaded
nltk.download('punkt', quiet=True)

class Evaluator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )

    def load_ground_truth(self, file_path: str) -> List[Dict]:
        with open(file_path, "r") as f:
            return json.load(f)

    def exact_match(self, predicted: str, ground_truth: str) -> bool:
        pred_tokens = set(normalize_text(predicted).split())
        gt_tokens = set(normalize_text(ground_truth).split())
        return pred_tokens == gt_tokens or pred_tokens.issubset(gt_tokens) or gt_tokens.issubset(pred_tokens)

    def f1(self, predicted: str, ground_truth: str) -> float:
        pred_tokens = normalize_text(predicted).split()
        gt_tokens = normalize_text(ground_truth).split()
        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    def semantic_similarity(self, predicted: str, ground_truth: str) -> float:
        emb1 = self.embedder.encode(predicted, convert_to_tensor=True)
        emb2 = self.embedder.encode(ground_truth, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(emb1, emb2)[0][0])

    def semantic_match(self, predicted: str, ground_truth: str, threshold: float = 0.8) -> bool:
        return self.semantic_similarity(predicted, ground_truth) >= threshold

    def compute_bleu(self, predicted: str, ground_truth: str) -> float:
        pred_tokens = nltk.word_tokenize(predicted.lower())
        gt_tokens = nltk.word_tokenize(ground_truth.lower())
        return nltk.translate.bleu_score.sentence_bleu([gt_tokens], pred_tokens)

    def compute_rouge(self, predicted: str, ground_truth: str) -> Dict[str, float]:
        scores = self.rouge_scorer.score(ground_truth, predicted)
        return {
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rougeL_f1': scores['rougeL'].fmeasure,
        }

    def evaluate(
        self,
        qa_pairs: List[Dict],
        answer_fn: Callable[[str], str],
        return_details: bool = False
    ) -> Dict[str, float]:
        results = []
        for pair in qa_pairs:
            question = pair["question"]
            expected = pair["answer"]
            generated = answer_fn(question)

            metrics = {
                "question": question,
                "generated": generated,
                "expected": expected,
                "exact_match": self.exact_match(generated, expected),
                "relaxed_match": self.semantic_match(generated, expected),
                "f1_score": self.f1(generated, expected),
                "semantic_sim": self.semantic_similarity(generated, expected),
                "bleu": self.compute_bleu(generated, expected),
                **self.compute_rouge(generated, expected),
            }
            results.append(metrics)

        aggregated = {
            "exact_match_accuracy": np.mean([r["exact_match"] for r in results]),
            "relaxed_match_accuracy": np.mean([r["relaxed_match"] for r in results]),
            "average_f1": np.mean([r["f1_score"] for r in results]),
            "average_semantic_similarity": np.mean([r["semantic_sim"] for r in results]),
            "average_bleu": np.mean([r["bleu"] for r in results]),
            "average_rouge1_f1": np.mean([r["rouge1_f1"] for r in results]),
            "average_rouge2_f1": np.mean([r["rouge2_f1"] for r in results]),
            "average_rougeL_f1": np.mean([r["rougeL_f1"] for r in results]),
        }

        if return_details:
            return {"metrics": aggregated, "details": results}
        return aggregated

    def save_ragas_format(
        self,
        qa_pairs: List[Dict],
        answer_fn: Callable[[str], str],
        output_path: str
    ):
        ragas_data = []
        for pair in qa_pairs:
            question = pair["question"]
            ground_truth = pair["answer"]
            answer = answer_fn(question)

            # Add placeholder context if not available
            ragas_data.append({
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "contexts": []  # Should ideally be retrieved passages
            })

        with open(output_path, "w") as f:
            json.dump(ragas_data, f, indent=2)
        print(f"[✓] Saved RAGAs-compatible dataset to {output_path}")
