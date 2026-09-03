import numpy as np
import nltk
import selfies as sf

from abc import ABC, abstractmethod
from typing import List
from Levenshtein import distance as lev
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rdkit import Chem, DataStructs, RDLogger
from collections import Counter
from rdkit.Chem import MACCSkeys, AllChem

RDLogger.DisableLog('rdApp.*')
nltk.download('wordnet')


def exact_match(ot_smi, gt_smi):
    m_out = Chem.MolFromSmiles(ot_smi)
    m_gt = Chem.MolFromSmiles(gt_smi)

    try:
        if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt):
            return 1
    except:
        pass
    return 0


def maccs_similarity(ot_m, gt_m):
    return DataStructs.FingerprintSimilarity(
        MACCSkeys.GenMACCSKeys(gt_m), 
        MACCSkeys.GenMACCSKeys(ot_m), 
        metric=DataStructs.TanimotoSimilarity
    )


def morgan_similarity(ot_m, gt_m, radius=2):
    return DataStructs.TanimotoSimilarity(
        AllChem.GetMorganFingerprint(gt_m, radius), 
        AllChem.GetMorganFingerprint(ot_m, radius)
    )


def rdk_similarity(ot_m, gt_m):
    return DataStructs.FingerprintSimilarity(
        Chem.RDKFingerprint(gt_m), 
        Chem.RDKFingerprint(ot_m), 
        metric=DataStructs.TanimotoSimilarity
    )


class Evaluator(ABC):

    @abstractmethod
    def build_evaluate_tuple(self, pred, gt):
        pass

    @abstractmethod
    def evaluate(self, predictions, references, metrics: List[str] = None, verbose: bool = False, full_results: bool = False):
        pass


class MoleculeSMILESEvaluator(Evaluator):
    _metric_functions = {
        "exact_match": exact_match,
        "bleu": corpus_bleu,
        "levenshtein": lev,
        "rdk_sims": rdk_similarity,
        "maccs_sims": maccs_similarity,
        "morgan_sims": morgan_similarity,
        "validity": lambda smiles: smiles is not None,
    }

    @staticmethod
    def sf_decode(selfies):
        try:
            smiles = sf.decoder(selfies)
            return smiles
        except Exception:
            return None

    @staticmethod
    def convert_to_canonical_smiles(smiles):
        if not smiles:
            return None
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=True, canonical=True)
            return canonical_smiles
        else:
            return None

    def build_evaluate_tuple(self, pred, gt, decode_selfies=False):
        if decode_selfies:
            pred = self.sf_decode(pred)
            gt = self.sf_decode(gt)
        return self.convert_to_canonical_smiles(pred), self.convert_to_canonical_smiles(gt)

    def evaluate(self, predictions, references, metrics: List[str] = None, verbose: bool = False, decode_selfies: bool = False, full_results: bool = False):
        # if all references are "", calculate number of predictions that are ""
        if all(ref == "" for ref in references):
            acc = sum(pred == "" for pred in predictions) / len(predictions)
            return {"accuracy": acc}

        assert len(predictions) == len(references), "Predictions and references must have the same length."
        assert all(ref != "" for ref in references), "All references must be non-empty strings."
            
        if metrics is None:
            metrics = ["exact_match", "bleu", "levenshtein", "rdk_sims", "maccs_sims", "morgan_sims", "validity"]

        results = {metric: [] for metric in metrics}
        if "bleu" in metrics:
            results["bleu"] = [[], []]

        for pred, gt in zip(predictions, references):
            pred, gt = self.build_evaluate_tuple(pred, gt, decode_selfies=decode_selfies)

            for metric in metrics:
                if metric == "bleu" and pred and gt:
                    gt_tokens = [c for c in gt]
                    pred_tokens = [c for c in pred]
                    results[metric][0].append([gt_tokens])
                    results[metric][1].append(pred_tokens)
                elif pred is None or gt is None:
                    results[metric].append(0)
                    continue
                elif metric == "validity":
                    results[metric].append(self._metric_functions[metric](pred))
                elif metric in ["maccs_sims", "morgan_sims", "rdk_sims"]:
                    results[metric].append(self._metric_functions[metric](Chem.MolFromSmiles(pred), Chem.MolFromSmiles(gt)))
                else:
                    results[metric].append(self._metric_functions[metric](pred, gt))

        if "bleu" in metrics:
            if results["bleu"][0] and results["bleu"][1]:
                results["bleu"] = corpus_bleu(results["bleu"][0], results["bleu"][1])
            else:
                results["bleu"] = 0

        if verbose:
            print("Evaluation results:")
            for metric, values in results.items():
                print(f"{metric}: {np.mean(values)}")

        if full_results:
            return results
        else:
            return {metric: np.mean(values) for metric, values in results.items()}

    def evaluate_top_m(self, predictions, references, metrics: List[str] = None, verbose: bool = False, decode_selfies: bool = False, full_results: bool = False):
        if all(ref == "" for ref in references):
            acc = sum(p == "" for pred_list in predictions for p in pred_list) / sum(len(pred_list) for pred_list in predictions)
            return {"accuracy": acc}

        assert len(predictions) == len(references), "Predictions and references must have the same length."
        assert all(ref != "" for ref in references), "All references must be non-empty strings."

        if metrics is None:
            metrics = ["exact_match", "bleu", "levenshtein", "rdk_sims", "maccs_sims", "morgan_sims", "validity"]

        results = {metric: [] for metric in metrics}

        for preds, gt in zip(predictions, references):
            instance_results = {metric: [] for metric in metrics}
            _, canonical_gt = self.build_evaluate_tuple(None, gt, decode_selfies=decode_selfies)

            if canonical_gt is None:
                for metric in metrics:
                    results[metric].append(0)
                continue

            for pred in preds:
                canonical_pred, _ = self.build_evaluate_tuple(pred, gt, decode_selfies=decode_selfies)
                
                if canonical_pred is None:
                    for metric in metrics:
                        instance_results[metric].append(0)
                    continue

                for metric in metrics:
                    if metric == "validity":
                        instance_results[metric].append(self._metric_functions[metric](canonical_pred))
                    elif metric in ["maccs_sims", "morgan_sims", "rdk_sims"]:
                        pred_mol = Chem.MolFromSmiles(canonical_pred)
                        gt_mol = Chem.MolFromSmiles(canonical_gt)
                        if pred_mol and gt_mol:
                            instance_results[metric].append(self._metric_functions[metric](pred_mol, gt_mol))
                        else:
                            instance_results[metric].append(0)
                    elif metric == "bleu":
                        gt_tokens = list(canonical_gt)
                        pred_tokens = list(canonical_pred)
                        instance_results[metric].append(corpus_bleu([[gt_tokens]], [pred_tokens], smoothing_function=SmoothingFunction().method1))
                    else:
                        instance_results[metric].append(self._metric_functions[metric](canonical_pred, canonical_gt))
            
            for metric in metrics:
                if not instance_results[metric]:
                    results[metric].append(0)
                    continue

                if metric == "levenshtein":
                    results[metric].append(min(instance_results[metric]))
                else:
                    results[metric].append(max(instance_results[metric]))
        
        if verbose:
            print("Top-M Evaluation results:")
            for metric, values in results.items():
                print(f"{metric}: {np.mean(values)}")

        if full_results:
            return results
        else:
            return {metric: np.mean(values) for metric, values in results.items()}

    def evaluate_majority_vote(self, predictions, references, metrics: List[str] = None, verbose: bool = False, decode_selfies: bool = False, full_results: bool = False):
        """
        Majority-vote evaluation:
        - predictions: List[List[str]] (각 인스턴스별 여러 후보)
        - references: List[str] (각 인스턴스 정답)
        - metrics: list of metric names (기본은 기존 evaluate_top_m와 동일)
        Returns mean metrics (or full per-instance results if full_results=True).
        Tie-breaker: 가장 먼저 등장한 후보를 선택.
        """
        # Special-case: all references empty -> compute accuracy on majority-chosen preds being ""
        if all(ref == "" for ref in references):
            chosen_preds = []
            for preds in predictions:
                if not preds:
                    chosen_preds.append("")
                    continue
                counts = Counter(preds)
                max_count = max(counts.values())
                candidates = {p for p, c in counts.items() if c == max_count}
                # choose earliest occurrence among candidates
                chosen = None
                for p in preds:
                    if p in candidates:
                        chosen = p
                        break
                chosen_preds.append(chosen if chosen is not None else "")
            acc = sum(p == "" for p in chosen_preds) / len(chosen_preds) if chosen_preds else 0.0
            return {"accuracy": acc}

        assert len(predictions) == len(references), "Predictions and references must have the same length."
        assert all(ref != "" for ref in references), "All references must be non-empty strings."

        if metrics is None:
            metrics = ["exact_match", "bleu", "levenshtein", "rdk_sims", "maccs_sims", "morgan_sims", "validity"]

        results = {metric: [] for metric in metrics}
        full_per_instance = {metric: [] for metric in metrics} if full_results else None

        for preds, gt in zip(predictions, references):
            # choose majority prediction (mode). tie -> first occurrence among tied candidates.
            if not preds:
                chosen_pred = ""
            else:
                counts = Counter(preds)
                max_count = max(counts.values())
                candidates = {p for p, c in counts.items() if c == max_count}
                chosen_pred = None
                for p in preds:
                    if p in candidates:
                        chosen_pred = p
                        break
                if chosen_pred is None:
                    chosen_pred = ""  # fallback

            instance_values = {}

            # canonicalize ground truth once
            _, canonical_gt = self.build_evaluate_tuple(None, gt, decode_selfies=decode_selfies)
            if canonical_gt is None:
                # cannot evaluate this instance
                for metric in metrics:
                    instance_values[metric] = 0
                    results[metric].append(0)
                    if full_results:
                        full_per_instance[metric].append(0)
                continue

            canonical_pred, _ = self.build_evaluate_tuple(chosen_pred, gt, decode_selfies=decode_selfies)
            if canonical_pred is None:
                for metric in metrics:
                    instance_values[metric] = 0
                    results[metric].append(0)
                    if full_results:
                        full_per_instance[metric].append(0)
                continue

            for metric in metrics:
                if metric == "validity":
                    val = self._metric_functions[metric](canonical_pred)
                    instance_values[metric] = val
                elif metric in ["maccs_sims", "morgan_sims", "rdk_sims"]:
                    pred_mol = Chem.MolFromSmiles(canonical_pred)
                    gt_mol = Chem.MolFromSmiles(canonical_gt)
                    if pred_mol and gt_mol:
                        val = self._metric_functions[metric](pred_mol, gt_mol)
                    else:
                        val = 0
                    instance_values[metric] = val
                elif metric == "bleu":
                    gt_tokens = list(canonical_gt)
                    pred_tokens = list(canonical_pred)
                    val = corpus_bleu([[gt_tokens]], [pred_tokens], smoothing_function=SmoothingFunction().method1)
                    instance_values[metric] = val
                else:
                    # exact_match, levenshtein, etc. 일반적인 함수 호출
                    instance_values[metric] = self._metric_functions[metric](canonical_pred, canonical_gt)

                results[metric].append(instance_values[metric])
                if full_results:
                    full_per_instance[metric].append(instance_values[metric])

        if verbose:
            print("Majority-vote Evaluation results:")
            for metric, values in results.items():
                print(f"{metric}: {np.mean(values)}")

        if full_results:
            return full_per_instance
        else:
            return {metric: np.mean(values) for metric, values in results.items()}

    def calcualte_answer_reward(self, predictions, references, decode_selfies=False):
        # Calculate score for each prediction, the score is the exact match score plus rdk_sims score
        scores = []
        for pred, gt in zip(predictions, references):
            score_batch = []
            for p, g in zip(pred, gt):
                p, g = self.build_evaluate_tuple(p, g, decode_selfies=decode_selfies)
                if p is None or g is None:
                    score = 0
                else:
                    score = exact_match(p, g) + rdk_similarity(Chem.MolFromSmiles(p), Chem.MolFromSmiles(g))
                score_batch.append(score)
            scores.append(score_batch)
        return scores
