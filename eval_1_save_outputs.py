"""
Generate RetroReasoner (RL) retrosynthesis predictions on the main
in-distribution test set and save them to disk.

Prerequisites:
    Serve the model with vLLM's OpenAI-compatible API, e.g.:
        vllm serve KU-AGI/RetroReasoner-RL --port 8000
    (see scripts/serve_vllm.sh for a multi-GPU example)

Environment variables:
    VLLM_MODEL_NAME     Model name as registered with vLLM (default: KU-AGI/RetroReasoner-RL)
    VLLM_BASE_URLS      Comma-separated "host:port" list of vLLM servers to load-balance
                         across (default: localhost:8000)
    RETRO_TEST_DATA_PATH
                         Local path to main-evaluation.json. If unset, the file is downloaded
                         from the Hugging Face Hub dataset repo (see HF_DATASET_REPO_ID /
                         HF_DATASET_FILENAME below).
    HF_DATASET_REPO_ID   HF dataset repo id (default: KU-AGI/RetroReasoner-data)
    HF_DATASET_FILENAME  Path within the dataset repo (default: testset/main-evaluation.json)

Output:
    outputs/retro_test/RetroReasoner(RL)_temp{temperature}.json
"""
import math
import os
import re
import time
from decimal import Decimal
from functools import lru_cache

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import json
from openai import OpenAI
from tqdm import tqdm
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

MODEL_LABEL = "RetroReasoner(RL)"

VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "KU-AGI/RetroReasoner-RL")
VLLM_BASE_URLS = [
    f"http://{ip_port.strip()}/v1"
    for ip_port in os.environ.get("VLLM_BASE_URLS", "localhost:8000").split(",")
]

MAX_TOKENS = 3000
TOTAL_TEST_SAMPLES = 500  # full main in-distribution test set
TOTAL_N = 100             # candidates sampled per instance at temperature > 0
CHUNK_N = 20              # candidates requested per vLLM call
TEMPERATURES = [0.0, 1.2]  # 0.0 -> greedy Exact@1, 1.2 -> sampling Exact@100 / Round-trip@100

OUTPUT_ROOT = "outputs/retro_test"


def parse_raw_response(raw_response: str):
    reasoning_pattern = r"<think>(.*?)</think>"
    smiles_pattern = r"<ANSWER>(.*?)</ANSWER>"

    reasoning_match = re.search(reasoning_pattern, raw_response, re.DOTALL)
    smiles_match = re.search(smiles_pattern, raw_response, re.DOTALL)

    reasoning_steps = reasoning_match.group(1).strip() if reasoning_match else ""
    smiles = smiles_match.group(1).strip() if smiles_match else ""
    return reasoning_steps, smiles


@lru_cache(maxsize=None)
def _get_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="EMPTY")


def _call_chat_completion(
    *,
    base_url: str,
    client_model_name: str,
    messages: list,
    max_tokens: int,
    temperature: float,
    max_retries: int = 3,
    retry_sleep_sec: float = 0.5,
    n: int = 1,
):
    if n > 500:
        raise ValueError(f"Per-request n must be <= 500. got n={n}")

    client = _get_client(base_url)

    if temperature == 0.0:
        n = 1  # deterministic

    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=client_model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
                extra_body={"skip_special_tokens": False},
            )
            return response
        except Exception as e:
            last_err = e
            time.sleep(retry_sleep_sec * (2 ** attempt))

    raise RuntimeError(f"Request failed after {max_retries} retries. last_err={last_err}")


def _worker_one(
    idx: int,
    d: dict,
    system_prompt: str,
    retro_user_prompt_template: str,
    base_url: str,
    client_model_name: str,
    max_tokens: int,
    temperature: float,
    total_n: int,
    chunk_n: int,
):
    input_smiles = ".".join(d["products"])
    user_prompt = retro_user_prompt_template.replace("[SMILES]", input_smiles)
    gt = ".".join(d["reactants"])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    reasoning_pred, smiles_pred = [], []
    rounds = math.ceil(total_n / chunk_n)

    for r in range(rounds):
        remain = total_n - r * chunk_n
        cur_n = chunk_n if remain >= chunk_n else remain
        if cur_n <= 0:
            break
        response = _call_chat_completion(
            base_url=base_url,
            client_model_name=client_model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            n=cur_n,
        )

        try:
            raw_response_list = [c.text.strip() for c in response.choices]
        except Exception:
            raw_response_list = [c.message.content.strip() for c in response.choices]
        for raw_response in raw_response_list:
            rr, smi = parse_raw_response(raw_response)
            reasoning_pred.append(rr)
            smiles_pred.append(smi)

    try:
        product_str = d['info']['product_str']
    except KeyError:
        product_str = ".".join(d["products"])
    return idx, reasoning_pred, product_str, smiles_pred, gt


def _exact_at_k(ot_smi, gt_smi):
    m_gt = Chem.MolFromSmiles(gt_smi)
    if m_gt is None:
        return 0
    try:
        gt_inchi = Chem.MolToInchi(m_gt)
    except Exception:
        return 0

    candidates = ot_smi if isinstance(ot_smi, (list, tuple)) else [ot_smi]

    for smi in candidates:
        if not smi:
            continue
        m_out = Chem.MolFromSmiles(smi)
        if m_out is None:
            continue
        try:
            if Chem.MolToInchi(m_out) == gt_inchi:
                return 1
        except Exception:
            continue
    return 0


def _first_exact_match_rank(pred_smiles_list, gt_smi):
    m_gt = Chem.MolFromSmiles(gt_smi)
    if m_gt is None:
        return None
    try:
        gt_inchi = Chem.MolToInchi(m_gt)
    except Exception:
        return None

    candidates = pred_smiles_list if isinstance(pred_smiles_list, (list, tuple)) else [pred_smiles_list]

    for i, smi in enumerate(candidates, start=1):  # 1-indexed
        if not smi:
            continue
        m_out = Chem.MolFromSmiles(smi)
        if m_out is None:
            continue
        try:
            if Chem.MolToInchi(m_out) == gt_inchi:
                return i
        except Exception:
            continue
    return None


def _exact_curve_from_hit_ranks(hit_ranks, total_n):
    N = len(hit_ranks)
    if N == 0:
        return [0.0] * total_n

    counts = [0] * (total_n + 1)
    for r in hit_ranks:
        if r is None:
            continue
        if 1 <= r <= total_n:
            counts[r] += 1

    curve = [0.0] * total_n
    cum = 0
    for k in range(1, total_n + 1):
        cum += counts[k]
        curve[k - 1] = cum / N
    return curve


def _temp_str(t: float) -> str:
    return f"{Decimal(str(t)).quantize(Decimal('0.0'))}"


def _load_test_data():
    local_path = os.environ.get("RETRO_TEST_DATA_PATH")
    if local_path:
        path = local_path
    else:
        from huggingface_hub import hf_hub_download
        repo_id = os.environ.get("HF_DATASET_REPO_ID", "KU-AGI/RetroReasoner-data")
        filename = os.environ.get("HF_DATASET_FILENAME", "testset/main-evaluation.json")
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

    with open(path, "r") as f:
        return json.load(f)


def main():
    system_prompt = "You are a chemist."
    retro_user_prompt_template = (
        "With the given product [SMILES], suggest some likely reactants that were used in its synthesis."
    )

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    test_data = _load_test_data()[:TOTAL_TEST_SAMPLES]

    for temperature in TEMPERATURES:
        reasoning_preds = [None] * len(test_data)
        smiles_inputs = [None] * len(test_data)
        smiles_preds = [None] * len(test_data)
        smiles_gts = [None] * len(test_data)

        executors = []
        try:
            executors = [
                ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"))
                for _ in VLLM_BASE_URLS
            ]

            futures = []
            for idx, d in enumerate(test_data):
                ex_i = idx % len(VLLM_BASE_URLS)
                base_url = VLLM_BASE_URLS[ex_i]
                fut = executors[ex_i].submit(
                    _worker_one,
                    idx,
                    d,
                    system_prompt,
                    retro_user_prompt_template,
                    base_url,
                    VLLM_MODEL_NAME,
                    MAX_TOKENS,
                    temperature if temperature > 0.0 else 0.0,
                    TOTAL_N if temperature > 0.0 else 1,
                    CHUNK_N if temperature > 0.0 else 1,
                )
                futures.append(fut)

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"retro, retro_test, {MODEL_LABEL}, temp={temperature}",
            ):
                idx, reasoning_pred, smiles_input, smiles_pred, smiles_gt = fut.result()
                reasoning_preds[idx] = reasoning_pred
                smiles_inputs[idx] = smiles_input
                smiles_preds[idx] = smiles_pred
                smiles_gts[idx] = smiles_gt

        finally:
            for ex in executors:
                ex.shutdown(wait=True, cancel_futures=True)

        # Exact@total_n
        exact_at_k_scores = []
        for ot_smi, gt_smi in zip(smiles_preds, smiles_gts):
            exact_at_k_scores.append(_exact_at_k(ot_smi, gt_smi))
        exact_at_total_n = sum(exact_at_k_scores) / len(exact_at_k_scores)

        # Exact@k curve (k=1..total_n)
        hit_ranks = []
        for pred_list, gt in zip(smiles_preds, smiles_gts):
            hit_ranks.append(_first_exact_match_rank(pred_list, gt))
        exact_curve = _exact_curve_from_hit_ranks(hit_ranks, TOTAL_N)

        results = []
        for i in range(len(test_data)):
            results.append({
                "input_smiles": smiles_inputs[i],
                "predicted_smiles": smiles_preds[i],
                "ground_truth_smiles": smiles_gts[i],
                "reasoning_pred": reasoning_preds[i],
            })

        data_to_save = {
            "model": MODEL_LABEL,
            "task": "retro",
            "temperature": float(_temp_str(temperature)),
            "num_samples": len(test_data),
            f"exact@{TOTAL_N}": exact_at_total_n,
            f"exact_curve_1_to_{TOTAL_N}": exact_curve,
            "results": results,
        }

        out_path = os.path.join(OUTPUT_ROOT, f"{MODEL_LABEL}_temp{temperature}.json")
        with open(out_path, "w") as wf:
            json.dump(data_to_save, wf, indent=4)

        print()
        print("=" * 100)
        print(f"Task: retro")
        print(f"Path tag: retro_test (main in-distribution test set)")
        print(f"Temperature: {temperature}")
        print(f"Exact@{TOTAL_N}: {exact_at_total_n:.3f}")
        print("=" * 100)


if __name__ == "__main__":
    main()
