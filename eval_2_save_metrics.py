"""
Compute Exact@1, Round-trip@1, Exact@100, Round-trip@100, Feasible Ratio, and
Template Diversity for RetroReasoner (RL) on the main in-distribution test
set, given the outputs saved by eval_1_save_outputs.py.

Prerequisites:
    Serve the round-trip model with vLLM's OpenAI-compatible API, e.g.:
        vllm serve KU-AGI/RetroReasoner-RoundTrip-8B --port 8090
    (see scripts/serve_vllm.sh for a multi-GPU example)

    Template counting uses the vendored localmapper/ package (a modified fork
    of https://pypi.org/project/localmapper/, kept in this repo since it
    returns atom-mapping templates as a dict rather than a flat string) —
    install its dependencies: torch, dgl, dgllife (see requirements.txt).

Environment variables:
    ROUNDTRIP_MODEL_NAME    Model name as registered with vLLM
                            (default: KU-AGI/RetroReasoner-RoundTrip-8B)
    ROUNDTRIP_BASE_URLS     Comma-separated "host:port" list of vLLM servers to
                            load-balance across (default: localhost:8090)
    ROUNDTRIP_PARALLELISM_PER_INSTANCE
                            Max concurrent round-trip calls per test instance in
                            sampling mode (default: min(3, num servers))
    RETRO_TEST_DATA_PATH    Local path to main-evaluation.json (same as eval_1_save_outputs.py)
    HF_DATASET_REPO_ID      HF dataset repo id (default: KU-AGI/RetroReasoner-data)
    HF_DATASET_FILENAME     Path within the dataset repo (default: testset/main-evaluation.json)
    LOCALMAPPER_DEVICE      Device for localmapper's atom-mapping model (default: cuda, i.e.
                            physical GPU 0). If a round-trip vLLM replica is also serving on
                            that GPU, point this elsewhere (e.g. cuda:4) to avoid silent
                            contention-induced template undercounting — see the comment
                            above LOCALMAPPER_DEVICE in this file.

Reads/writes in place:
    outputs/retro_test/RetroReasoner(RL)_temp{temperature}.json
"""
import json
import multiprocessing as mp
import os
import re
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import cycle

from openai import OpenAI
from rdkit import Chem, RDLogger
from tqdm import tqdm

from evaluator.smiles_evaluator import MoleculeSMILESEvaluator
from localmapper import localmapper

RDLogger.DisableLog('rdApp.*')

MODEL_LABEL = "RetroReasoner(RL)"
OUTPUT_ROOT = "outputs/retro_test"

# localmapper always binds to this device regardless of CUDA_VISIBLE_DEVICES scoping
# elsewhere in the process. If a vLLM round-trip replica is also serving on this same
# physical GPU, the 30 concurrent localmapper worker processes below can silently fail
# under memory/compute contention (swallowed by the except-clause in
# get_canonical_template), undercounting templates without any visible error. Point this
# at a GPU that isn't hosting a round-trip vLLM replica, or free up headroom on it.
LOCALMAPPER_DEVICE = os.environ.get("LOCALMAPPER_DEVICE", "cuda")

# ---------------------------
# Multi vLLM round-trip config
# ---------------------------
ROUNDTRIP_BASE_URLS = [
    f"http://{ip_port.strip()}/v1"
    for ip_port in os.environ.get("ROUNDTRIP_BASE_URLS", "localhost:8090").split(",")
]

ROUNDTRIP_MODEL_NAME = os.environ.get("ROUNDTRIP_MODEL_NAME", "KU-AGI/RetroReasoner-RoundTrip-8B")

ROUNDTRIP_PARALLELISM_PER_INSTANCE = int(os.environ.get(
    "ROUNDTRIP_PARALLELISM_PER_INSTANCE",
    str(min(3, len(ROUNDTRIP_BASE_URLS)))
))

# ---------------------------
# Lazy singletons per process
# ---------------------------
_MOLECULE_EVALUATOR = None
_MAPPER = None


def _get_molecule_evaluator() -> MoleculeSMILESEvaluator:
    global _MOLECULE_EVALUATOR
    if _MOLECULE_EVALUATOR is None:
        _MOLECULE_EVALUATOR = MoleculeSMILESEvaluator()
    return _MOLECULE_EVALUATOR


def _get_mapper():
    """
    Lazy initialization of localmapper.
    GPU/initialization cost is high, so only initialize when templates are needed in sampling mode.
    """
    global _MAPPER
    if _MAPPER is None:
        _MAPPER = localmapper(device=LOCALMAPPER_DEVICE)
    return _MAPPER


class RoundtripEndpointPool:
    """Round-robin selection of base_url within a process."""
    def __init__(self, base_urls):
        self.base_urls = list(base_urls)
        if not self.base_urls:
            raise ValueError("ROUNDTRIP_BASE_URLS is empty.")
        self._it = cycle(self.base_urls)
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            return next(self._it)


_ROUNDTRIP_POOL = RoundtripEndpointPool(ROUNDTRIP_BASE_URLS)

_thread_local = threading.local()


def _get_openai_client(base_url: str) -> OpenAI:
    clients = getattr(_thread_local, "clients", None)
    if clients is None:
        clients = {}
        _thread_local.clients = clients
    if base_url not in clients:
        clients[base_url] = OpenAI(base_url=base_url, api_key="EMPTY")
    return clients[base_url]


def get_canonical_template(reactant: str, product: str) -> str:
    try:
        mapper = _get_mapper()
        result = mapper.get_atom_map(f"{reactant}>>{product}", return_dict=True)
        return result['template']['canonical']
    except Exception as e:
        # Note: a spike in these warnings (rather than occasional RDKit/atom-mapping
        # failures on genuinely malformed candidates) usually means LOCALMAPPER_DEVICE
        # is contending with a vLLM replica on the same GPU — see the comment above
        # LOCALMAPPER_DEVICE.
        print(f"[get_canonical_template] failed on {reactant!r} >> {product!r}: {e}", file=sys.stderr)
        return None


def parse_raw_response(raw_response: str):
    reasoning_pattern = r"<think>(.*?)</think>"
    smiles_pattern = r"<ANSWER>(.*?)</ANSWER>"

    reasoning_match = re.search(reasoning_pattern, raw_response, re.DOTALL)
    smiles_match = re.search(smiles_pattern, raw_response, re.DOTALL)

    reasoning_steps = reasoning_match.group(1).strip() if reasoning_match else ""
    smiles = smiles_match.group(1).strip() if smiles_match else ""
    return reasoning_steps, smiles


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


def _call_roundtrip(reactant: str) -> str:
    system_prompt = "You are a chemist."
    user_prompt = (
        f"{reactant} Considering the given starting materials, "
        f"what might be the resulting product in a chemical reaction?"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = _call_chat_completion(
        base_urls=ROUNDTRIP_BASE_URLS,
        client_model_name=ROUNDTRIP_MODEL_NAME,
        messages=messages,
        max_tokens=500,
        temperature=0.0,
        n=1,
    )
    try:
        raw_response = response.choices[0].text.strip()
    except Exception:
        raw_response = response.choices[0].message.content.strip()

    _, roundtrip_pred_product = parse_raw_response(raw_response)
    return roundtrip_pred_product


def _call_chat_completion(
    *,
    base_urls: list,
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

    if temperature == 0.0:
        n = 1  # deterministic

    last_err = None

    start_url = _ROUNDTRIP_POOL.next()
    ordered_urls = [start_url] + [u for u in base_urls if u != start_url]

    for attempt in range(max_retries):
        for url in ordered_urls:
            try:
                client = _get_openai_client(url)
                response = client.chat.completions.create(
                    model=client_model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=n,
                )
                return response
            except Exception as e:
                last_err = e
                continue

        time.sleep(retry_sleep_sec * (2 ** attempt))

    raise RuntimeError(f"Request failed after {max_retries} retries across servers. last_err={last_err}")


def _call_roundtrip_many(reactants: list, max_workers: int) -> list:
    """
    Parallel roundtrip for multiple predicted_smiles within one instance in sampling mode.
    max_workers is limited according to the number of servers and load.
    """
    if not reactants:
        return []

    mw = max(1, min(max_workers, len(reactants)))
    out = [None] * len(reactants)

    with ThreadPoolExecutor(max_workers=mw) as ex:
        fut_to_i = {ex.submit(_call_roundtrip, r): i for i, r in enumerate(reactants)}
        for fut in as_completed(fut_to_i):
            i = fut_to_i[fut]
            try:
                out[i] = fut.result()
            except Exception:
                out[i] = ""  # On failure, empty string (treated as exact match failure in original logic)

    return out


def process_one_instance_greedy(idx: int, test_d: dict, dump_d: dict):
    me = _get_molecule_evaluator()
    dump_d['predicted_smiles'][0] = me.convert_to_canonical_smiles(dump_d['predicted_smiles'][0])

    return idx, \
        dump_d['ground_truth_smiles'], \
        dump_d['input_smiles'], \
        dump_d['predicted_smiles'][0], \
        _call_roundtrip(dump_d['predicted_smiles'][0])


def process_one_instance_sampling(idx: int, test_d: dict, dump_d: dict):
    me = _get_molecule_evaluator()

    # 1) Canonicalize first
    canon_smiles = [me.convert_to_canonical_smiles(smi) for smi in dump_d['predicted_smiles']]

    # 2) Call roundtrip with limited parallelism within the instance
    roundtrip_preds_for_all = _call_roundtrip_many(
        canon_smiles,
        max_workers=ROUNDTRIP_PARALLELISM_PER_INSTANCE
    )

    # 3) Calculate templates only for feasible ones
    templates = []
    for smi, roundtrip_pred in zip(canon_smiles, roundtrip_preds_for_all):
        if _exact_at_k([roundtrip_pred], dump_d['input_smiles']) == 1:
            template = get_canonical_template(smi, dump_d['input_smiles'])
            if template is not None:
                templates.append(template)

    templates = list(set(templates))  # unique

    return idx, \
        dump_d['ground_truth_smiles'], \
        dump_d['input_smiles'], \
        canon_smiles, \
        roundtrip_preds_for_all, \
        templates


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
    """Evaluate RetroReasoner (RL) predictions on the main in-distribution retrosynthesis test set."""
    test_data = _load_test_data()
    summary = {}

    for temperature in [0.0, 1.2]:
        dump_data_path = os.path.join(OUTPUT_ROOT, f"{MODEL_LABEL}_temp{temperature}.json")

        try:
            with open(dump_data_path, "r") as f:
                dump_data = json.load(f)
        except Exception as e:
            print(f"Error loading dump data for temperature={temperature}: {e}")
            continue

        assert len(test_data) == len(dump_data['results']), \
            f"len(test_data)={len(test_data)} != len(dump_data)={len(dump_data['results'])}"

        temperature = dump_data['temperature']

        molecule_evaluator = MoleculeSMILESEvaluator()

        reactant_preds = [None] * len(test_data)
        reactant_gts = [None] * len(test_data)
        product_roundtrip_preds = [None] * len(test_data)
        product_gts = [None] * len(test_data)
        templates_all = [None] * len(test_data)

        executor = ProcessPoolExecutor(max_workers=30, mp_context=mp.get_context("spawn"))
        futures = []
        for idx, (test_d, dump_d) in enumerate(zip(test_data, dump_data['results'])):
            process_func = process_one_instance_greedy if temperature == 0.0 else process_one_instance_sampling
            futures.append(executor.submit(process_func, idx, test_d, dump_d))

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Evaluating results {dump_data_path}",
        ):
            if temperature == 0.0:
                idx, reactant_gt, product_gt, reactant_pred, product_roundtrip_pred = fut.result()
                reactant_gts[idx] = reactant_gt
                product_gts[idx] = product_gt
                reactant_preds[idx] = [reactant_pred]
                product_roundtrip_preds[idx] = [product_roundtrip_pred]
            else:
                idx, reactant_gt, product_gt, reactant_pred, product_roundtrip_pred, templates = fut.result()
                reactant_gts[idx] = reactant_gt
                product_gts[idx] = product_gt
                reactant_preds[idx] = reactant_pred
                product_roundtrip_preds[idx] = product_roundtrip_pred
                templates_all[idx] = templates

        executor.shutdown(wait=True, cancel_futures=False)

        if temperature == 0.0:
            molecule_eval_results = molecule_evaluator.evaluate_top_m(
                predictions=reactant_preds,
                references=reactant_gts,
                metrics=["exact_match"],
            )
            roundtrip_eval_results = molecule_evaluator.evaluate_top_m(
                predictions=product_roundtrip_preds,
                references=product_gts,
                metrics=["exact_match"],
            )

            summary["Exact@1"] = molecule_eval_results['exact_match']
            summary["Round-trip@1"] = roundtrip_eval_results['exact_match']

            dump_data['exact@1'] = round(molecule_eval_results['exact_match'], 3)
            dump_data['Round-trip@1'] = round(roundtrip_eval_results['exact_match'], 3)

        else:
            exact_at_k_scores = []
            for ot_smi, gt_smi in zip(reactant_preds, reactant_gts):
                exact_at_k_scores.append(_exact_at_k(ot_smi, gt_smi))
            exact_at_total_n = sum(exact_at_k_scores) / len(exact_at_k_scores)

            exact_at_k_scores_roundtrip = []
            for ot_smi, gt_smi in zip(product_roundtrip_preds, product_gts):
                exact_at_k_scores_roundtrip.append(_exact_at_k(ot_smi, gt_smi))
            exact_at_total_n_roundtrip = sum(exact_at_k_scores_roundtrip) / len(exact_at_k_scores_roundtrip)

            feasible_ratio_list = []
            for ot_smi, gt_smi in zip(product_roundtrip_preds, product_gts):
                feasible_list = []
                for smi in ot_smi:
                    feasible_list.append(_exact_at_k([smi], gt_smi))
                feasible_ratio_list.append(sum(feasible_list) / len(feasible_list))
            feasible_ratio = sum(feasible_ratio_list) / len(feasible_ratio_list)

            avg_template_count = sum(len(tpls) for tpls in templates_all) / len(templates_all)

            n = len(reactant_preds[0])
            summary[f"Exact@{n}"] = exact_at_total_n
            summary[f"Round-trip@{n}"] = exact_at_total_n_roundtrip
            summary["Feasible Ratio"] = feasible_ratio
            summary["Template Diversity"] = avg_template_count

            dump_data[f'exact@{n}'] = round(exact_at_total_n, 3)
            dump_data[f'Round-trip@{n}'] = round(exact_at_total_n_roundtrip, 3)
            dump_data['feasible_ratio'] = round(feasible_ratio, 3)
            dump_data['avg_template_count'] = round(avg_template_count, 3)

        with open(dump_data_path, "w") as f:
            json.dump(dump_data, f, indent=4)

    print()
    print(f"{MODEL_LABEL} — main in-distribution test set")
    for key in ["Exact@1", "Round-trip@1", "Exact@100", "Round-trip@100", "Feasible Ratio", "Template Diversity"]:
        if key in summary:
            print(f"{key}: {summary[key]:.3f}")


if __name__ == "__main__":
    main()
