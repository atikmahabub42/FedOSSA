#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
fedossa_core.py -- shared engine for the FedOSSA experiments.

Federated LoRA fine tuning of LLMs with
    Centralized      pooled data, no federation (reference)
    FedAvg / FedIT   LoRA factors A and B averaged separately (FedIT = FedAvg on LoRA)
    FedPTuning       federated prompt tuning (20 soft prompt tokens, averaged)
    FedASK           randomized sketch (rank r) of the product space aggregate
    FedOSSA          product space aggregate -> optimal singular value shrinkage
                     (Gavish & Donoho) calibrated to the noise floor -> rank r refactor
    FedOSSA+SecAgg   FedOSSA under pairwise masking secure aggregation (identical
                     aggregate: the masks cancel in the sum; uplink = dense product)

Studies (each appends to its own CSV, resumable, one row per finished run):
    run_sweep  -> sweep_results.csv + sweep_curves.json   main comparison
    run_dp     -> dp_sweep.csv                            aggregation noise rho
    run_async  -> async.csv                               stragglers phi

The laptop entry points are run_sweep.py, run_dp.py and run_async.py.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback

import numpy as np

# ------------------------------------------------------------------ heavy stack -- #
HAVE_STACK = True
_FAILED = ""
_TB = ""
try:
    _stage = "torch"
    import torch
    _stage = "transformers"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _stage = "peft"
    from peft import (LoraConfig, get_peft_model,
                      get_peft_model_state_dict, set_peft_model_state_dict)
except Exception:
    HAVE_STACK = False
    _FAILED = _stage
    _TB = traceback.format_exc()


def _versions() -> str:
    out = []
    for pkg in ["torch", "transformers", "peft", "datasets", "accelerate",
                "nltk", "numpy", "matplotlib"]:
        try:
            m = __import__(pkg)
            out.append(f"{pkg}={getattr(m, '__version__', '?')}")
        except Exception:
            out.append(f"{pkg}=MISSING")
    return "  ".join(out)


def require_stack():
    """Print a clear install hint and return False if the DL stack is missing."""
    if HAVE_STACK:
        return True
    print("=" * 78)
    print(f" IMPORT FAILED at: {_FAILED}")
    print(" Installed:  " + _versions())
    print("-" * 78)
    print(" Fix (see README.md):")
    print("   python -m pip install torch --index-url https://download.pytorch.org/whl/cu128")
    print("   python -m pip install -r requirements.txt")
    print("-" * 78)
    print(_TB.strip() if _TB else "(no traceback captured)")
    print("=" * 78)
    return False


# ============================ numeric cores ===================================== #
def _svd(M):
    return np.linalg.svd(M, full_matrices=False)


def optimal_shrink(Y, v):
    m, n = Y.shape; tr = False
    if m > n:
        Y = Y.T; m, n = n, m; tr = True
    beta = m / n
    U, s, Vt = _svd(Y)
    if v <= 0:
        X = (U * s) @ Vt; return X.T if tr else X
    scale = v * math.sqrt(n); y = s / scale; cut = 1.0 + math.sqrt(beta)
    x = np.zeros_like(y); big = y > cut
    x[big] = np.sqrt((y[big] ** 2 - beta - 1.0) ** 2 - 4.0 * beta) / y[big]
    X = (U * (x * scale)) @ Vt
    return X.T if tr else X


def recoverable_rank(Y, v):
    m, n = Y.shape
    if m > n:
        m, n = n, m
    beta = m / n
    s = np.linalg.svd(Y, compute_uv=False)
    edge = v * math.sqrt(n) * (1.0 + math.sqrt(beta))
    return int(max(1, np.sum(s > edge)))


def refactor(R, rank):
    U, s, Vt = _svd(R); rank = min(rank, len(s))
    return (U[:, :rank] * s[:rank]), Vt[:rank, :]


def randomized_refactor(M, rank, seed=0):
    """Randomized-SVD low-rank approximation (FedASK-style sketch)."""
    rng = np.random.default_rng(seed)
    m, n = M.shape
    k = min(rank, min(m, n))
    Om = rng.standard_normal((n, k))
    Q, _ = np.linalg.qr(M @ Om)
    Bm = Q.T @ M
    Ub, s, Vt = np.linalg.svd(Bm, full_matrices=False)
    U = Q @ Ub
    return (U[:, :k] * s[:k]), Vt[:k, :]


def aggregate_module(factors, p, method, rank, dp_sigma=0.0, seed=0):
    """Aggregate one adapted module.

    factors: list of (B_k, A_k) LoRA factors (one per client); p: client weights.
    Returns the new global (B, A). `dp_sigma` is the RELATIVE aggregation noise level rho:
    every client upload receives Gaussian noise of std rho x (mean std of the client
    products), so the level is comparable across models. All methods share the same noise
    realization for a given (round, module)."""
    out = factors[0][0].shape[0]; inn = factors[0][1].shape[1]
    prods = [B @ A for (B, A) in factors]
    base_std = float(np.mean([np.std(d) for d in prods])) + 1e-12
    noise_std = dp_sigma * base_std                      # absolute per-entry noise std
    v = noise_std * math.sqrt(float(np.sum(np.asarray(p) ** 2)))   # noise floor nu
    rng = np.random.default_rng(seed)

    # ---- FedAvg / FedIT: average the FACTORS separately (biased product) ----
    if method in ("fedavg", "fedit"):
        rmax = max(B.shape[1] for (B, A) in factors)
        Bs = np.zeros((out, rmax)); As = np.zeros((rmax, inn))
        for (B, A), pk in zip(factors, p):
            r = B.shape[1]
            if noise_std > 0:
                B = B + noise_std * rng.standard_normal(B.shape)
                A = A + noise_std * rng.standard_normal(A.shape)
            Bs[:, :r] += pk * B; As[:r, :] += pk * A
        return refactor(Bs @ As, rank)

    def npz(d):                                          # add noise to a product upload
        return d + noise_std * rng.standard_normal(d.shape) if noise_std > 0 else d

    # ---- product space aggregate G = sum_k p_k (dW_k + E_k) ----
    G = sum(pk * npz(d) for d, pk in zip(prods, p))
    if method in ("fedossa", "fedossa_secagg"):          # SecAgg: masks cancel -> same G
        return refactor(optimal_shrink(G, v), rank)
    if method == "fedask":
        return randomized_refactor(G, rank, seed=seed)
    raise ValueError(f"unknown aggregation method '{method}'")


# ============================ data and models =================================== #
def _load_ds(name, cfg):
    """Robust load: namespaced ids (newest hub), trust_remote_code (script datasets on
    datasets<4), and verification_mode='no_checks' so a dataset that declares a split it
    doesn't actually ship (e.g. Natural Instructions' 'validation') doesn't abort the load."""
    from datasets import load_dataset
    args = (name, cfg) if cfg else (name,)
    for kw in (dict(trust_remote_code=True, verification_mode="no_checks"),
               dict(verification_mode="no_checks"),
               dict(trust_remote_code=True),
               dict()):
        try:
            return load_dataset(*args, **kw)
        except TypeError:                 # this datasets version rejects one of these kwargs
            continue
    return load_dataset(*args)


def lora_targets(hf_id):
    s = hf_id.lower()
    if "gpt2" in s:
        return ["c_attn"]                             # GPT-2 fused QKV
    if "bloom" in s:
        return ["query_key_value"]                    # Bloom fused QKV
    return ["q_proj", "v_proj"]                       # Qwen3 / Llama and most modern decoders


def build_model(hf_id, device, rank, lora_alpha, load_4bit, hf_token):
    """Load an OPEN causal-LM with a LoRA adapter. 4-bit (QLoRA) path is used for
    big models on GPU; otherwise full/half precision."""
    kw = {}
    if hf_token:
        kw["token"] = hf_token
    if load_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb, device_map="auto", **kw)
        model = prepare_model_for_kbit_training(model)
    else:
        dt = torch.bfloat16 if device == "cuda" else torch.float32
        try:                                          # transformers 5.x uses dtype=
            model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dt, **kw)
        except TypeError:                             # older transformers uses torch_dtype=
            model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dt, **kw)
    lcfg = LoraConfig(r=rank, lora_alpha=lora_alpha, target_modules=lora_targets(hf_id),
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    if not (load_4bit and device == "cuda"):
        model = model.to(device)
    return model


def format_instruction(ex, ds):
    if ds == "dolly":
        ctx = ex.get("context", "")
        src = f"Instruction: {ex['instruction']}\n" + (f"Context: {ctx}\n" if ctx else "")
        return src + "Response: ", ex.get("response", "")
    if ds == "alpaca":
        inp = ex.get("input", "")
        src = f"Instruction: {ex['instruction']}\n" + (f"Input: {inp}\n" if inp else "")
        return src + "Output: ", ex.get("output", "")
    if ds == "natinst":
        return f"{ex.get('definition','')}\nInput: {ex.get('inputs','')}\nOutput: ", ex.get("targets", "")
    return ex.get("text", ""), ""


def _partition_indices(m, n_clients, alpha, seed):
    """Assign m item indices to n_clients.
      alpha <= 0 -> IID: shuffle and split evenly (round-robin).
      alpha  > 0 -> non-IID Dirichlet(alpha) QUANTITY skew; because the caller pre-sorts
                    the pool by sequence length, the contiguous chunks handed to each
                    client also differ in LENGTH distribution (feature skew). Smaller
                    alpha => more skew (0.1 hard, 0.5 moderate, >=5 approaches IID)."""
    rng = np.random.default_rng(seed)
    if not alpha or alpha <= 0:
        order = list(range(m)); rng.shuffle(order)
        return [order[k::n_clients] for k in range(n_clients)]
    props = rng.dirichlet([float(alpha)] * n_clients)
    counts = np.maximum(1, np.floor(props * m).astype(int))
    while counts.sum() > m:                               # trim overflow to fit the pool
        counts[int(np.argmax(counts))] -= 1
    shards, start = [], 0
    for c in counts:
        shards.append(list(range(start, min(start + int(c), m)))); start += int(c)
    return shards


def load_clients(ds, tok, n_clients, per_client, max_len, val_size, test_size,
                 gen_eval, seed, alpha_dirichlet):
    """Return (shards, val_ids, test_ids, pairs) from three DISJOINT slices:
    training pool (partitioned non-IID across clients), a validation set used only for
    early stopping, a test set used only for reporting, and (source, reference) pairs
    drawn from the test slice for CIDEr-D."""
    name, cfg = DATASETS[ds]
    raw = _load_ds(name, cfg)
    split = raw["train"] if "train" in raw else raw[list(raw.keys())[0]]
    need = n_clients * per_client + val_size + test_size + 256
    idx = list(range(min(len(split), need + 512)))
    random.Random(seed).shuffle(idx)

    records = []
    for i in idx:
        ex = split[i]
        if ds == "wikitext2":
            t = ex.get("text", "")
            if not t or len(t.strip()) < 32:
                continue
            ids = tok(t, truncation=True, max_length=max_len,
                      return_tensors="pt").input_ids[0]
            src, tgt = None, None
        else:
            src, tgt = format_instruction(ex, ds)
            ids = tok(src + tgt, truncation=True, max_length=max_len,
                      return_tensors="pt").input_ids[0]
        records.append((ids, src, tgt))
        if len(records) >= need:
            break

    nt = n_clients * per_client
    train_rec = records[:nt]
    val_rec = records[nt:nt + val_size]
    test_rec = records[nt + val_size:nt + val_size + test_size]
    val_ids = [r[0] for r in val_rec]
    test_ids = [r[0] for r in test_rec]
    pairs = [(r[1], r[2]) for r in test_rec if r[1] is not None][:200] if gen_eval else []

    pool = [r[0] for r in train_rec]
    order = sorted(range(len(pool)), key=lambda k: pool[k].size(0))
    pool = [pool[k] for k in order]
    parts = _partition_indices(len(pool), n_clients, alpha_dirichlet, seed)
    shards = [[pool[k] for k in part] for part in parts]
    return shards, val_ids, test_ids, pairs


def lora_factor_dict(model):
    """Return {module_key: (B, A)} from current peft LoRA weights (factors, not product)."""
    sd = get_peft_model_state_dict(model)
    keys = {}
    for k in sd:
        if "lora_A" in k:
            base = k.replace("lora_A", "")
            keys.setdefault(base, {})["A"] = sd[k]
        elif "lora_B" in k:
            base = k.replace("lora_B", "")
            keys.setdefault(base, {})["B"] = sd[k]
    out = {}
    for base, ab in keys.items():
        if "A" in ab and "B" in ab:
            A = ab["A"].detach().cpu().float().numpy()
            B = ab["B"].detach().cpu().float().numpy()
            out[base] = (B, A)
    return out


def write_back(model, agg, device):
    sd = get_peft_model_state_dict(model)
    new = {}
    for k in sd:
        if "lora_A" in k:
            base = k.replace("lora_A", "")
            if base in agg:
                new[k] = torch.tensor(agg[base][1], dtype=sd[k].dtype, device=device)
                continue
        if "lora_B" in k:
            base = k.replace("lora_B", "")
            if base in agg:
                new[k] = torch.tensor(agg[base][0], dtype=sd[k].dtype, device=device)
                continue
        new[k] = sd[k]
    set_peft_model_state_dict(model, new)


def train_local(model, batch_ids, tok, device, steps, lr):
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    tot, nb = 0.0, 0
    for _ in range(steps):
        random.shuffle(batch_ids)
        for i in range(0, len(batch_ids), 4):
            chunk = batch_ids[i:i + 4]
            if not chunk:
                continue
            maxlen = max(x.size(0) for x in chunk)
            inp = torch.full((len(chunk), maxlen), tok.pad_token_id, dtype=torch.long)
            for r, x in enumerate(chunk):
                inp[r, :x.size(0)] = x
            inp = inp.to(device)
            out = model(input_ids=inp, labels=inp)
            opt.zero_grad(); out.loss.backward(); opt.step()
            tot += float(out.loss.item()); nb += 1
    return tot / max(1, nb)                      # average local training loss


def eval_loss(model, eval_ids, tok, device):
    """Mean per-token cross-entropy on the held-out set (the 'test loss'). perplexity=exp(this)."""
    model.eval(); nll = 0.0; ntok = 0
    with torch.no_grad():
        for x in eval_ids:
            x = x.unsqueeze(0).to(device)
            out = model(input_ids=x, labels=x)
            nll += out.loss.item() * (x.size(1) - 1); ntok += (x.size(1) - 1)
    return float(nll / max(1, ntok))


def perplexity(model, eval_ids, tok, device):
    return float(math.exp(eval_loss(model, eval_ids, tok, device)))


# ============================ metrics =========================================== #
def _cider_d(preds, refs, n=4, sigma=6.0):
    """Self-contained CIDEr-D (Vedantam et al.), following the pycocoevalcap algorithm:
    TF-IDF over n-grams (IDF from the reference set), cosine similarity, repetition
    clipping, and a Gaussian length penalty. One reference per candidate here."""
    from collections import defaultdict
    def cook(toks):
        c = defaultdict(int)
        for k in range(1, n + 1):
            for i in range(len(toks) - k + 1):
                c[tuple(toks[i:i + k])] += 1
        return c
    crefs = [cook(r.split()) for r in refs]
    ctest = [cook(p.split()) for p in preds]
    df = defaultdict(float)
    for cr in crefs:
        for ng in set(cr):
            df[ng] += 1.0
    ref_len = math.log(max(1, len(crefs)))
    def vec(cnts):
        v = [defaultdict(float) for _ in range(n)]
        norm = [0.0] * n; length = 0
        for ng, tf in cnts.items():
            d = math.log(max(1.0, df[ng]))
            k = len(ng) - 1
            v[k][ng] = tf * (ref_len - d)
            norm[k] += v[k][ng] ** 2
            if k == 0:
                length += tf
        return v, [math.sqrt(x) for x in norm], length
    def sim(vh, nh, lh, vr, nr, lr):
        delta = float(lh - lr); val = [0.0] * n
        for k in range(n):
            for ng, c in vh[k].items():
                val[k] += min(c, vr[k].get(ng, 0.0)) * vr[k].get(ng, 0.0)
            if nh[k] and nr[k]:
                val[k] /= (nh[k] * nr[k])
            val[k] *= math.e ** (-(delta ** 2) / (2 * sigma ** 2))
        return sum(val) / n
    sc = []
    for ct, cr in zip(ctest, crefs):
        vh, nh, lh = vec(ct); vr, nr, lr = vec(cr)
        sc.append(10.0 * sim(vh, nh, lh, vr, nr, lr))
    return float(np.mean(sc)) if sc else float("nan")


def gen_metrics(model, pairs, tok, device, max_new=64):
    """Greedy generation on the first 100 (source, reference) test pairs, scored with a
    self-contained CIDEr-D. Perplexity is computed separately as the headline metric."""
    preds, refs = [], []
    model.eval()
    with torch.no_grad():
        for src, tgt in pairs[:100]:
            ids = tok(src, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            gen = tok.decode(out[0, ids.size(1):], skip_special_tokens=True)
            preds.append(gen.strip() or " "); refs.append(tgt.strip() or " ")
    res = {}
    try:
        res["cider"] = round(_cider_d(preds, refs), 4)
    except Exception as e:
        print("[gen] CIDEr failed:", e)
    return res


# ============================ helpers =========================================== #
def _param_counts(model):
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    try:
        bpe = next(p for p in model.parameters() if p.requires_grad).element_size()
    except StopIteration:
        bpe = 4
    return tr, tot, bpe


def _gpu_actually_works():
    """is_available() can be True on a GPU whose kernels still fail at runtime
    (e.g. a too-new architecture vs the installed CUDA, like sm_120 on an old
    wheel). So actually run a tiny matmul on the GPU and see if it succeeds."""
    if not (HAVE_STACK and torch.cuda.is_available()):
        return False
    try:
        x = torch.randn(8, 8, device="cuda")
        y = (x @ x).sum().item()
        return y == y            # finite (not NaN)
    except Exception as e:
        print(f"[warn] a CUDA GPU is present but failed a test computation "
              f"({type(e).__name__}: {e}). Falling back to CPU.")
        return False


def resolve_devices(mode):
    if mode == "cpu":
        return ["cpu"]
    if mode == "gpu":
        if _gpu_actually_works():
            return ["cuda"]
        print("[warn] device_mode='gpu' but no usable CUDA GPU -> running on CPU.")
        return ["cpu"]
    # "auto" (and legacy "both"): try GPU first, verify it really computes, else CPU
    if _gpu_actually_works():
        return ["cuda"]
    return ["cpu"]


def resolve_device_str(mode="auto"):
    devs = resolve_devices(mode)
    return "cuda" if "cuda" in devs else "cpu"


def _append_curve(path, key, curve):
    """Append a per-round loss curve to the JSON sidecar, keyed by 'dataset|model|method'.
    Tolerates the file being locked (Windows): retries briefly, and if it still cannot
    write it warns and carries on -- the curve is a diagnostic sidecar, so losing one
    must never abort a run whose result row was already saved."""
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            data = {}
    data[key] = curve
    tmp = path + ".tmp"
    for attempt in range(12):
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=1)
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(5.0)
        except Exception as e:
            print(f"  [warn] could not save curve '{key}': {type(e).__name__}: {e}")
            return
    print(f"  [warn] curve sidecar {path} stayed locked; skipped curve '{key}' "
          f"(the result row itself was saved).")


def _seed_all(sd):
    random.seed(sd); np.random.seed(sd)
    if HAVE_STACK:
        torch.manual_seed(sd)


def _free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_tok(model_id, hf_token):
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token or None)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def recoverable_rank_factored(B, A, v):
    """EXACT same value as recoverable_rank(B @ A, v), computed without the full SVD.

    d = B @ A with B (out x r) and A (r x in) has rank <= r by construction, so at most
    r of its singular values are nonzero -- yet np.linalg.svd(d) computes min(out, in)
    of them (1280 for GPT-2 large) and throws away all but r. Instead, take thin QRs of
    the two factors, B = Qb Rb and A^T = Qa Ra, so

        d = Qb (Rb Ra^T) Qa^T,

    where Qb and Qa have orthonormal columns. The nonzero singular values of d are then
    exactly the singular values of the tiny r x r matrix Rb Ra^T. That is an identity,
    not an approximation: same singular values to floating-point roundoff, and the
    caller caps the count at r anyway, so the returned integer is identical.

    Cost drops from an O(out * in * min(out,in)) dense SVD to two thin QRs plus an
    r x r SVD -- for GPT-2 large's 1280 x 3840 modules that is ~1.3 s -> ~1 ms."""
    out, inn = B.shape[0], A.shape[1]
    m, n = (out, inn) if out <= inn else (inn, out)
    beta = m / n
    Rb = np.linalg.qr(B, mode="r")                  # r x r
    Ra = np.linalg.qr(A.T, mode="r")                # r x r
    s = np.linalg.svd(Rb @ Ra.T, compute_uv=False)  # exact nonzero spectrum of B @ A
    edge = v * math.sqrt(n) * (1.0 + math.sqrt(beta))
    return int(max(1, np.sum(s > edge)))


def fed_rounds(model, shards, tok, device, method, rounds, rank, dp_sigma, steps, lr,
               weights=None, staleness=None, val_ids=None, es_patience=0, es_min_delta=0.0):
    """Run `rounds` of federated LoRA training on an EXISTING model, in place (used by the
    straggler study). weights = custom aggregation weights p (e.g. staleness weighted);
    staleness = per client delay: a stale client re-sends its last cached update instead of
    retraining. With `val_ids`, validation loss is tracked every round and the best round
    is restored at the end. Returns {best_round, rounds_ran, stopped_early, best_val}."""
    K = len(shards)
    p = np.array(weights, float) if weights is not None else np.full(K, 1.0 / K)
    p = p / p.sum()
    cache = [None] * K
    best_val, best_round, best_snap, bad, stopped, ran = float("inf"), 0, None, 0, False, 0
    for rd in range(rounds):
        gstate = {k: t.clone() for k, t in get_peft_model_state_dict(model).items()}
        cf = []
        for k in range(K):
            if staleness is not None and cache[k] is not None and (rd % (int(staleness[k]) + 1) != 0):
                cf.append(cache[k])                      # stale client: reuse its older update
            else:
                set_peft_model_state_dict(model, {kk: t.clone() for kk, t in gstate.items()})
                train_local(model, list(shards[k]), tok, device, steps, lr)
                fk = lora_factor_dict(model)
                cf.append(fk); cache[k] = fk
        agg = {}
        for bi, base in enumerate(cf[0].keys()):
            facs = [c[base] for c in cf]
            agg[base] = aggregate_module(facs, p, method, rank, dp_sigma=dp_sigma,
                                         seed=1234 + rd * 101 + bi * 7)
        write_back(model, agg, device)
        ran = rd + 1
        if val_ids is not None:
            vl = eval_loss(model, val_ids, tok, device)
            if best_snap is None or (vl == vl and vl < best_val - es_min_delta):
                best_val, best_round = vl, ran
                best_snap = {kk: vv.detach().cpu().clone()
                             for kk, vv in get_peft_model_state_dict(model).items()}
                bad = 0
            else:
                bad += 1
                if es_patience and bad >= es_patience:
                    stopped = True
                    break
    if val_ids is not None and best_snap is not None:
        set_peft_model_state_dict(model, {kk: vv.to(device) for kk, vv in best_snap.items()})
    else:
        best_round, best_val = ran, float("nan")
    return {"best_round": best_round, "rounds_ran": ran, "stopped_early": stopped,
            "best_val": round(float(best_val), 5) if best_val == best_val else float("nan")}


# ================================ registries ==================================== #
MODELS = {
    "gpt2-l":     "gpt2-large",
    "bloom-1b1":  "bigscience/bloom-1b1",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}
DATASETS = {
    "dolly":     ("databricks/databricks-dolly-15k", None),
    "alpaca":    ("yahma/alpaca-cleaned", None),
    "natinst":   ("Muennighoff/natural-instructions", None),
}
ALL_METHODS = ["centralized", "fedavg", "fedit", "fedptuning", "fedask",
               "fedossa", "fedossa_secagg"]
# FedIT aggregates exactly like FedAvg on LoRA, so the default runs FedAvg once
# (reported as "FedAvg / FedIT"); add `fedit` explicitly to run it under its own name.
DEFAULT_METHODS = ["centralized", "fedavg", "fedptuning", "fedask", "fedossa",
                   "fedossa_secagg"]
# methods with an aggregation step (noise and straggler studies apply to these)
AGG_METHODS = ["fedavg", "fedit", "fedptuning", "fedask", "fedossa", "fedossa_secagg"]
LORA_AGG = ["fedavg", "fedit", "fedask", "fedossa", "fedossa_secagg"]


def _agg_only(methods):
    keep = [m for m in methods if m in AGG_METHODS]
    dropped = [m for m in methods if m not in AGG_METHODS]
    if dropped:
        print(f"  [note] {dropped} have no aggregation step -> not part of this study.")
    return keep


# ============================ experimental defaults ============================= #
# Default configuration: 3 LLMs x 3 datasets x seeds 0, 1, 2.
DEFAULTS = dict(
    clients=8, per_client=128, max_len=256, local_steps=1, lr=2e-4,
    rank=16, lora_alpha=32, dp_sigma=0.1,
    rounds=30, patience=0, min_delta=0.002,       # patience=0 -> full horizon
    val_size=128, test_size=256, gen_eval=True, alpha_dirichlet=0.1,
    seeds=[0, 1, 2],
    models=list(MODELS),
    datasets=list(DATASETS),
    methods=list(DEFAULT_METHODS),
    dp_grid=[0.0, 0.2, 0.4, 0.6, 0.8],
    straggler_grid=[0.25, 0.5, 0.75],
    straggler_delay=2,
    # FedPTuning
    num_virtual_tokens=20,
    prompt_lr=None,                               # None -> chosen by validation (below)
    prompt_lr_grid=[3e-3, 1e-2, 3e-2],
    prompt_lr_dataset="dolly",
)


# ========================= shared CLI for the entry scripts ===================== #
def common_parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--models", nargs="+", default=None, choices=list(MODELS))
    ap.add_argument("--datasets", nargs="+", default=None, choices=list(DATASETS))
    ap.add_argument("--methods", nargs="+", default=None, choices=ALL_METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--clients", type=int, default=None)
    ap.add_argument("--per_client", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--local_steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="LoRA learning rate")
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--lora_alpha", type=int, default=None)
    ap.add_argument("--dp_sigma", type=float, default=None,
                    help="relative aggregation noise rho used by the sweep and stragglers")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None,
                    help="0 = run the full horizon (default); >0 = early stop")
    ap.add_argument("--min_delta", type=float, default=None)
    ap.add_argument("--val_size", type=int, default=None)
    ap.add_argument("--test_size", type=int, default=None)
    ap.add_argument("--alpha_dirichlet", type=float, default=None)
    ap.add_argument("--gen_eval", dest="gen_eval", action="store_true", default=None)
    ap.add_argument("--no_gen_eval", dest="gen_eval", action="store_false",
                    help="skip CIDEr-D (perplexity still reported; faster)")
    ap.add_argument("--dp_grid", nargs="+", type=float, default=None)
    ap.add_argument("--straggler_grid", nargs="+", type=float, default=None)
    ap.add_argument("--straggler_delay", type=int, default=None)
    ap.add_argument("--num_virtual_tokens", type=int, default=None)
    ap.add_argument("--prompt_lr", type=float, default=None,
                    help="FedPTuning learning rate (default: chosen by validation)")
    ap.add_argument("--prompt_lr_grid", nargs="+", type=float, default=None)
    ap.add_argument("--load_4bit", action="store_true", default=False)
    ap.add_argument("--device_mode", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--hf_token", default="")
    ap.add_argument("--out", default=None)
    return ap


_CFG_KEYS = ["models", "datasets", "methods", "seeds", "clients", "per_client",
             "max_len", "local_steps", "lr", "rank", "lora_alpha", "dp_sigma",
             "rounds", "patience", "min_delta", "val_size", "test_size",
             "alpha_dirichlet", "gen_eval", "dp_grid", "straggler_grid",
             "straggler_delay", "num_virtual_tokens", "prompt_lr", "prompt_lr_grid"]


def build_cfg(args, default_out="results"):
    """Merge: DEFAULTS <- explicit CLI overrides."""
    base = dict(DEFAULTS)
    for k in _CFG_KEYS:
        v = getattr(args, k, None)
        if v is not None:
            base[k] = v
    cfg = argparse.Namespace(**base)
    cfg.load_4bit = args.load_4bit
    cfg.device_mode = args.device_mode
    cfg.hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    cfg.out = args.out or default_out
    os.makedirs(cfg.out, exist_ok=True)
    return cfg


# ================================ trainers ====================================== #
class BaseTrainer:
    is_fed = False

    def __init__(self, model, tok, device, method):
        self.model = model
        self.tok = tok
        self.device = device
        self.method = method
        self.train_time = 0.0
        self.rec_rounds = []

    def snapshot(self):
        return {k: v.detach().cpu().clone()
                for k, v in get_peft_model_state_dict(self.model).items()}

    def restore(self, snap):
        set_peft_model_state_dict(
            self.model, {k: v.to(self.device) for k, v in snap.items()})

    def evaluate(self, ids):
        return eval_loss(self.model, ids, self.tok, self.device) if ids else float("nan")

    def generate_metrics(self, pairs):
        return gen_metrics(self.model, pairs, self.tok, self.device) if pairs else {}

    def step(self, rd):
        raise NotImplementedError

    def cleanup(self):
        try:
            del self.model
        except Exception:
            pass
        _free_gpu()


class FedTrainer(BaseTrainer):
    is_fed = True

    def __init__(self, model, tok, device, label, agg_name,
                 shards, rank, dp_sigma, steps, lr):
        super().__init__(model, tok, device, label)
        self.agg = agg_name
        self.shards = shards
        self.rank = rank
        self.dp = dp_sigma
        self.steps = steps
        self.lr = lr
        self.K = len(shards)
        self.p = np.full(self.K, 1.0 / self.K)

    def step(self, rd):
        gstate = {k: t.clone() for k, t in get_peft_model_state_dict(self.model).items()}
        cf, losses = [], []
        for k in range(self.K):
            set_peft_model_state_dict(self.model, {kk: t.clone() for kk, t in gstate.items()})
            t0 = time.perf_counter()
            losses.append(train_local(self.model, list(self.shards[k]),
                                      self.tok, self.device, self.steps, self.lr))
            self.train_time += time.perf_counter() - t0
            cf.append(lora_factor_dict(self.model))
        agg, rr = {}, []
        for bi, base in enumerate(cf[0].keys()):
            facs = [c[base] for c in cf]
            seed = 1234 + rd * 100003 + bi * 131
            agg[base] = aggregate_module(facs, self.p, self.agg, self.rank,
                                         dp_sigma=self.dp, seed=seed)
            prods = [bb @ aa for bb, aa in facs]
            base_std = float(np.mean([np.std(d) for d in prods])) + 1e-12
            v = self.dp * base_std * math.sqrt(float(np.sum(self.p ** 2)))
            for (bb, aa) in facs:
                # exact same integer as recoverable_rank(bb @ aa, ...), ~2000x faster
                rr.append(min(self.rank, recoverable_rank_factored(bb, aa, max(v, 1e-9))))
        write_back(self.model, agg, self.device)
        self.rec_rounds.append(float(np.mean(rr)) if rr else float(self.rank))
        return float(np.mean(losses))


class CentralizedTrainer(BaseTrainer):
    def __init__(self, model, tok, device, shards, steps, lr):
        super().__init__(model, tok, device, "centralized")
        self.pool = [x for shard in shards for x in shard]
        self.steps = steps
        self.lr = lr

    def step(self, rd):
        t0 = time.perf_counter()
        loss = train_local(self.model, list(self.pool), self.tok, self.device, self.steps, self.lr)
        self.train_time += time.perf_counter() - t0
        return loss


def train_full_horizon(trainer, val_ids, test_ids, pairs,
                       rounds, patience, min_delta, gen_eval):
    """Train `rounds` rounds, monitoring VALIDATION loss and logging the full curve.
    Reports BOTH horizons so overfitting is visible:
      * as-run final round  (metric 1a, 1d),
      * best validation round (metric 1b) -- the recommended operating point,
      * their difference     (metric 1c, the overfit gap).
    If `patience` > 0 the run also HARD-STOPS after `patience` non-improving rounds
    (to save laptop compute); with patience = 0 it runs the full horizon so the
    round-`rounds` numbers are always available. Test metrics are measured on the
    held-out test set at both the final and the best round."""
    best_val = float("inf")
    best_round = 0
    best_snap = None
    bad = 0
    stopped = False
    curve = []
    for rd in range(1, rounds + 1):
        tr_loss = trainer.step(rd)
        vl = trainer.evaluate(val_ids)
        curve.append({"round": rd,
                      "train_loss": round(float(tr_loss), 5),
                      "val_loss": round(float(vl), 5)})
        improved = (best_snap is None) or (vl == vl and vl < best_val - min_delta)
        if improved:
            best_val, best_round = vl, rd
            best_snap = trainer.snapshot()
            bad = 0
        else:
            bad += 1
            if patience and bad >= patience:
                stopped = True
                break

    t0 = time.perf_counter()
    # --- as-run FINAL round (model is currently at the last round trained) ---
    val_loss_final = curve[-1]["val_loss"] if curve else float("nan")
    train_loss_final = curve[-1]["train_loss"] if curve else float("nan")
    test_loss_final = trainer.evaluate(test_ids)
    ppl_final = (float(math.exp(test_loss_final))
                 if test_loss_final == test_loss_final else float("nan"))
    # --- roll back to BEST validation round and measure there ---
    if best_snap is not None:
        trainer.restore(best_snap)
    test_loss_best = trainer.evaluate(test_ids)
    ppl_best = (float(math.exp(test_loss_best))
                if test_loss_best == test_loss_best else float("nan"))
    gen = trainer.generate_metrics(pairs) if gen_eval else {}
    infer_time = time.perf_counter() - t0

    overfit_gap = (round(float(val_loss_final - best_val), 5)
                   if (best_val != float("inf") and val_loss_final == val_loss_final)
                   else float("nan"))
    return dict(
        curve=curve, best_round=best_round, rounds_ran=len(curve), stopped_early=stopped,
        # best-round (recommended) -- metric 1b, 7a
        val_loss_best=round(float(best_val), 5) if best_val != float("inf") else float("nan"),
        test_loss=round(float(test_loss_best), 5), perplexity=round(ppl_best, 4),
        train_loss_at_best=(curve[best_round - 1]["train_loss"] if best_round else float("nan")),
        # as-run final -- metric 1a, 1d
        eval_loss_final=val_loss_final, test_loss_final=round(float(test_loss_final), 5),
        perplexity_final=round(ppl_final, 4), train_loss_final=train_loss_final,
        # stability -- metric 1c
        overfit_gap=overfit_gap,
        infer_time=round(infer_time, 2), gen=gen,
    )


# ================================ CSV helpers =================================== #
def _result_files(path):
    """The main CSV plus any `_recovered_*.csv` siblings written while it was locked."""
    import glob
    files = [path] if os.path.exists(path) else []
    files += sorted(glob.glob(f"{os.path.splitext(path)[0]}_recovered_*.csv"))
    return files


def load_done(path, keycols):
    done = set()
    for p in _result_files(path):
        try:
            with open(p, newline="") as f:
                for r in csv.DictReader(f):
                    done.add(tuple(str(r.get(k, "")) for k in keycols))
        except Exception as e:
            print(f"  [warn] could not read {p}: {type(e).__name__}: {e}")
    return done


def _write_row(path, fields, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def append_row(path, fields, row, retries=120, wait=5.0):
    """Append one result row, surviving Windows file locks.

    On Windows a CSV that is open in Excel (or being scanned by OneDrive/antivirus)
    is locked, and writing raises PermissionError [Errno 13]. A long run must never
    die -- or silently lose a result -- because of that. So: retry for up to
    `retries*wait` seconds (just closing Excel lets the run continue on its own), and
    if it is still locked, write to a `_recovered_<timestamp>.csv` sibling instead.
    Recovered files are picked up automatically by load_done() and by the reporting
    scripts, so nothing is lost and nothing is recomputed."""
    for attempt in range(retries):
        try:
            _write_row(path, fields, row)
            return path
        except PermissionError:
            if attempt == 0:
                print(f"\n  [locked] cannot write {path}")
                print( "  [locked] that file is open in another program -- almost always "
                       "Excel. Close it and this run continues by itself.")
                print(f"  [locked] retrying every {wait:.0f}s for up to "
                      f"{retries * wait / 60:.0f} min ...")
            time.sleep(wait)
    alt = f"{os.path.splitext(path)[0]}_recovered_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    try:
        _write_row(alt, fields, row)
        print(f"  [locked] still locked -- this result was saved to {alt} instead "
              f"(it will be picked up automatically; nothing is lost).")
        return alt
    except Exception as e:
        print(f"  [ERROR] could not save this result anywhere: {type(e).__name__}: {e}")
        return None


# ================================ FedPTuning ===================================== #
def build_prompt_model(model_id, device, hf_token, tok, num_virtual_tokens=20):
    """Frozen LLM + trainable soft prompt of `num_virtual_tokens` tokens.

    The prompt is initialised from the input embeddings of real vocabulary tokens
    (sampled with the global torch RNG, which the drivers seed before every run), so the
    soft tokens start on the embedding manifold. A plain N(0, 1) initialisation is far
    off that manifold and can make the frozen model much worse than with no prompt."""
    from peft import PromptTuningConfig, TaskType
    kw = {"token": hf_token} if hf_token else {}
    dt = torch.bfloat16 if device == "cuda" else torch.float32
    try:
        base = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, **kw)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt, **kw)
    pcfg = PromptTuningConfig(task_type=TaskType.CAUSAL_LM,
                              num_virtual_tokens=num_virtual_tokens)
    model = get_peft_model(base, pcfg)
    emb = model.get_input_embeddings().weight
    special = set(getattr(tok, "all_special_ids", []) or [])
    n_vocab = min(len(tok), emb.shape[0])
    ids = []
    while len(ids) < num_virtual_tokens:
        i = int(torch.randint(0, n_vocab, (1,)).item())
        if i not in special:
            ids.append(i)
    pe = model.prompt_encoder[model.active_adapter].embedding.weight
    with torch.no_grad():
        pe.copy_(emb[torch.tensor(ids, device=emb.device)].detach().to(pe.dtype).to(pe.device))
    return model.to(device)


class PromptTrainer(BaseTrainer):
    """Federated prompt tuning: every client starts from the global prompt, trains it
    locally, and the server takes the weighted average of the uploaded prompts.
    Optional, for the robustness studies:
      dp_sigma  -> each upload gets Gaussian noise of std rho x (mean std of the uploads),
                   the same relative definition as for the LoRA updates;
      staleness -> a stale client re-sends its cached prompt (same rule as fed_rounds);
      weights   -> custom aggregation weights (e.g. staleness weighted)."""
    def __init__(self, model, tok, device, shards, steps, lr,
                 dp_sigma=0.0, weights=None, staleness=None):
        super().__init__(model, tok, device, "fedptuning")
        self.shards = shards
        self.steps = steps
        self.lr = lr
        self.K = len(shards)
        p = np.asarray(weights, float) if weights is not None else np.full(self.K, 1.0)
        self.p = p / p.sum()
        self.dp = float(dp_sigma)
        self.staleness = staleness
        self.cache = [None] * self.K

    def step(self, rd):
        rd0 = rd - 1                                      # same round index as fed_rounds
        gstate = {n: t.detach().clone()
                  for n, t in self.model.named_parameters() if t.requires_grad}
        cstates, losses = [], []
        for k in range(self.K):
            if (self.staleness is not None and self.cache[k] is not None
                    and rd0 % (int(self.staleness[k]) + 1) != 0):
                cstates.append(self.cache[k])             # stale: re-send older prompt
                continue
            for n, t in self.model.named_parameters():
                if t.requires_grad and n in gstate:
                    t.data.copy_(gstate[n])
            t0 = time.perf_counter()
            losses.append(train_local(self.model, list(self.shards[k]),
                                      self.tok, self.device, self.steps, self.lr))
            self.train_time += time.perf_counter() - t0
            cs = {n: t.detach().cpu().float().clone()
                  for n, t in self.model.named_parameters() if t.requires_grad}
            cstates.append(cs)
            self.cache[k] = cs
        params = dict(self.model.named_parameters())
        names = [n for n, t in self.model.named_parameters() if t.requires_grad]
        for j, n in enumerate(names):
            ups = [cstates[k][n] for k in range(self.K)]
            if self.dp > 0:                               # relative noise on the uploads
                base_std = float(np.mean([u.std().item() for u in ups])) + 1e-12
                rng = np.random.default_rng(1234 + rd0 * 101 + j * 7)
                ups = [u + torch.from_numpy(
                           (self.dp * base_std * rng.standard_normal(tuple(u.shape)))
                           .astype(np.float32)) for u in ups]
            avg = sum(self.p[k] * ups[k] for k in range(self.K))
            params[n].data.copy_(avg.to(params[n].dtype).to(params[n].device))
        return float(np.mean(losses)) if losses else float("nan")


LR_CHECK_FIELDS = ["seed", "model_key", "dataset", "lr", "val_loss_best", "best_round",
                   "perplexity"]


def prompt_lr_for(cfg, device, mkey, tok):
    """FedPTuning learning rate for model `mkey`: cfg.prompt_lr if given, else a saved
    choice, else a short check over cfg.prompt_lr_grid on cfg.prompt_lr_dataset (first
    seed, no aggregation noise) that keeps the lowest VALIDATION loss. The test set is
    never used to choose. Saved in <out>/fedptuning_lr_<model>.json."""
    if cfg.prompt_lr is not None:
        return float(cfg.prompt_lr)
    jpath = os.path.join(cfg.out, f"fedptuning_lr_{mkey}.json")
    if os.path.exists(jpath):
        return float(json.load(open(jpath))["lr"])
    model_id = MODELS.get(mkey, mkey)
    sd, ds = cfg.seeds[0], cfg.prompt_lr_dataset
    cpath = os.path.join(cfg.out, "fedptuning_lr_check.csv")
    done = load_done(cpath, ("seed", "model_key", "dataset", "lr"))
    shards, val_ids, test_ids, _ = load_clients(
        ds, tok, cfg.clients, cfg.per_client, cfg.max_len, cfg.val_size, cfg.test_size,
        False, sd, cfg.alpha_dirichlet)
    print(f"[prompt-lr] {mkey}: choosing the FedPTuning learning rate on {ds} "
          f"(validation loss) from {cfg.prompt_lr_grid}")
    for lr in cfg.prompt_lr_grid:
        if (str(sd), mkey, ds, str(lr)) in done:
            continue
        _seed_all(sd)
        tr = PromptTrainer(build_prompt_model(model_id, device, cfg.hf_token, tok,
                                              cfg.num_virtual_tokens),
                           tok, device, shards, cfg.local_steps, lr)
        res = train_full_horizon(tr, val_ids, test_ids, [], cfg.rounds, cfg.patience,
                                 cfg.min_delta, False)
        tr.cleanup()
        append_row(cpath, LR_CHECK_FIELDS, {
            "seed": sd, "model_key": mkey, "dataset": ds, "lr": lr,
            "val_loss_best": res["val_loss_best"], "best_round": res["best_round"],
            "perplexity": res["perplexity"]})
        print(f"  [prompt-lr] lr={lr:<7g} val_loss_best={res['val_loss_best']}")
    rows = [r for r in csv.DictReader(open(cpath)) if r["model_key"] == mkey
            and r["dataset"] == ds and r["seed"] == str(sd)]
    best = min(rows, key=lambda r: (float(r["val_loss_best"]), float(r["lr"])))
    json.dump({"model_key": mkey, "lr": float(best["lr"]), "chosen_by": "validation loss",
               "grid": cfg.prompt_lr_grid, "dataset": ds, "seed": sd,
               "init": "vocabulary embeddings", "num_virtual_tokens": cfg.num_virtual_tokens},
              open(jpath, "w"), indent=1)
    print(f"[prompt-lr] {mkey}: lr = {best['lr']}  -> {jpath}")
    return float(best["lr"])


def make_trainer(method, model_id, tok, device, shards, cfg, dp_sigma, prompt_lr=None,
                 weights=None, staleness=None):
    if method == "centralized":
        m = build_model(model_id, device, cfg.rank, cfg.lora_alpha, cfg.load_4bit, cfg.hf_token)
        return CentralizedTrainer(m, tok, device, shards, cfg.local_steps, cfg.lr)
    if method == "fedptuning":
        m = build_prompt_model(model_id, device, cfg.hf_token, tok, cfg.num_virtual_tokens)
        return PromptTrainer(m, tok, device, shards, cfg.local_steps, prompt_lr,
                             dp_sigma=dp_sigma, weights=weights, staleness=staleness)
    m = build_model(model_id, device, cfg.rank, cfg.lora_alpha, cfg.load_4bit, cfg.hf_token)
    return FedTrainer(m, tok, device, method, method, shards, cfg.rank, dp_sigma,
                      cfg.local_steps, cfg.lr)


# ================================ CSV schemas =================================== #
SWEEP_FIELDS = [
    "seed", "dataset", "model_key", "method", "model",
    # quality at the best validation round (reported)
    "perplexity", "test_loss", "val_loss_best", "cider",
    # quality at the final round (as run)
    "perplexity_final", "test_loss_final", "eval_loss_final",
    # stability + training
    "overfit_gap", "train_loss_final", "train_loss_at_best",
    "best_round", "rounds_ran", "stopped_early",
    # memory / communication
    "trainable_params", "total_params", "adapter_mb", "uplink_mb",
    "mean_recoverable_rank",
    # config + timing
    "rank", "lora_alpha", "lr", "dp_sigma", "clients", "rounds", "patience",
    "alpha_dirichlet", "train_time_s", "infer_time_s", "device", "wall_time_s",
]
DP_FIELDS = ["seed", "model_key", "dataset", "method", "dp_sigma",
             "perplexity", "perplexity_final", "eval_loss_final", "mean_recoverable_rank"]
ASYNC_FIELDS = ["seed", "model_key", "dataset", "method",
                "straggler_frac", "mode", "eval_loss", "perplexity"]


def _dense_product_entries(model):
    """Entries of the dense products B_k A_k over all adapted modules (SecAgg upload)."""
    return int(sum(B.shape[0] * A.shape[1] for (B, A) in lora_factor_dict(model).values()))


# ================================ SWEEP driver ================================== #
def _sweep_row(me, model_id, tok, device, shards, val_ids, test_ids, pairs, cfg, prompt_lr):
    # main comparison: LoRA aggregation methods at rho = cfg.dp_sigma; Centralized and
    # FedPTuning without aggregation noise (as in the paper)
    dp = cfg.dp_sigma if me in LORA_AGG else 0.0
    trainer = make_trainer(me, model_id, tok, device, shards, cfg, dp, prompt_lr=prompt_lr)
    res = train_full_horizon(trainer, val_ids, test_ids, pairs,
                             cfg.rounds, cfg.patience, cfg.min_delta, cfg.gen_eval)
    tr, tot, bpe = _param_counts(trainer.model)
    K = len(shards)
    if me == "centralized":
        uplink, recrank = 0.0, float("nan")
    else:
        per_round = (_dense_product_entries(trainer.model) if me == "fedossa_secagg" else tr)
        uplink = round(K * res["rounds_ran"] * per_round * bpe / 1e6, 4)
        recrank = (round(float(np.mean(trainer.rec_rounds)), 3)
                   if getattr(trainer, "is_fed", False) and trainer.rec_rounds else float("nan"))
    row = {
        "method": trainer.method, "model": model_id,
        "perplexity": res["perplexity"], "test_loss": res["test_loss"],
        "val_loss_best": res["val_loss_best"], "cider": res["gen"].get("cider"),
        "perplexity_final": res["perplexity_final"], "test_loss_final": res["test_loss_final"],
        "eval_loss_final": res["eval_loss_final"], "overfit_gap": res["overfit_gap"],
        "train_loss_final": res["train_loss_final"], "train_loss_at_best": res["train_loss_at_best"],
        "best_round": res["best_round"], "rounds_ran": res["rounds_ran"],
        "stopped_early": res["stopped_early"],
        "trainable_params": int(tr), "total_params": int(tot),
        "adapter_mb": round(tr * bpe / 1e6, 4), "uplink_mb": uplink,
        "mean_recoverable_rank": recrank,
        "rank": cfg.rank if me != "fedptuning" else f"{cfg.num_virtual_tokens}tok",
        "lora_alpha": cfg.lora_alpha,
        "lr": prompt_lr if me == "fedptuning" else cfg.lr, "dp_sigma": dp,
        "clients": K, "rounds": cfg.rounds, "patience": cfg.patience,
        "alpha_dirichlet": cfg.alpha_dirichlet,
        "train_time_s": round(trainer.train_time, 2), "infer_time_s": res["infer_time"],
    }
    trainer.cleanup()
    return row, res["curve"]


def run_sweep(cfg, device):
    """Main comparison: every method x dataset x model x seed (one row each)."""
    csv_path = os.path.join(cfg.out, "sweep_results.csv")
    curves_path = os.path.join(cfg.out, "sweep_curves.json")
    done = load_done(csv_path, ("seed", "dataset", "model_key", "method"))
    print(f"[sweep] seeds={cfg.seeds} models={cfg.models} datasets={cfg.datasets}")
    print(f"[sweep] methods={cfg.methods} rounds={cfg.rounds} gen_eval={cfg.gen_eval}")
    for sd in cfg.seeds:
        for mkey in cfg.models:
            model_id = MODELS[mkey]
            tok = _load_tok(model_id, cfg.hf_token)
            plr = prompt_lr_for(cfg, device, mkey, tok) if "fedptuning" in cfg.methods else None
            for ds in cfg.datasets:
                todo = [me for me in cfg.methods if (str(sd), ds, mkey, me) not in done]
                if not todo:
                    continue
                shards, val_ids, test_ids, pairs = load_clients(
                    ds, tok, cfg.clients, cfg.per_client, cfg.max_len,
                    cfg.val_size, cfg.test_size, cfg.gen_eval, sd, cfg.alpha_dirichlet)
                for me in todo:
                    _seed_all(sd)
                    t0 = time.perf_counter()
                    try:
                        row, curve = _sweep_row(me, model_id, tok, device, shards,
                                                val_ids, test_ids, pairs, cfg, plr)
                    except Exception as e:
                        print(f"    [sweep][FAIL] s{sd}/{ds}/{mkey}/{me}: {type(e).__name__}: {e}")
                        _free_gpu(); continue
                    row.update(seed=sd, dataset=ds, model_key=mkey, device=device,
                               wall_time_s=round(time.perf_counter() - t0, 2))
                    append_row(csv_path, SWEEP_FIELDS, row)
                    _append_curve(curves_path, f"s{sd}|{ds}|{mkey}|{me}", curve)
                    done.add((str(sd), ds, mkey, me))
                    print(f"    [sweep] s{sd}/{ds}/{mkey}/{me:<15} ppl={row['perplexity']:<9} "
                          f"cider={row['cider']} best_round={row['best_round']}")
    print(f"[sweep] done -> {csv_path}")


# ================================ NOISE driver ================================== #
def run_dp(cfg, device):
    """Aggregation noise study: test perplexity versus the relative noise level rho."""
    methods = _agg_only(cfg.methods)
    out = os.path.join(cfg.out, "dp_sweep.csv")
    done = load_done(out, ("seed", "model_key", "dataset", "method", "dp_sigma"))
    print(f"[dp] seeds={cfg.seeds} models={cfg.models} datasets={cfg.datasets} "
          f"methods={methods} grid={cfg.dp_grid}")
    for sd in cfg.seeds:
        for mkey in cfg.models:
            model_id = MODELS[mkey]
            tok = _load_tok(model_id, cfg.hf_token)
            plr = prompt_lr_for(cfg, device, mkey, tok) if "fedptuning" in methods else None
            for ds in cfg.datasets:
                shards, val_ids, test_ids, _ = load_clients(
                    ds, tok, cfg.clients, cfg.per_client, cfg.max_len,
                    cfg.val_size, cfg.test_size, False, sd, cfg.alpha_dirichlet)
                for me in methods:
                    for s in cfg.dp_grid:
                        if (str(sd), mkey, ds, me, str(s)) in done:
                            continue
                        _seed_all(sd)
                        try:
                            tr = make_trainer(me, model_id, tok, device, shards, cfg,
                                              float(s), prompt_lr=plr)
                            res = train_full_horizon(tr, val_ids, test_ids, [], cfg.rounds,
                                                     cfg.patience, cfg.min_delta, False)
                            recrank = (round(float(np.mean(tr.rec_rounds)), 3)
                                       if getattr(tr, "is_fed", False) and tr.rec_rounds else "")
                            tr.cleanup()
                        except Exception as e:
                            print(f"  [dp][FAIL] s{sd}/{mkey}/{ds}/{me}/rho={s}: {type(e).__name__}: {e}")
                            _free_gpu(); continue
                        append_row(out, DP_FIELDS, {
                            "seed": sd, "model_key": mkey, "dataset": ds, "method": me,
                            "dp_sigma": s, "perplexity": res["perplexity"],
                            "perplexity_final": res["perplexity_final"],
                            "eval_loss_final": res["eval_loss_final"],
                            "mean_recoverable_rank": recrank})
                        print(f"  [dp] s{sd}/{mkey}/{ds}/{me:<15} rho={s:<5} ppl={res['perplexity']}")
    print(f"[dp] done -> {out}")


# ================================ STRAGGLER driver ============================== #
def straggler_set(sd, ds, frac, K, delay):
    """Per client delay vector for a straggler fraction. Drawn from a seed that depends
    only on (seed, dataset, fraction), so EVERY method sees the same straggling clients."""
    staleness = np.zeros(K, dtype=int)
    n_slow = int(round(float(frac) * K))
    if n_slow > 0:
        key = [int(sd), sum(map(ord, ds)), int(round(float(frac) * 1000))]
        rng = np.random.default_rng(key)
        staleness[rng.choice(K, n_slow, replace=False)] = delay
    return staleness


def run_async(cfg, device):
    """Straggler study: a fraction phi of the clients is delayed by `straggler_delay`
    rounds and re-sends its cached update. async_naive weights stale updates equally,
    async_weighted down weights them by 1/(1+delay); sync (phi = 0) is the reference.
    The aggregation noise level is cfg.dp_sigma, as in the main comparison."""
    methods = _agg_only(cfg.methods)
    out = os.path.join(cfg.out, "async.csv")
    done = load_done(out, ("seed", "model_key", "dataset", "method", "straggler_frac", "mode"))
    print(f"[async] seeds={cfg.seeds} models={cfg.models} datasets={cfg.datasets} "
          f"methods={methods} grid={cfg.straggler_grid} delay={cfg.straggler_delay}")
    for sd in cfg.seeds:
        for mkey in cfg.models:
            model_id = MODELS[mkey]
            tok = _load_tok(model_id, cfg.hf_token)
            plr = prompt_lr_for(cfg, device, mkey, tok) if "fedptuning" in methods else None
            for ds in cfg.datasets:
                shards, val_ids, test_ids, _ = load_clients(
                    ds, tok, cfg.clients, cfg.per_client, cfg.max_len,
                    cfg.val_size, cfg.test_size, False, sd, cfg.alpha_dirichlet)
                shards = list(shards); K = len(shards)
                cells = [(0.0, "sync", None)]
                for frac in cfg.straggler_grid:
                    st = straggler_set(sd, ds, frac, K, cfg.straggler_delay)
                    cells += [(frac, "async_naive", st), (frac, "async_weighted", st)]
                for me in methods:
                    for frac, mode, staleness in cells:
                        if (str(sd), mkey, ds, me, str(float(frac)), mode) in done:
                            continue
                        w = ([1.0 / (1.0 + int(x)) for x in staleness]
                             if mode == "async_weighted" else None)
                        _seed_all(sd)
                        try:
                            if me == "fedptuning":
                                tr = make_trainer(me, model_id, tok, device, shards, cfg,
                                                  cfg.dp_sigma, prompt_lr=plr, weights=w,
                                                  staleness=staleness)
                                train_full_horizon(tr, val_ids, [], [], cfg.rounds,
                                                   cfg.patience, cfg.min_delta, False)
                                model = tr.model
                            else:
                                model = build_model(model_id, device, cfg.rank, cfg.lora_alpha,
                                                    cfg.load_4bit, cfg.hf_token)
                                fed_rounds(model, shards, tok, device, me, cfg.rounds, cfg.rank,
                                           cfg.dp_sigma, cfg.local_steps, cfg.lr, weights=w,
                                           staleness=staleness, val_ids=val_ids,
                                           es_patience=cfg.patience, es_min_delta=cfg.min_delta)
                            l = eval_loss(model, test_ids, tok, device)   # at the best round
                            del model; _free_gpu()
                        except Exception as e:
                            print(f"  [async][FAIL] s{sd}/{mkey}/{ds}/{me}/{frac}/{mode}: "
                                  f"{type(e).__name__}: {e}")
                            _free_gpu(); continue
                        append_row(out, ASYNC_FIELDS, {
                            "seed": sd, "model_key": mkey, "dataset": ds, "method": me,
                            "straggler_frac": float(frac), "mode": mode,
                            "eval_loss": round(l, 5), "perplexity": round(math.exp(l), 4)})
                        print(f"  [async] s{sd}/{mkey}/{ds}/{me:<15} phi={frac:<4} {mode:<15} "
                              f"ppl={round(math.exp(l), 3)}")
    print(f"[async] done -> {out}")


# ============================ laptop launcher =================================== #
STUDY = {"sweep": run_sweep, "dp": run_dp, "async": run_async}


def run_child(study, argv):
    """Run one study for the models in argv inside this process."""
    if not require_stack():
        return 2
    args = common_parser(f"FedOSSA {study}").parse_args(argv)
    cfg = build_cfg(args)
    device = resolve_device_str(cfg.device_mode)
    _seed_all(cfg.seeds[0])
    print("=" * 78)
    print(f"[{study}] device={device}" + (f" ({torch.cuda.get_device_name(0)})"
                                          if device == "cuda" else ""))
    print("=" * 78)
    STUDY[study](cfg, device)
    return 0


def launch(study, script_path):
    """Laptop entry point: one study, models one after another, each in its own
    subprocess (frees all GPU memory between models). Results go to
    <out>/part_<model>/. Extra command line arguments are passed through, e.g.
        python run_dp.py --models gpt2-l --methods fedossa fedask
    """
    argv = sys.argv[1:]
    if argv[:1] == ["--child"]:
        return run_child(study, argv[1:])
    pre = common_parser(study).parse_known_args(argv)[0]
    models = pre.models or DEFAULTS["models"]
    out_base = pre.out or "results"
    passthrough = []
    skip = False
    for i, a in enumerate(argv):                      # drop --models / --out (set per child)
        if skip:
            if a.startswith("--"):
                skip = False
            else:
                continue
        if a in ("--models", "--out"):
            skip = True
            continue
        passthrough.append(a)
    t_all, status = time.time(), []
    for i, mk in enumerate(models, 1):
        out_dir = os.path.join(out_base, f"part_{mk}")
        cmd = [sys.executable, script_path, "--child", "--models", mk, "--out", out_dir,
               *passthrough]
        print(f"\n{'-' * 78}\n [{i}/{len(models)}] {study} -- {mk} -> {out_dir}\n{'-' * 78}")
        t0 = time.time()
        try:
            rc = subprocess.run(cmd).returncode
        except KeyboardInterrupt:
            print("\n[interrupted] run the same command again to resume.")
            status.append((mk, "INTERRUPTED", time.time() - t0))
            break
        status.append((mk, "ok" if rc == 0 else f"FAILED (exit {rc})", time.time() - t0))
    print(f"\n{'=' * 78}\n {study} summary ({(time.time() - t_all) / 3600:.2f} h)")
    for mk, s, el in status:
        print(f"   {mk:<14}{s:<20}{el / 3600:6.2f} h")
    print("=" * 78)
    return 0 if status and all(s == "ok" for _, s, _ in status) else 1
