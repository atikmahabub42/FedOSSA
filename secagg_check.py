#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""secagg_check.py -- numerical check of FedOSSA under pairwise masking secure aggregation.

    python secagg_check.py            -> results/secagg_check.csv (+ printed table)

For K clients holding product updates dW_k (one adapted module), the script runs the
protocol of the paper end to end:

  1. every pair (k, j) shares a seed s_kj; client k forms the mask
         M_k = sum_{j>k} PRG(s_kj) - sum_{j<k} PRG(s_jk)   (mod q),
  2. client k uploads  Y_k = Q(p_k dW_k) + M_k (mod q),  Q(x) = round(2^f x) mod q,
  3. the server sums the uploads, the masks cancel, and it decodes the centred lift,
  4. FedOSSA (optimal shrinkage + rank r refactor) is applied to the decoded sum.

It reports, for several fixed point precisions f: the decoding error against the bound
K * 2^(-f-1) (Proposition 2), the modulus condition, and the difference between FedOSSA
on the decoded sum and FedOSSA on the exact sum. Pure NumPy, deterministic (fixed seed).
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fedossa_core import optimal_shrink, refactor   # same estimator as the experiments

SEED = 0
K = 8                     # clients
D1, D2 = 384, 128         # one adapted module (d1 x d2)
RANK = 16                 # adapter rank
RHO = 0.1                 # relative aggregation noise level
Q_BITS = 60               # modulus q = 2^60 (int64 arithmetic stays exact)
PRECISIONS = [8, 12, 16, 20, 24, 28, 32]
OUT = os.path.join("results", "secagg_check.csv")


def prg(seed, shape, q):
    """Pseudorandom mask over Z_q expanded from a shared seed."""
    return np.random.default_rng(seed).integers(0, q, size=shape, dtype=np.int64)


def encode(x, f, q):
    return np.mod(np.rint(x * (2.0 ** f)).astype(np.int64), q)


def masks(K, shape, q, seeds):
    M = [np.zeros(shape, dtype=np.int64) for _ in range(K)]
    for k in range(K):
        for j in range(K):
            if j > k:
                M[k] = np.mod(M[k] + prg(seeds[(k, j)], shape, q), q)
            elif j < k:
                M[k] = np.mod(M[k] - prg(seeds[(j, k)], shape, q), q)
    return M


def main():
    rng = np.random.default_rng(SEED)
    q = 2 ** Q_BITS
    # client updates: low rank products (rank RANK each), LoRA scale, plus noise
    dW = []
    for _ in range(K):
        B = rng.standard_normal((D1, RANK)) * 1e-2
        A = rng.standard_normal((RANK, D2)) * 3e-2
        dW.append(B @ A)
    p = np.full(K, 1.0 / K)
    s_bar = float(np.mean([np.std(d) for d in dW]))
    sigma_E = RHO * s_bar
    noisy = [d + sigma_E * rng.standard_normal(d.shape) for d in dW]
    G_hat = sum(pk * d for pk, d in zip(p, noisy))               # exact aggregate
    nu = sigma_E * math.sqrt(float(np.sum(p ** 2)))
    B0, A0 = refactor(optimal_shrink(G_hat, nu), RANK)            # FedOSSA, exact sum
    ref = B0 @ A0

    seeds = {(k, j): int(rng.integers(0, 2 ** 31)) for k in range(K) for j in range(k + 1, K)}
    M = masks(K, G_hat.shape, q, seeds)
    mask_sum = np.mod(sum(M), q)
    rows = []
    for f in PRECISIONS:
        maxabs = max(float(np.max(np.abs(pk * d))) for pk, d in zip(p, noisy))
        q_ok = q > 2 * K * (2 ** f * maxabs + 1)
        Y = [np.mod(encode(pk * d, f, q) + Mk, q) for pk, d, Mk in zip(p, noisy, M)]
        S = np.zeros(G_hat.shape, dtype=np.int64)
        for y in Y:
            S = np.mod(S + y, q)
        lift = np.where(S > q // 2, S - q, S).astype(np.float64)   # centred lift
        G_dec = lift / (2.0 ** f)
        err = float(np.max(np.abs(G_dec - G_hat)))
        bound = K * 2.0 ** (-f - 1)
        B1, A1 = refactor(optimal_shrink(G_dec, nu), RANK)        # FedOSSA, decoded sum
        rel = float(np.linalg.norm(B1 @ A1 - ref) / np.linalg.norm(ref))
        upload_mean = float(np.mean(Y[0]) / q)                     # ~0.5 if mask is uniform
        rows.append([f, int(q_ok), f"{err:.3e}", f"{bound:.3e}", int(err <= bound),
                     f"{rel:.3e}", f"{upload_mean:.4f}"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    head = ["precision_f", "modulus_ok", "max_decode_error", "bound_K_2^-f-1",
            "within_bound", "fedossa_rel_diff", "upload_mean_over_q"]
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(rows)
    print(f"masks cancel exactly: {bool(np.all(mask_sum == 0))}   (K={K}, module {D1}x{D2}, "
          f"rank {RANK}, rho={RHO}, q=2^{Q_BITS})")
    print(" ".join(f"{h:>18}" for h in head))
    for r in rows:
        print(" ".join(f"{str(v):>18}" for v in r))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
