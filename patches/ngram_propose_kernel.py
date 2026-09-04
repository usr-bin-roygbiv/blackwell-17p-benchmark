import torch
import triton
import triton.language as tl


@triton.jit
def _ngram_scan_kernel(
    tokens_ptr, best_ptr, W, MAX_GRAM: tl.constexpr, BLOCK: tl.constexpr
):
    """Program g scans candidate starts in parallel; threads find matches
    and record the LATEST matching start for gram length g+2.
    """
    g_idx = tl.program_id(0)          # 0 => gram length 2
    g = g_idx + 2
    best_ptr += g_idx

    cur_best = -1
    offs = tl.arange(0, BLOCK)
    for base in range(0, W, BLOCK):
        start = base + offs           # candidate positions
        valid = (start + g <= W - g)  # occurrence must end before the suffix begins
        match = valid
        for i in range(0, MAX_GRAM):
            if i < g:
                a = tl.load(tokens_ptr + start + i, mask=valid, other=-2)
                bsv = tl.full((1,), W - g + i, tl.int64)
                b = tl.load(tokens_ptr + bsv)
                bcast = tl.sum(b)     # scalar broadcast
                match = match & (a == bcast)
        m_start = tl.where(match, start, -1)
        cur_best = tl.maximum(cur_best, tl.max(m_start, axis=0))

    tl.store(best_ptr, cur_best)


@triton.jit
def _ngram_fill_kernel(
    tokens_ptr, best_ptr, out_ptr, W, N, MAX_GRAM: tl.constexpr, BLOCK_N: tl.constexpr
):
    done = 0
    for g in range(MAX_GRAM, 1, -1):
        if done == 0:
            s = tl.load(best_ptr + (g - 2))
            if s >= 0:
                L = W - s - g
                P = 0
                for p in range(1, 17):
                    if P == 0 and p < g:
                        per = 1
                        for i in range(0, 32):
                            if i + p < g:
                                a = tl.load(tokens_ptr + s + i)
                                b = tl.load(tokens_ptr + s + i + p)
                                per = per & (a == b)
                        if per == 1:
                            P = p
                if P == 0:
                    P = g
                Lp = L - (L % P)
                for base_j in range(0, N, BLOCK_N):
                    offs_j = base_j + tl.arange(0, BLOCK_N)
                    mask_j = offs_j < N
                    pos = s + g + (offs_j % Lp)
                    val = tl.load(tokens_ptr + pos, mask=mask_j)
                    tl.store(out_ptr + offs_j, val, mask=mask_j)
                done = 1
    if done == 0:
        for base_j in range(0, N, BLOCK_N):
            offs_j = base_j + tl.arange(0, BLOCK_N)
            mask_j = offs_j < N
            tl.store(out_ptr + offs_j, -1, mask=mask_j)


def ngram_propose(
    window: torch.Tensor,
    n_slots: int,
    max_gram: int = 32,
    min_gram: int = 4,
) -> torch.Tensor:
    W = window.shape[0]
    out = torch.full((n_slots,), -1, dtype=torch.int64, device=window.device)
    best = torch.full((max_gram - 1,), -1, dtype=torch.int64, device=window.device)
    _ngram_scan_kernel[(max_gram - 1,)](window, best, W, MAX_GRAM=max_gram, BLOCK=256)
    if min_gram > 2:
        # Discard sub-threshold gram slots so the fill kernel can only pick a
        # match of length >= min_gram.
        best[: min(min_gram - 2, max_gram - 1)] = -1
    _ngram_fill_kernel[(1,)](window, best, out, W, n_slots, MAX_GRAM=max_gram, BLOCK_N=128)
    return out


def reference(
    window: torch.Tensor, n_slots: int, max_gram: int = 32, min_gram: int = 1
):
    W = window.shape[0]
    w = window.tolist()
    for g in range(min(max_gram, W - 1), max(min_gram, 2) - 1, -1):
        suffix = w[W - g:]
        for start in range(W - 2 * g, -1, -1):
            if start + g <= W - g and w[start:start + g] == suffix:
                gram = w[start:start + g]
                P = 0
                for p in range(1, 17):
                    if P == 0 and p < g and all(
                        gram[i] == gram[i + p] for i in range(g - p)
                    ):
                        P = p
                if P == 0:
                    P = g
                cont = w[start + g:W]
                L = len(cont)
                Lp = L - (L % P)
                return torch.tensor(
                    [cont[j % Lp] for j in range(n_slots)], dtype=torch.int64
                )
    return torch.full((n_slots,), -1, dtype=torch.int64)
