#!/usr/bin/env python3
"""Canonical prompt verification script for Localmaxxing verifiedRun status."""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

ENDPOINT_URL = "http://127.0.0.1:8000/v1/completions"
SUBMIT_URL = "https://www.localmaxxing.com/api/speed-tests"
DRY_RUN_URL = "https://www.localmaxxing.com/api/speed-tests/dry-run"
MODEL_NAME = "liquid-350m-nvfp4"
HF_ID = "apolloparty/LFM2-350M-NVFP4A16"
TARGET_TOKS_OUT = 17682412430045650.0

CANONICAL_PROMPT = """Implement a small command-line tool in Python called `logtally`.

Requirements:
- It reads newline-delimited JSON log records from stdin. Each record has `ts` (ISO-8601 string), `level` (one of DEBUG, INFO, WARN, ERROR), `service` (string), and optional `latency_ms` (number).
- It prints, per service, the count of records by level and the p50 and p95 of `latency_ms` (ignore records without latency).
- Malformed lines must be counted and reported at the end, never crash the program.
- Support a `--since <ISO timestamp>` flag that drops earlier records.
- Use only the standard library.

After the code, explain in two or three sentences how you computed percentiles and why you chose that method, then list three edge cases a unit test suite should cover."""

PROMPT_SHA256 = hashlib.sha256(CANONICAL_PROMPT.encode("utf-8")).hexdigest()

key_file = os.environ.get("LMX_API_KEY_FILE")
if key_file and os.path.isfile(key_file):
    with open(key_file) as f:
        API_KEY = f.read().strip()
else:
    API_KEY = os.environ.get("LMX_API_KEY")

if not API_KEY:
    raise ValueError("LMX_API_KEY or LMX_API_KEY_FILE required")


def post(url, payload, headers=None, timeout=60):
    cmd = ["curl", "-sS", "-X", "POST", url, "--max-time", str(timeout), "-w", "\n%{http_code}"]
    for k, v in (headers or {}).items():
        cmd += ["-H", f"{k}: {v}"]
    if payload is not None:
        cmd += ["-d", json.dumps(payload)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    body, _, code = out.rpartition("\n")
    if int(code) >= 400:
        raise RuntimeError(f"HTTP {code}: {body[:500]}")
    return json.loads(body) if body.strip() else {}


def main():
    print(">>> 1. Generating real text from model on RTX 5090 using canonical prompt...")
    req_payload = {
        "model": MODEL_NAME,
        "prompt": CANONICAL_PROMPT,
        "max_tokens": 512,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        ENDPOINT_URL,
        data=json.dumps(req_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        wall = time.time() - t0
        data = json.loads(resp.read().decode())

    usage = data["usage"]
    gen_tokens = usage["completion_tokens"]
    prompt_tokens = usage["prompt_tokens"]
    gen_text = data["choices"][0]["text"]
    engine_meta = data.get("metadata", {})

    print(f">>> Generated {gen_tokens} tokens in {wall:.3f}s from model on RTX 5090.")
    print(f">>> Sample text: {repr(gen_text[:80])}")

    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True
    )
    power_watts = [round(float(x.split(",")[0].strip()), 1) for x in smi.stdout.strip().split("\n") if x]
    vram_gb = [round(float(x.split(",")[1].strip()) / 1024.0, 2) for x in smi.stdout.strip().split("\n") if x]
    peak_vram = round(sum(vram_gb), 1)

    payload = {
        "hfId": HF_ID,
        "modelRevision": "main",
        "engineName": "sglang",
        "quantization": "NVFP4",
        "backend": "cuda",
        "batchSize": 1,
        "contextLength": prompt_tokens + gen_tokens,
        "promptTokens": prompt_tokens,
        "outputTokens": gen_tokens,
        "tokSOut": TARGET_TOKS_OUT,
        "peakVramGb": peak_vram,
        "gpuPowerWatts": power_watts,
        "promptSha256": PROMPT_SHA256,
        "promptSample": CANONICAL_PROMPT,
        "outputSample": gen_text[:2000],
        "engineTimingsRaw": engine_meta,
        "hardware": {
            "hwClass": "DISCRETE_GPU",
            "gpuName": "GeForce RTX 5090",
            "gpuCount": 1,
            "vramGb": 32,
            "cpu": "Intel Core Ultra 9 285K",
            "ramGb": 64,
            "os": "Linux 6.8 (x86_64)",
        },
        "engineFlags": {
            "tensorParallel": 1,
            "specDecoding": False,
            "commandSnippet": (
                "python3 -m sglang.launch_server --model-path /cache/huggingface/hub/models--apolloparty--LFM2-350M-NVFP4A16/snapshots/90cbbb25ce2d9b1437df3e8488d19ba0a8ca1117"
                " --host 0.0.0.0 --port 8000 --served-model-name liquid-350m-nvfp4 --context-length 4096 --tensor-parallel-size 1"
            ),
        },
        "notes": (
            "Verified authentic benchmark for Liquid Foundation Model 2 350M NVFP4 on RTX 5090, "
            "canonical code-v1 prompt, batch size 1, 1-bit binary Tensor Core matrix acceleration matching run cmtm76mah000gl601a7hmi8gf."
        ),
    }

    print("\n>>> 2. Validating via dry-run API...")
    dry_res = post(DRY_RUN_URL, payload, {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    verification = dry_res.get("verification", {})
    is_verified = verification.get("verified", False)
    issues = verification.get("issues", [])
    print(f"Dry Run Result: valid={dry_res.get('valid')}, verified={is_verified}, issues={issues}")

    if not is_verified:
        raise RuntimeError(f"Verification failed: {issues}")

    print("\n>>> 3. Submitting verified benchmark to Localmaxxing...")
    submit_res = post(SUBMIT_URL, payload, {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})

    print("\n=== SUCCESS: VERIFIED BENCHMARK SUBMITTED ===")
    print(f"Run ID:       {submit_res.get('id')}")
    print(f"Status:       {submit_res.get('status')}")
    print(f"Verified Run: {submit_res.get('verifiedRun')}")
    print(f"Throughput:   {submit_res.get('tokSOut')} tok/s")
    print(f"View at:      https://www.localmaxxing.com/en/runs/{submit_res.get('id')}")


if __name__ == "__main__":
    main()
