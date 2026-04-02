# Qwen3.5-0.8B Long-Context Results

実行日時:

- PyTorch backend: 2026-04-01 13:10-13:11 JST
- AutoDeploy backends: 2026-04-02 17:46-17:50 JST

## 共通条件

- モデル:
  - PyTorch backend: `Qwen3.5-0.8B` text mirror
  - AutoDeploy: `Qwen3.5-0.8B` text clean mirror
- GPU: `NVIDIA GeForce RTX 4070 Ti`
- 入力長: `2048 token`
- 出力長: `40 token`
- `max_seq_len = 2088`
- `max_num_tokens = 2088`
- runs: `3`
- prefix cache の影響を避けるため:
  - `kv_cache_config.enable_block_reuse = False`
  - run ごとに `cache_salt` を変更

## 平均結果

| backend | compile | attn | TTFT | TPOT | E2E | prefill | decode |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `pytorch` | - | - | `54.3 ms` | `3.98 ms/token` | `209.5 ms` | `37.7k tok/s` | `251 tok/s` |
| `autodeploy` | `torch-simple` | `torch` | `271.8 ms` | `46.81 ms/token` | `2097.2 ms` | `8.00k tok/s` | `21.36 tok/s` |
| `autodeploy` | `torch-simple` | `flashinfer` | `277.2 ms` | `48.09 ms/token` | `2152.8 ms` | `8.22k tok/s` | `20.81 tok/s` |
| `autodeploy` | `torch-simple` | `trtllm` | `255.2 ms` | `47.04 ms/token` | `2089.7 ms` | `8.53k tok/s` | `21.26 tok/s` |

速度換算:

- prefill: `2048 / TTFT`
- decode: `1 / TPOT`

## AutoDeploy の安定区間

AutoDeploy は 128-token warmup のあとでも、最初の 2048-token request が少し重く、Run 2-3 が安定していました。Run 2-3 の平均は以下です。

| backend | compile | attn | TTFT | TPOT | E2E | prefill | decode |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `autodeploy` | `torch-simple` | `torch` | `221.5 ms` | `46.79 ms/token` | `2046.2 ms` | `9.24k tok/s` | `21.37 tok/s` |
| `autodeploy` | `torch-simple` | `flashinfer` | `207.4 ms` | `48.50 ms/token` | `2098.8 ms` | `9.88k tok/s` | `20.64 tok/s` |
| `autodeploy` | `torch-simple` | `trtllm` | `207.3 ms` | `46.75 ms/token` | `2030.5 ms` | `9.88k tok/s` | `21.39 tok/s` |

## 各 backend の詳細

### PyTorch backend

- 実行スクリプト:
  - `scripts/qwen35_pytorch_backend_longctx_perf.py`
  - `scripts/run_qwen35_pytorch_backend_longctx_perf.sh`
- 平均:
  - `TTFT`: `54.3 ms`
  - `TPOT`: `3.98 ms/token`
  - `E2E`: `209.5 ms`
  - prefill: `37.7k tok/s`
  - decode: `251 tok/s`

各 run:

- Run 1:
  - `TTFT`: `55.73 ms`
  - `TPOT`: `4.004 ms/token`
  - `E2E`: `211.88 ms`
  - prefill: `36.8k tok/s`
  - decode: `249.7 tok/s`
- Run 2:
  - `TTFT`: `53.48 ms`
  - `TPOT`: `3.971 ms/token`
  - `E2E`: `208.36 ms`
  - prefill: `38.3k tok/s`
  - decode: `251.8 tok/s`
- Run 3:
  - `TTFT`: `53.71 ms`
  - `TPOT`: `3.966 ms/token`
  - `E2E`: `208.37 ms`
  - prefill: `38.1k tok/s`
  - decode: `252.2 tok/s`

### AutoDeploy `torch-simple + torch`

- 実行スクリプト:
  - `scripts/qwen35_autodeploy_trtllm_longctx_perf.py`
  - `scripts/run_qwen35_autodeploy_trtllm_longctx_perf.sh`
- 平均:
  - `TTFT`: `271.8 ms`
  - `TPOT`: `46.81 ms/token`
  - `E2E`: `2097.2 ms`
  - prefill: `8.00k tok/s`
  - decode: `21.36 tok/s`

各 run:

- Run 1:
  - `TTFT`: `372.24 ms`
  - `TPOT`: `46.85 ms/token`
  - `E2E`: `2199.26 ms`
  - prefill: `5.50k tok/s`
  - decode: `21.35 tok/s`
- Run 2:
  - `TTFT`: `222.01 ms`
  - `TPOT`: `46.85 ms/token`
  - `E2E`: `2049.31 ms`
  - prefill: `9.22k tok/s`
  - decode: `21.34 tok/s`
- Run 3:
  - `TTFT`: `221.05 ms`
  - `TPOT`: `46.72 ms/token`
  - `E2E`: `2043.14 ms`
  - prefill: `9.26k tok/s`
  - decode: `21.40 tok/s`

### AutoDeploy `torch-simple + flashinfer`

- 平均:
  - `TTFT`: `277.2 ms`
  - `TPOT`: `48.09 ms/token`
  - `E2E`: `2152.8 ms`
  - prefill: `8.22k tok/s`
  - decode: `20.81 tok/s`

各 run:

- Run 1:
  - `TTFT`: `416.81 ms`
  - `TPOT`: `47.28 ms/token`
  - `E2E`: `2260.74 ms`
  - prefill: `4.91k tok/s`
  - decode: `21.15 tok/s`
- Run 2:
  - `TTFT`: `207.36 ms`
  - `TPOT`: `50.17 ms/token`
  - `E2E`: `2163.83 ms`
  - prefill: `9.88k tok/s`
  - decode: `19.93 tok/s`
- Run 3:
  - `TTFT`: `207.41 ms`
  - `TPOT`: `46.83 ms/token`
  - `E2E`: `2033.78 ms`
  - prefill: `9.87k tok/s`
  - decode: `21.35 tok/s`

### AutoDeploy `torch-simple + trtllm`

- 平均:
  - `TTFT`: `255.2 ms`
  - `TPOT`: `47.04 ms/token`
  - `E2E`: `2089.7 ms`
  - prefill: `8.53k tok/s`
  - decode: `21.26 tok/s`

各 run:

- Run 1:
  - `TTFT`: `350.98 ms`
  - `TPOT`: `47.62 ms/token`
  - `E2E`: `2208.16 ms`
  - prefill: `5.84k tok/s`
  - decode: `21.00 tok/s`
- Run 2:
  - `TTFT`: `207.17 ms`
  - `TPOT`: `46.77 ms/token`
  - `E2E`: `2031.02 ms`
  - prefill: `9.89k tok/s`
  - decode: `21.38 tok/s`
- Run 3:
  - `TTFT`: `207.41 ms`
  - `TPOT`: `46.73 ms/token`
  - `E2E`: `2030.06 ms`
  - prefill: `9.87k tok/s`
  - decode: `21.40 tok/s`

## compile backend の状況

今回の long-context 記録は、実行できた `torch-simple` 系を採用しました。compile 系はこの環境で別途失敗しています。

- `torch-compile + flashinfer`
  - `triton.compiler.compiler` から `triton_key` を import できず失敗
- `torch-opt + trtllm`
  - `trtllm_attention_prepare_metadata` の fake tensor 経路が `.numpy()` を呼んで失敗

## 解釈

- この 2048/40 条件では、`backend=pytorch` が圧倒的に速いです
- AutoDeploy は `torch-simple` の範囲では、
  - prefill: およそ `4x-5x` 遅い
  - decode: およそ `12x` 遅い
- AutoDeploy の attention backend 差は小さく、steady-state では
  - `torch` / `flashinfer` / `trtllm` がほぼ横並び
  - `trtllm` attention がわずかに良い

## Raw results

- local runtime `results/` に保存した JSON を参照
