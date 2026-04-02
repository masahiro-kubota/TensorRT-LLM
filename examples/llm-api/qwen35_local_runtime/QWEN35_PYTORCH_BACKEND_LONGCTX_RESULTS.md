# Qwen3.5-0.8B PyTorch Backend Long-Context Result

実行日時: 2026-04-01 13:10-13:11 JST

## 条件

- モデル: `Qwen3.5-0.8B` text mirror
- backend: `pytorch`
- GPU: `NVIDIA GeForce RTX 4070 Ti`
- 入力長: `2048 token`
- 出力長: `40 token`
- `max_seq_len = 2088`
- `max_num_tokens = 2088`
- runs: `3`
- prefix cache の影響を避けるため:
  - `kv_cache_config.enable_block_reuse = False`
  - run ごとに `cache_salt` を変更

実行スクリプト:

- `scripts/qwen35_pytorch_backend_longctx_perf.py`
- `scripts/run_qwen35_pytorch_backend_longctx_perf.sh`

## 平均結果

- `TTFT`: `54.3 ms`
- `TPOT`: `3.98 ms/token`
- `E2E`: `209.5 ms`

速度換算:

- prefill 速度: 約 `37.7k tok/s`
  - `2048 / TTFT` から逆算した wall-clock ベースの近似値
- decode 速度: 約 `251 tok/s`
  - `1 / TPOT`

解釈:

- 1 token目が出るまで: 約 `54 ms`
- 残り `39 token` の decode: 約 `155 ms`
- 40 token 全体の生成: 約 `210 ms`

## 各 run

### Run 1

- `TTFT`: `55.73 ms`
- `TPOT`: `4.004 ms/token`
- `E2E`: `211.88 ms`
- prefill: 約 `36.8k tok/s`
- decode: 約 `249.7 tok/s`

### Run 2

- `TTFT`: `53.48 ms`
- `TPOT`: `3.971 ms/token`
- `E2E`: `208.36 ms`
- prefill: 約 `38.3k tok/s`
- decode: 約 `251.8 tok/s`

### Run 3

- `TTFT`: `53.71 ms`
- `TPOT`: `3.966 ms/token`
- `E2E`: `208.37 ms`
- prefill: 約 `38.1k tok/s`
- decode: 約 `252.2 tok/s`

## 注意

- ここでの prefill 速度は、pure kernel time ではなく `TTFT` から逆算した近似値
- 今回 TRT-LLM から返った `time_breakdown_metrics` は空だったため、GPU event ベースの厳密な prefill/decode 内訳は取れていない
