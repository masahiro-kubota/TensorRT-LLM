# Qwen3.5 関連 open PR メモ

更新日: 2026-04-02 JST

## 調査対象

- repository: `NVIDIA/TensorRT-LLM`
- 対象: 2026-04-02 時点で open の PR
- 検索語:
  - `Qwen3.5`
  - `Qwen 3.5`
  - `qwen3_5`

このメモは、**今 open の PR がどこを触っているか**を整理するためのものです。  
merge 済み PR や issue の全体像は、別メモ [`QWEN35_UNIMPLEMENTED_STATUS.md`](./QWEN35_UNIMPLEMENTED_STATUS.md) を参照してください。

## このメモで区別する Qwen3.5 系

| 系列 | 主な HF architecture / model_type | 例 | このメモでの意味 |
| --- | --- | --- | --- |
| dense VLM / VLA | `Qwen3_5ForConditionalGeneration` / `qwen3_5` | `Qwen3.5-0.8B` | ユーザーが今気にしている edge 向け対象。 |
| dense text | `Qwen3_5ForCausalLM` / `qwen3_5_text` | dense VLM の text backbone | text-only の `_torch` / backend 議論で出てくる対象。 |
| MoE VLM | `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe` | `Qwen3.5-35B-A3B` | multimodal / AutoDeploy / position fix 文脈でよく出る対象。 |
| MoE text | `Qwen3_5MoeForCausalLM` / `qwen3_5_moe_text` | `Qwen3.5-397B-A17B` | support matrix や NVFP4/perf 文脈でよく出る対象。 |

## 結論

- 2026-04-02 時点で、Qwen3.5 関連の open PR は複数あります。
- ただし、主戦場は **`_torch` / AutoDeploy / multimodal / MTP / perf / docs** です。
- **classic な `tensorrt_llm.models` + `trtllm-bench build` の engine build path を追加する open PR は見当たりません。**
- `Qwen3.5-0.8B` に比較的近いのは、dense multimodal を触る [#12203](https://github.com/NVIDIA/TensorRT-LLM/pull/12203) と [#12611](https://github.com/NVIDIA/TensorRT-LLM/pull/12611) です。
- ただし、どちらも **PyTorch backend / `_torch` 側**であり、classic engine build の話ではありません。

## classic にこだわらない場合の読み方

今の upstream の主流経路として読むなら、重要度は次の順です。

1. **dense VLM / VLA が `_torch` でどこまで入っているか**
   - 直接見るべき open PR は [#12203](https://github.com/NVIDIA/TensorRT-LLM/pull/12203) と [#12611](https://github.com/NVIDIA/TensorRT-LLM/pull/12611) です。
   - `Qwen3.5-0.8B` に一番近いのはここです。
2. **Qwen3.5 hybrid stack 共通機能**
   - [#12646](https://github.com/NVIDIA/TensorRT-LLM/pull/12646) の MTP
   - [#12557](https://github.com/NVIDIA/TensorRT-LLM/pull/12557) の GDN / BF16 MoE perf
3. **AutoDeploy の docs / support matrix / infra**
   - [#12340](https://github.com/NVIDIA/TensorRT-LLM/pull/12340)
   - [#12419](https://github.com/NVIDIA/TensorRT-LLM/pull/12419)

補足:
- 現行 tree には、dense text 側の足場はすでにかなりあります。
- 例えば `tensorrt_llm/_torch/models/modeling_qwen3_5.py` と `tensorrt_llm/_torch/pyexecutor/config_utils.py` には、`Qwen3_5ForCausalLM` と top-level `Qwen3_5ForConditionalGeneration` から text backbone を正規化するための処理が入っています。
- 一方で、`Qwen3.5-0.8B` のような **dense multimodal / VLA を end-to-end で扱う経路** は、まだ open PR 側の比重が大きいです。

## 1. 直接関係する open PR

### [#12203](https://github.com/NVIDIA/TensorRT-LLM/pull/12203) Support Qwen3.5 Dense and MoE Models in Pytorch Backend

- 状態: open
- 作成: 2026-03-13
- 最終更新: 2026-04-01
- 対象:
  - dense multimodal
  - MoE multimodal
  - PyTorch backend
- 主な変更ファイル:
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py`
  - `tensorrt_llm/_torch/models/modeling_qwen3_5_moe.py`
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_moe_weight_mapper.py`
  - `tensorrt_llm/_torch/pyexecutor/config_utils.py`
- 読み方:
  - **Qwen3.5 dense と MoE を `_torch` 側でまとめて前進させる PR** です。
  - PR 本文にも `dense multimodal pytorch backend support` とあります。
  - `Qwen3.5-0.8B` に一番近い open PR の1つです。
- 我々との重なり:
  - `Qwen3.5-0.8B` の dense VLM / VLA にはかなり近い
  - ただし **classic engine build path とは重ならない**

### [#12611](https://github.com/NVIDIA/TensorRT-LLM/pull/12611) Add the Qwen3.5 multimodal support

- 状態: open
- 作成: 2026-03-31
- 最終更新: 2026-04-01
- 対象:
  - dense multimodal
  - `_torch` / multimodal path
- PR 本文の要約:
  - `Qwen3.5-35B` multimodal support
- 主な変更ファイル:
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py`
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py`
  - `tensorrt_llm/_torch/models/modeling_qwen3vl.py`
  - `tensorrt_llm/_torch/pyexecutor/config_utils.py`
  - multimodal accuracy test
- 読み方:
  - **dense multimodal を細く追加する PR** です。
  - 12203 よりスコープが絞られています。
- 我々との重なり:
  - `Qwen3.5-0.8B` の VLA/VL という意味では近い
  - ただしタイトル上は `Qwen3.5-35B` 寄り
  - **classic engine build には触っていない**

### [#12646](https://github.com/NVIDIA/TensorRT-LLM/pull/12646) Add Qwen3.5 MTP support

- 状態: open
- 作成: 2026-04-01
- 最終更新: 2026-04-02
- 対象:
  - Qwen3Next / Qwen3.5 hybrid stack
  - speculative / MTP
- 主な変更ファイル:
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py`
  - `tensorrt_llm/_torch/models/modeling_speculative.py`
  - `tensorrt_llm/_torch/speculative/mtp.py`
  - `tensorrt_llm/_torch/modules/mamba/gdn_mixer.py`
- 読み方:
  - **Qwen3.5 の基礎サポートというより、その上に乗る MTP 機能追加**です。
  - dense/VL の baseline を通す前に必須、という種類の PR ではありません。
- 我々との重なり:
  - `Qwen3.5-0.8B` をまず動かす、という目的には直接は効かない
  - hybrid stack の共通実装を見る参考にはなる

### [#12265](https://github.com/NVIDIA/TensorRT-LLM/pull/12265) AutoDeploy: Optimize Qwen3.5 perf

- 状態: open
- 作成: 2026-03-17
- 最終更新: 2026-04-02
- 対象:
  - AutoDeploy
  - Qwen3.5-MoE
  - NVFP4 / TP8 / sharding / cudagraph
- 主な変更ファイル:
  - `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_qwen3_5_moe.py`
  - `tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py`
  - `tensorrt_llm/_torch/auto_deploy/transform/library/quantization.py`
  - `examples/auto_deploy/model_registry/configs/qwen3.5_moe_400b.yaml`
- 読み方:
  - **「Qwen3.5 を載せる」より「Qwen3.5-MoE を速くする」PR** です。
  - モデル規模も 0.8B edge VLA とはかなり離れています。
- 我々との重なり:
  - `Qwen3.5-0.8B` とは薄い
  - AutoDeploy の perf work が ongoing だと分かる

### [#12557](https://github.com/NVIDIA/TensorRT-LLM/pull/12557) Optimize GDN of Qwen3-Next/3.5; adds BF16 TRTLLM MoE

- 状態: open
- 作成: 2026-03-26
- 最終更新: 2026-04-02
- 対象:
  - Qwen3Next / Qwen3.5 の hybrid shared stack
  - Gated Delta Net kernel
  - BF16 TRTLLM MoE
- 主な変更ファイル:
  - `cpp/tensorrt_llm/kernels/causalConv1d/causalConv1d.cu`
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py`
  - `tensorrt_llm/_torch/modules/fused_moe/*`
  - `tensorrt_llm/_torch/modules/fla/*`
- 読み方:
  - **Qwen3.5 固有 onboarding というより、hybrid stack の性能改善 PR** です。
  - PR 本文では `Qwen3.5-35B-A3B BF16` の性能表も出ています。
- 我々との重なり:
  - dense VLA の直接サポートではない
  - GDN / shared stack の実装参照としては有用

## 2. 間接的に関係する open PR

### [#12221](https://github.com/NVIDIA/TensorRT-LLM/pull/12221) Add GLM-4.7-Flash and Qwen3.5 NVFP4 models to BTK benchmark registry

- 状態: open
- 対象:
  - AutoDeploy model registry
  - `Qwen3.5-397B-A17B-NVFP4`
- 主な変更ファイル:
  - `examples/auto_deploy/model_registry/models.yaml`
  - `examples/auto_deploy/model_registry/configs/qwen3.5_moe_400b_nvfp4.yaml`
- 読み方:
  - **Qwen3.5 の model registry / benchmark entry を足す PR** です。
  - モデル実装を増やす PR ではありません。

### [#12340](https://github.com/NVIDIA/TensorRT-LLM/pull/12340) Update supported models matrix with AD-onboarded architectures

- 状態: open
- 対象:
  - docs
  - support matrix
  - AutoDeploy onboarded architectures
- 主な変更ファイル:
  - `docs/source/models/supported-models.md`
  - `examples/auto_deploy/model_registry/configs/qwen3.5_dense.yaml`
  - `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_qwen3_5.py`
  - 多数の AutoDeploy custom model / test
- 読み方:
  - **Qwen3.5 単独の PR ではなく、AutoDeploy 大型更新の中に Qwen3.5 が含まれている**形です。
  - support matrix の見え方が今後変わるなら、この PR 経由の可能性があります。

### [#12419](https://github.com/NVIDIA/TensorRT-LLM/pull/12419) New sharding infrastructure

- 状態: open
- 対象:
  - new sharding infra
  - AutoDeploy
  - `qwen3_5_moe_sharding_poc`
- 主な変更ファイル:
  - `examples/auto_deploy/new_sharding/qwen/qwen3_5_moe_sharding_poc.yaml`
  - `tensorrt_llm/_torch/auto_deploy/models/custom/new_sharding/modeling_qwen3_5_moe.py`
  - `tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py`
- 読み方:
  - **Qwen3.5 向けというより、sharding 基盤の新設 PR** です。
  - Qwen3.5-MoE はその実験対象の1つです。

## 3. まだ open の古い draft bugfix PR

### [#11839](https://github.com/NVIDIA/TensorRT-LLM/pull/11839) Qwen3.5 fix position_id input flow

- 状態: open, draft
- 作成 / 最終更新: 2026-03-02
- 主な変更ファイル:
  - `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_qwen3_5_moe.py`
  - `examples/auto_deploy/model_registry/configs/qwen3.5_moe_35b.yaml`
- 読み方:
  - かなり初期の draft bugfix です。

### [#11865](https://github.com/NVIDIA/TensorRT-LLM/pull/11865) Qwen3.5 pos id fixes

- 状態: open, draft
- 作成 / 最終更新: 2026-03-03
- 主な変更ファイル:
  - `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_qwen3_5_moe.py`
  - `examples/auto_deploy/model_registry/configs/qwen3.5_moe_35b.yaml`
- 読み方:
  - 11839 の近縁版に見えます。

### [#11867](https://github.com/NVIDIA/TensorRT-LLM/pull/11867) Qwen3.5 posid fixes

- 状態: open, draft
- 作成: 2026-03-03
- 最終更新: 2026-03-06
- 主な変更ファイル:
  - `tensorrt_llm/_torch/auto_deploy/llm.py`
  - `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_qwen3_5_moe.py`
  - `tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py`
- 読み方:
  - これも early draft の位置付けに見えます。

補足:
- これら 3 本は今も open ですが、どれも draft のままで更新が止まっています。
- **推測**としては、時期と内容から見て、後に merge された [#12114](https://github.com/NVIDIA/TensorRT-LLM/pull/12114) の前段 draft が残っているように見えます。

## 4. 我々の作業との関係

### `Qwen3.5-0.8B` VLA / VL に近い open PR

- [#12203](https://github.com/NVIDIA/TensorRT-LLM/pull/12203)
- [#12611](https://github.com/NVIDIA/TensorRT-LLM/pull/12611)

この 2 本は、dense multimodal / VLM 系を `_torch` 側で前進させています。  
`Qwen3.5-0.8B` に最も近いのはここです。

### `Qwen3.5-0.8B` edge 実運用とは距離がある open PR

- [#12265](https://github.com/NVIDIA/TensorRT-LLM/pull/12265)
- [#12557](https://github.com/NVIDIA/TensorRT-LLM/pull/12557)
- [#12646](https://github.com/NVIDIA/TensorRT-LLM/pull/12646)

これらは MoE / perf / MTP / shared-stack 改善で、0.8B dense VLA をまず通す仕事とは別です。

### 今も見当たらないもの

以下を触る open PR は、今回確認した範囲では見つかっていません。

- `tensorrt_llm/models/__init__.py`
- `tensorrt_llm/models/qwen/config.py`
- `tensorrt_llm/models/qwen/model.py`
- classic `trtllm-bench build`
- classic HF weight convert -> TensorRT engine build path

結論:
- **「Qwen3.5 の open PR は存在する」**
- しかし **「classic engine build を誰かが今やっている」わけではない**
- upstream の現在地は、かなり明確に `_torch` / AutoDeploy 側です
