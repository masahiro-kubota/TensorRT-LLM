# Qwen3.5 未実装状況メモ

更新日: 2026-04-02 JST

## 調査対象

- upstream `main` の現行コード
- upstream のサポート行列
- merge 済み PR
- 2026-04-02 時点で open の PR / issue

このメモは **upstream TensorRT-LLM の現状** を整理したものです。  
ローカルで入れている実験用パッチや、`Qwen3.5-0.8B` 向けの私家版修正は含めません。

## このメモで区別する Qwen3.5 系

`Qwen3.5` は 1 モデル名ではなく、複数の系列を含みます。  
このメモでは、以下を明確に分けて扱います。

| 系列 | 主な HF architecture / model_type | 例 | このメモでの意味 |
| --- | --- | --- | --- |
| dense VLM / VLA | `Qwen3_5ForConditionalGeneration` / `qwen3_5` | ローカルの `Qwen3.5-0.8B` | ユーザーが今気にしている edge 向け対象。vision + text を持つトップレベル checkpoint。 |
| dense text backbone | `Qwen3_5ForCausalLM` / `qwen3_5_text` | 上の VLM から取り出した text mirror | classic engine build を議論するときの text-only 形。 |
| MoE VLM | `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe` | `Qwen3.5-35B-A3B` | upstream の multimodal bugfix / AutoDeploy 文脈でよく出てくる対象。 |
| MoE text | `Qwen3_5MoeForCausalLM` / `qwen3_5_moe_text` | `Qwen3.5-397B-A17B` | upstream の support matrix に載っている Qwen3.5 の代表例。 |

以降、曖昧さを避けるために、

- `Qwen3.5-MoE` は主に `Qwen3_5Moe*` 系
- `Qwen3.5 dense` は主に `Qwen3_5*` 系
- `Qwen3.5-0.8B` はローカルの dense VLM / VLA checkpoint

を指します。

## 結論

- `Qwen3.5` 全体はもう完全未対応ではありません。
- ただし、対応は **`_torch` / AutoDeploy 側に偏っていて、classic な engine build path には入っていません**。
- 2026-04-02 時点で upstream が公式に見せているのは、主に **`Qwen3.5-MoE`、特に `Qwen3_5MoeForCausalLM` 系の AutoDeploy / `_torch` 側**です。
- 一方で、**`Qwen3.5-0.8B` のような dense VLM / VLA を classic engine build したい、という話とはかなり別**です。
- `Qwen3.5 dense`、`Qwen3.5 dense multimodal/VL`、`Qwen3.5 MTP`、`Transformers v5 系との整合` は、まだ未整理または未マージの部分が残っています。

## すでに upstream に入っているもの

### 1. AutoDeploy の Qwen3.5 ベース対応

ここで主に入っているのは **MoE 系 (`Qwen3_5Moe*`)** です。

- PR: [#11394 AutoDeploy: Support Qwen3.5](https://github.com/NVIDIA/TensorRT-LLM/pull/11394)
- merge 日: 2026-02-20
- 内容:
  - Gated Delta / hybrid linear-attention
  - Qwen3.5 MoE 向け custom model
  - AutoDeploy config とテスト

補足:
- changed files を見る限り、追加先は `tensorrt_llm/_torch/auto_deploy/*` と関連テストです。
- classic `tensorrt_llm/models/*` には入っていません。

### 2. `_torch` 側の Qwen3.5 サポート

ここは **dense text (`Qwen3_5ForCausalLM`) と MoE text (`Qwen3_5MoeForCausalLM`) の routing / weight mapper** が中心です。
トップレベルの dense VLM / VLA (`Qwen3_5ForConditionalGeneration`) を fully supported にした、という話ではありません。

- PR: [#12302 Add Qwen 3.5 supporting (NVFP4)](https://github.com/NVIDIA/TensorRT-LLM/pull/12302)
- merge 日: 2026-03-24
- 内容:
  - `Qwen3_5ForCausalLM`
  - `Qwen3_5MoeForCausalLM`
  - Qwen3.5 config normalization
  - weight mapper
  - PyTorch backend の精度テスト

補足:
- changed files は `tensorrt_llm/_torch/models/*` と `tensorrt_llm/_torch/pyexecutor/config_utils.py` が中心です。
- これも classic engine build path ではありません。

### 3. Qwen3.5 MoE の multimodal fix

これは **MoE VLM (`Qwen3_5MoeForConditionalGeneration`)** の修正です。  
`Qwen3.5-0.8B` のような dense VLM の話ではありません。

- PR: [#12114 Qwen 3.5 fix 3d position ID handling](https://github.com/NVIDIA/TensorRT-LLM/pull/12114)
- merge 日: 2026-03-25
- 内容:
  - Qwen3.5 MoE multimodal path の position / mRoPE 修正
  - request-scoped cache とテスト

### 4. 公式ドキュメント上の扱い

現行 `main` の supported models では、Qwen3.5 として明示されているのは **MoE text** の `Qwen3_5MoeForCausalLM` です。

- source: [supported-models.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md)
- 現行行列では
  - `Qwen3_5MoeForCausalLM` は掲載されている
  - 注記で「AutoDeploy backend 経由」と書かれている
  - dense text の `Qwen3_5ForCausalLM` は掲載されていない
  - dense VLM / VLA の `Qwen3_5ForConditionalGeneration` は掲載されていない
  - multimodal feature matrix に Qwen3.5 はまだ出ていない

## まだ未実装、または未マージのもの

### 1. classic TensorRT engine build path

この節は主に **dense text (`Qwen3_5ForCausalLM`) と、その元になる dense VLM (`Qwen3_5ForConditionalGeneration`)** にとって重要です。  
ユーザーの `Qwen3.5-0.8B` の話に一番近い未実装はここです。

これが最大の未実装です。

対象:
- `tensorrt_llm.models/*`
- `trtllm-bench build`
- classic な HF weight convert -> engine build の流れ

現行 `main` の根拠:
- `MODEL_MAP` には `Qwen3ForCausalLM` と `Qwen3MoeForCausalLM` はあるが、`Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` はありません。
- source: [tensorrt_llm/models/__init__.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/models/__init__.py)
- `QWenConfig.from_hugging_face()` の `valid_types` も `qwen3` / `qwen3_moe` までで、`qwen3_5` 系を受けません。
- source: [tensorrt_llm/models/qwen/config.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/models/qwen/config.py)
- `QWenDecoderLayer` は attention + MLP / MoE 前提で、Qwen3.5 の `layer_types` や GatedDeltaNet を持っていません。
- source: [tensorrt_llm/models/qwen/model.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/models/qwen/model.py)

推論:
- **Qwen3.5 の classic engine build path は upstream では未実装** と見てよいです。
- GitHub 上でも、これを直接進めている open PR / issue は見つかっていません。

### 2. Qwen3.5 dense の「公式サポート整理」

ここで言う `dense` は **`Qwen3_5ForCausalLM` / `Qwen3_5ForConditionalGeneration` 系**です。

コード上は `_torch` 側に `Qwen3_5ForCausalLM` が入り始めていますが、公式サポートの見せ方はまだ追いついていません。

根拠:
- `supported-models.md` に `Qwen3_5ForCausalLM` の行がありません。
- 一方で `_torch` 側には Qwen3.5 dense 用の wrapper は入っています。
- source: [#12302](https://github.com/NVIDIA/TensorRT-LLM/pull/12302)

推論:
- **dense text のコードは入り始めているが、docs / support matrix / official support story は未整理** です。

### 3. Qwen3.5 dense multimodal / VL

ここで言う対象は **dense VLM / VLA (`Qwen3_5ForConditionalGeneration`)** です。  
ローカルの `Qwen3.5-0.8B` はこの系列に属します。

これはまだ open PR の領域です。

- open PR: [#12611 Add the Qwen3.5 multimodal support](https://github.com/NVIDIA/TensorRT-LLM/pull/12611)
- 2026-04-02 時点の state: open
- changed files:
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py`
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py`
  - `tensorrt_llm/_torch/models/modeling_qwen3vl.py`
  - `tensorrt_llm/_torch/pyexecutor/config_utils.py`
  - multimodal accuracy test

加えて、より広い文脈ではこの PR も open です。

- open PR: [#12203 Support Qwen3.5 Dense and MoE Models in Pytorch Backend](https://github.com/NVIDIA/TensorRT-LLM/pull/12203)
- 2026-04-02 時点の state: open
- PR 本文にも「dense multimodal pytorch backend support」とあります

関連 open issue:
- [#11947 Support Dense Multi-Modal Qwen3.5 unified HF checkpoints in trtllm-serve PyTorch/AutoDeploy backends](https://github.com/NVIDIA/TensorRT-LLM/issues/11947)

結論:
- **Qwen3.5 dense/VL は upstream で進行中だが、まだ fully merged ではない** です。

### 4. Qwen3.5 MTP

これもまだ open PR です。  
PR タイトルは Qwen3.5 ですが、実装上は **Qwen3Next / Qwen3.5 の hybrid stack** を触っています。

- open PR: [#12646 Add Qwen3.5 MTP support](https://github.com/NVIDIA/TensorRT-LLM/pull/12646)
- 2026-04-02 時点の state: open

補足:
- [#12203](https://github.com/NVIDIA/TensorRT-LLM/pull/12203) の本文にも `Note: MTP to be implemented` とあります。

結論:
- **Qwen3.5 の MTP は未マージ** です。

### 5. Qwen3.5 の performance / scale 改善

ここで主に議論されているのは **大きい Qwen3.5-MoE 系**です。  
`Qwen3.5-0.8B` の edge 実運用でそのまま効く話ではありません。

機能が全く無いわけではありませんが、性能最適化はまだ ongoing です。

- open PR: [#12265 AutoDeploy: Optimize Qwen3.5 perf](https://github.com/NVIDIA/TensorRT-LLM/pull/12265)
- open issue: [#11833 AutoDeploy: Improve AllReduce perf for Qwen3.5 model](https://github.com/NVIDIA/TensorRT-LLM/issues/11833)

結論:
- **Qwen3.5 は「動かす」段階から、「速くする」段階の open work も残っている** 状態です。

### 6. Transformers v5 系との整合

これも未解決です。

- open issue: [#12321 Qwen 3.5 support from huggingface's transformers v5.2+](https://github.com/NVIDIA/TensorRT-LLM/issues/12321)

issue の主張だけでなく、現行コードにも関連箇所があります。

- `tensorrt_llm/models/gpt/convert.py` は `AutoModelForVision2Seq` を import しています
- `tensorrt_llm/tools/multimodal_builder.py` も `AutoModelForVision2Seq` を import しています

source:
- [tensorrt_llm/models/gpt/convert.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/models/gpt/convert.py)
- [tensorrt_llm/tools/multimodal_builder.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/tools/multimodal_builder.py)

推論:
- **Qwen3.5 対応とは別に、release/container と Transformers v5 の整合もまだ安定していない** と見てよいです。

## close 済みの関連バグ

以下は「未実装」ではなく、いったん upstream で潰されているものです。

- [#11440 Support Qwen3.5 model in AutoDeploy](https://github.com/NVIDIA/TensorRT-LLM/issues/11440)
  - 対応 PR: [#11394](https://github.com/NVIDIA/TensorRT-LLM/pull/11394)
- [#12271 qwen3_5_moe model type missing multimodal placeholder registration](https://github.com/NVIDIA/TensorRT-LLM/issues/12271)
  - close 済み
- [#12290 Fix Qwen3.5 Multimodal Position ID handling](https://github.com/NVIDIA/TensorRT-LLM/issues/12290)
  - 対応 PR: [#12114](https://github.com/NVIDIA/TensorRT-LLM/pull/12114)

## 2026-04-02 時点の整理

### upstream で「ある」と言ってよいもの

- AutoDeploy の Qwen3.5-MoE ベース対応
- `_torch` 側の Qwen3.5 text/MoE routing の一部
- Qwen3.5-MoE multimodal の follow-up fix

### upstream で「まだ無い / まだ終わっていない」と言ってよいもの

- classic `tensorrt_llm.models` での Qwen3.5 engine build
- `trtllm-bench build` 前提の Qwen3.5 classic engine path
- Qwen3.5 dense/VL の fully merged な公式サポート
- Qwen3.5 MTP の merge 済み対応
- Qwen3.5 向け perf work の完了
- Transformers v5 系との release-level 整合

## 自分たちの文脈での読み替え

もし目的が **ローカルの dense VLM / VLA である `Qwen3.5-0.8B`** を edge で速く動かすことなら、2026-04-02 時点で upstream の穴は主に次の 2 つです。

1. **classic engine build path が無い**
2. **dense/VL 側がまだ fully merged ではない**

つまり、今 upstream が強いのは **`Qwen3.5-MoE + AutoDeploy/_torch`** 側であって、  
**`Qwen3.5-0.8B` のような dense VLA を classic TensorRT engine build したい** という話とはかなり別の場所が主戦場になっています。

## 参照リンク

- current main code
  - [tensorrt_llm/models/__init__.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/models/__init__.py)
  - [tensorrt_llm/models/qwen/config.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/models/qwen/config.py)
  - [tensorrt_llm/models/qwen/model.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/models/qwen/model.py)
  - [docs/source/models/supported-models.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md)
  - [tensorrt_llm/models/gpt/convert.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/models/gpt/convert.py)
  - [tensorrt_llm/tools/multimodal_builder.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/tools/multimodal_builder.py)
- merged PR
  - [#11394](https://github.com/NVIDIA/TensorRT-LLM/pull/11394)
  - [#12114](https://github.com/NVIDIA/TensorRT-LLM/pull/12114)
  - [#12302](https://github.com/NVIDIA/TensorRT-LLM/pull/12302)
- open PR
  - [#12203](https://github.com/NVIDIA/TensorRT-LLM/pull/12203)
  - [#12265](https://github.com/NVIDIA/TensorRT-LLM/pull/12265)
  - [#12611](https://github.com/NVIDIA/TensorRT-LLM/pull/12611)
  - [#12646](https://github.com/NVIDIA/TensorRT-LLM/pull/12646)
- open issue
  - [#11833](https://github.com/NVIDIA/TensorRT-LLM/issues/11833)
  - [#11947](https://github.com/NVIDIA/TensorRT-LLM/issues/11947)
  - [#12321](https://github.com/NVIDIA/TensorRT-LLM/issues/12321)
