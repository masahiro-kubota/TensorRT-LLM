# PR候補2 具体化メモ: dense Qwen3.5 multimodal routing fix

更新日: 2026-04-02 JST

## このPRの狙い

`Qwen3_5ForConditionalGeneration` を `Qwen3VLModel` が正しく dense text backbone に流せるようにする。

この PR は、**Qwen3.5 全対応**ではない。狙いはあくまで、

- top-level architecture が `Qwen3_5ForConditionalGeneration`
- nested `text_config` を持つ multimodal checkpoint
- `_torch` 側の VLM wrapper で `Unsupported architecture` にならない

ようにすることだけ。

## なぜこのPRが成立するか

2026-04-02 時点の `main` では、候補2に必要な前提のかなりの部分は既に入っている。

- `tensorrt_llm/_torch/models/modeling_qwen3_5.py`
  - `Qwen3_5ForCausalLM`
  - `Qwen3_5MoeForCausalLM`
  を既に register 済み
- `tensorrt_llm/_torch/pyexecutor/config_utils.py`
  - `_Qwen35ConfigCompat`
  があり、top-level `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration` から nested `text_config` を引き出して正規化する経路が既にある
- open PR `#12611`
  - `Qwen3.5-MoE multimodal` を足している
  - reviewer からも「dense model もあるのでは」と明示的に指摘されている

つまり、**dense Qwen3.5 text backbone の登録や config 正規化は既にある**。足りないのは、multimodal wrapper 側の architecture routing だけに近い。

## 現在の欠けている部分

`tensorrt_llm/_torch/models/modeling_qwen3vl.py` の `Qwen3VLModelBase.__init__()` では、`self.original_arch` に応じて text LLM 側の architecture を選び直している。

2026-04-02 時点の branch では、この分岐は次だけを明示的に扱っている。

- `Qwen3VLForConditionalGeneration` -> `Qwen3ForCausalLM`
- `Qwen3VLMoeForConditionalGeneration` -> `Qwen3MoeForCausalLM`

該当箇所:

- [modeling_qwen3vl.py](/media/masa/ssd_data/minipamayo_experiments/qwen35_trtllm_fork/tensorrt_llm/_torch/models/modeling_qwen3vl.py#L926)

一方で、手元の `Qwen3.5-0.8B` は top-level architecture が `Qwen3_5ForConditionalGeneration` である。

該当箇所:

- [Qwen3.5-0.8B config.json](/home/masa/minipamayo/shared_checkpoints/hf_models/Qwen3.5-0.8B/config.json#L2)

このため、**dense Qwen3.5 multimodal checkpoint だけ wrapper routing の最後で落ちる**。

## このPRで実際に変えるもの

### 変更ファイル

- `tensorrt_llm/_torch/models/modeling_qwen3vl.py`
- `tests/unittest/_torch/modeling/test_modeling_qwen3vl.py`

### 本体コードの変更

`Qwen3VLModelBase.__init__()` の architecture remap 分岐に、dense Qwen3.5 用の 1 branch を足す。

イメージはこれ。

```python
if self.original_arch == "Qwen3VLForConditionalGeneration":
    llm_model_config.pretrained_config.architectures = ["Qwen3ForCausalLM"]
elif self.original_arch == "Qwen3_5ForConditionalGeneration":
    llm_model_config.pretrained_config.architectures = ["Qwen3_5ForCausalLM"]
elif self.original_arch == "Qwen3VLMoeForConditionalGeneration":
    llm_model_config.pretrained_config.architectures = ["Qwen3MoeForCausalLM"]
elif self.original_arch == "Qwen3_5MoeForConditionalGeneration":
    llm_model_config.pretrained_config.architectures = ["Qwen3_5MoeForCausalLM"]
else:
    raise ValueError(f"Unsupported architecture: {self.original_arch}")
```

注意点:

- `Qwen3_5MoeForConditionalGeneration` 側は `#12611` が先に入る前提
- こちらの PR 候補 2 は、その上に dense branch を足す follow-up として出すのが自然
- もし `#12611` が未 merge のままなら、MoE branch を含めるか author と調整が必要

### テストの変更

heavy な multimodal E2E scenario を増やすより、**routing だけを見る targeted unit test** の方がこの PR には合っている。

推奨するテスト方針:

1. `test_modeling_qwen3vl.py` に small unit test を 1 つ追加する
2. top-level architecture が `Qwen3_5ForConditionalGeneration` の synthetic config を作る
3. `AutoModelForCausalLM.from_config` を mock して、呼ばれた config の `architectures` を検査する
4. `Qwen3VisionModelBase` も mock して、vision encoder 初期化を軽くする
5. assert は
   - `from_config` が 1 回呼ばれる
   - 渡された text config の `architectures == ["Qwen3_5ForCausalLM"]`
   に絞る

この形にすると、

- private checkpoint 不要
- full weight load 不要
- perf / accuracy baseline 不要
- routing の regressions だけ見られる

ので、小PRとして非常に説明しやすい。

## synthetic config に入れる最小項目

top-level は dense Qwen3.5 multimodal を模した dict にする。

必要なもの:

- `architectures = ["Qwen3_5ForConditionalGeneration"]`
- `model_type = "qwen3_5"`
- `text_config`
  - `model_type = "qwen3_5_text"`
  - `hidden_size`
  - `intermediate_size`
  - `num_hidden_layers`
  - `num_attention_heads`
  - `num_key_value_heads`
  - `head_dim`
  - `linear_key_head_dim`
  - `linear_value_head_dim`
  - `linear_num_key_heads`
  - `linear_num_value_heads`
  - `full_attention_interval` あるいは `layer_types`
  - `rope_scaling` あるいは `rope_parameters`
- `vision_config`
  - `hidden_size`
  - `out_hidden_size`
  - `depth`
  - `num_heads`
  - `patch_size`
  - `spatial_merge_size`
  - `temporal_patch_size`

ただし、この PR では full multimodal forward までは見ないので、**本当に必要なのは constructor が通る最低限**でよい。

## PR本文で主張すること

この PR は次の 1 文に尽きる。

`Qwen3.5 dense multimodal checkpoints already have text-backbone registration and config normalization in main, but Qwen3VLModel still rejects the top-level Qwen3_5ForConditionalGeneration architecture. This PR adds the missing routing and a narrow regression test.`

重要なのは、**「新機能」ではなく「既存 Qwen3.5 path の missing routing fix」**として説明すること。

## PRタイトル案

- `[None][fix] Route dense Qwen3.5 multimodal checkpoints in Qwen3VLModel`

別案:

- `[None][fix] Add dense Qwen3.5 multimodal routing in Qwen3VLModel`

## Test Coverage に書くべきこと

- Added a narrow unit test for `Qwen3_5ForConditionalGeneration` routing in `Qwen3VLModel`
- The test asserts that dense Qwen3.5 multimodal checkpoints are remapped to `Qwen3_5ForCausalLM`
- No weight loading or end-to-end multimodal accuracy coverage is added in this PR

## このPRでやらないこと

- `Qwen3.5-0.8B` の end-to-end VLA support を主張する
- weight mapper 修正
- config discovery fallback 修正
- MoE support
- multimodal accuracy test 追加
- perf 改善

この PR は、**routing bugfix + regression test** に閉じるべき。

## maintainer にとっての価値

この PR が小さいわりに意味がある理由は 3 つ。

- `#12611` の reviewer comment に直接つながる
- diff が 1 production file + 1 test file に閉じる
- `Qwen3.5` 本流の active work の穴埋めとして説明できる

つまり、これは「ローカル hack の upstream 化」ではなく、**review 中の Qwen3.5 multimodal work に対する natural follow-up** として出せる。

## このPRを出してよい条件

次の 2 条件を満たしたら出してよい。

1. `#12611` が merge されるか、author が dense support を別PRに分けることに合意している
2. `main` 上で `Qwen3_5ForConditionalGeneration` がまだ `Qwen3VLModel` routing で落ちる

逆に、`#12611` に dense branch まで取り込まれたら、この候補は消える。

## 実務的な進め方

1. `#12611` の merge を待つ
2. merge 後の `main` で `modeling_qwen3vl.py` の branch を再確認する
3. dense branch がまだ無ければ、この PR を 1 production file + 1 unit test file で切る
4. PR 本文では `reviewer noted that dense Qwen3.5 exists too` を文脈として使う

## 一言で言うと

候補2は、`Qwen3.5-0.8B` を動かす local bring-up から出てきた修正ではあるが、PR としては **「Qwen3.5 multimodal support の unfinished edge を埋める小さな follow-up」** として出すのが正しい。
