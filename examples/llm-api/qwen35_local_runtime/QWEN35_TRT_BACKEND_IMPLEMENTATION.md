# Qwen3.5 TensorRT Backend 実装メモ

## 結論

- 新しく `git clone` してくるのは必須ではありません。
- `/media/masa/ssd_data/minipamayo_experiments/qwen35_trtllm_qwen35_commit` にある既存の source tree で十分調査と実装が進められます。
- ただし、その tree は dirty な detached HEAD なので、実装作業を直接そこに入れるのは避けた方が安全です。
- 実装に入るなら、同じ commit から clean な `git worktree` を切るか fresh clone を用意するのを推奨します。
- upstream 向けには、いきなり「Qwen3.5 全対応」を 1 PR で狙わず、まずは `dense text-only + 単一 GPU + BF16` の最小 TensorRT engine build を通す 1 PR に切るのがよいです。

## 現状整理

- 公開 wheel の `tensorrt_llm==1.2.0` では `qwen3_5` は足りません。
- main/source 側には `_torch` 経路での `qwen3_5` text モデル対応がすでに入っています。
- 一方で、legacy TensorRT backend には dense Qwen3.5 の実装が揃っていません。
- TensorRT backend での build 入口までは到達していますが、normalized した text mirror の HF weight loading で失敗しています。
- 現在実際に動いているのは「本物の TensorRT engine」ではなく、TRT-LLM の PyTorch backend を使った小さな text-only 実験です。

## 要するに何が足りないか

Qwen3.5 で real TensorRT engine を作るには、単なる model registration だけでは足りません。legacy な `tensorrt_llm.models/*` 系に対して、Qwen3.5 の hybrid linear-attention を表現できる model 実装と、それに対応する HF weight converter / loader が必要です。

## Upstream に出すときの切り方

- 1 本目の PR は `Qwen3.5 dense` の最小対応だけに絞る
- 1 本目では MoE、VL、FP8、AutoDeploy の整理、serve 周りは入れない
- 1 本目の成功条件は「single GPU / batch=1 / bf16 or fp16 / dense only で convert + build + generate が通ること」
- 量子化、MoE、VL は後続 PR に分ける

理由:

- TensorRT-LLM では新モデル対応自体は受け入れられるが、レビューしやすい増分に分ける方が通しやすい
- 今回の詰まり方も registration だけではなく converter / loader / hybrid layer 実装まで含むため、全部入りにすると論点が増えすぎる
- 実装順としても dense の最小 TensorRT engine build が先に通らないと、FP8 や MoE の問題を切り分けにくい

### 推奨する PR の分け方

#### PR 1: Qwen3.5 dense の最小サポート

- `architecture` / `MODEL_MAP` 追加
- `config.py` の Qwen3.5 dense 対応
- `model.py` の hybrid layer 対応
- `convert.py` / weight loader の Qwen3.5 dense 対応
- single GPU, batch=1, BF16 での tiny engine build
- 最低限の conversion / smoke test

#### PR 2: Qwen3.5 dense の量子化対応

- FP8 あるいは他の quantization path
- 精度比較
- パフォーマンス比較

#### PR 3: Qwen3.5 MoE 対応

- `Qwen3_5MoeForCausalLM`
- MoE 特有の weight conversion
- tp / ep / moe_tp / moe_ep の検証

#### PR 4: Qwen3.5-VL / multimodal 対応

- vision tower
- projector
- processor
- 画像入力 E2E

## 参考にした過去 PR

- [#5650 `[feat] Add TensorRT-Engine Qwen3 (dense) model support`](https://github.com/NVIDIA/TensorRT-LLM/pull/5650)
  - 今回の `PR 1` に最も近い型です
  - `models/__init__.py`、`config.py`、`model.py`、`convert.py` をまとめて追加し、dense の最小対応に絞って upstream に入っています
  - こちらの `Qwen3 dense` 追加を、`Qwen3.5 dense` に置き換えるイメージで差分を作るのが最もレビューされやすいです

- [#6344 `[FIX] fix bugs caused by None attention_bias during Qwen3 model convert engine`](https://github.com/NVIDIA/TensorRT-LLM/pull/6344)
  - 初回の model onboarding のあとに、変換まわりの不具合を小さい follow-up PR で直している例です
  - つまり upstream 的にも「最初の dense 対応を小さく入れて、その後に conversion bugfix を足す」進め方は自然です

- [#11394 `[#11440] [feat] AutoDeploy : Support Qwen3.5`](https://github.com/NVIDIA/TensorRT-LLM/pull/11394)
  - legacy TensorRT backend の PR ではありませんが、Qwen3.5 の Gated Delta / hybrid linear-attention 実装を読むための重要な参照です
  - cached prefill / decode、Qwen3.5 固有 config、テストの切り方はここを参考にできます
  - legacy backend 側へ port する際の「構造の元ネタ」として使います

- [#12302 `[TRTLLM-11544][feat] Add Qwen 3.5 supporting(NVFP4).`](https://github.com/NVIDIA/TensorRT-LLM/pull/12302)
  - `_torch` 側の `Qwen3.5` 対応が merge された PR で、今ローカルで見ている `qwen35` 系 source tree のベースにもなっています
  - `_Qwen35ConfigCompat`、dense / MoE 分岐、HF weight normalization など、今回 legacy backend に移植したい知見はここから拾うのが最短です
  - ただしこれは TensorRT engine backend ではなく `_torch` / AutoDeploy 系なので、PR の粒度や差分構成そのものは `#5650` の方を真似るべきです

補足:

- [#5913 `[fix] improve head_dim calculation in Qwen config`](https://github.com/NVIDIA/TensorRT-LLM/pull/5913) も関連 PR ですが、これは merge されず、後で `#6344` に吸収されています
- そのため「merged 済みの参考 PR」としては、まず `#5650` と `#6344` を見る方がよいです

## Qwen3.5 の upstream 状況

`Qwen3.5` 自体は、少なくとも 2026-02-20 以降 upstream に入り始めています。つまり「まだ誰も触っていない完全未対応モデル」ではありません。

### すでに merge 済みのもの

- [#11394 `[#11440] [feat] AutoDeploy : Support Qwen3.5`](https://github.com/NVIDIA/TensorRT-LLM/pull/11394)
  - 2026-02-20 merge
  - AutoDeploy 側の Qwen3.5 対応
  - Gated Delta / hybrid linear-attention、cached prefill / decode、MoE まわりの実装とテストが入っています

- [#12302 `[TRTLLM-11544][feat] Add Qwen 3.5 supporting(NVFP4).`](https://github.com/NVIDIA/TensorRT-LLM/pull/12302)
  - 2026-03-24 merge
  - `_torch` 側の Qwen3.5 dense / MoE 対応
  - 今ローカルで見ている `qwen35` 系 source tree のベースはこれです

- [#12114 `[#12290][fix] Qwen 3.5 fix 3d position ID handling`](https://github.com/NVIDIA/TensorRT-LLM/pull/12114)
  - 2026-03-25 merge
  - Qwen3.5 MoE / multimodal path の mRoPE / 3D position 修正
  - Qwen3.5 系の follow-up fix もすでに upstream で回り始めている、という意味で参考になります

### 2026-04-01 時点で open なもの

- [#12265 `[#11548][feat] AutoDeploy: Optimize Qwen3.5 perf`](https://github.com/NVIDIA/TensorRT-LLM/pull/12265)
  - Qwen3.5 の perf 最適化

- [#12611 `[None][feat] Add the Qwen3.5 multimodal support.`](https://github.com/NVIDIA/TensorRT-LLM/pull/12611)
  - Qwen3.5 multimodal 対応

- [#12646 `[None][feat] Add Qwen3.5 MTP support.`](https://github.com/NVIDIA/TensorRT-LLM/pull/12646)
  - Qwen3.5 MTP 対応

### この状況から言えること

- upstream は `Qwen3.5` を受け入れ始めている
- したがって、`Qwen3.5 dense` の追加自体は提案しやすい状態です
- ただし、今 merge されている主戦場は `_torch` / AutoDeploy / NVFP4 / MoE / multimodal 側です
- 今回こちらがやる `legacy tensorrt_llm.models` の real TensorRT engine 対応は、まだ別の gap を埋める作業です
- そのため PR の出し方としては、「Qwen3.5 全体対応」ではなく「legacy TensorRT backend の dense 最小対応」という説明に寄せる方が通しやすいです

## 関連する open PR / issue と重なり具合

2026-04-01 時点で確認した関連 open PR の範囲では、今回こちらがやろうとしている `legacy tensorrt_llm.models` の `Qwen3.5 dense` TensorRT engine 対応と、直接同じレイヤーを触っている open PR は見当たりませんでした。

### 直接はかぶっていない open PR

- [#12611 `[None][feat] Add the Qwen3.5 multimodal support.`](https://github.com/NVIDIA/TensorRT-LLM/pull/12611)
  - 変更ファイルは `_torch` 側のみ
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py`
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py`
  - `tensorrt_llm/_torch/models/modeling_qwen3vl.py`
  - `tensorrt_llm/_torch/pyexecutor/config_utils.py`
  - multimodal テスト
  - つまり、Qwen3.5 multimodal の PyTorch / AutoDeploy 側であり、`legacy tensorrt_llm.models` は触っていません

- [#12203 `[None][feat] Support Qwen3.5 Dense and MoE Models in Pytorch Backend`](https://github.com/NVIDIA/TensorRT-LLM/pull/12203)
  - open
  - `_torch/models/__init__.py`
  - `_torch/models/modeling_qwen3_5.py`
  - `_torch/models/modeling_qwen3_5_moe.py`
  - `_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`
  - `_torch/pyexecutor/*`
  - かなり近い参照実装ですが、やはり対象は PyTorch backend です

- [#12557 `[None][feat] Optimize GDN of Qwen3-Next/3.5; adds BF16 TRTLLM MoE`](https://github.com/NVIDIA/TensorRT-LLM/pull/12557)
  - draft open
  - `cpp` kernel と `_torch/modules/*` を中心に変更
  - GDN / MoE の実装参照としては有用ですが、legacy model onboarding の PR ではありません

- [#12265 `[#11548][feat] AutoDeploy: Optimize Qwen3.5 perf`](https://github.com/NVIDIA/TensorRT-LLM/pull/12265)
  - AutoDeploy の perf / sharding まわり
  - legacy TensorRT engine backend とは別です

- [#12646 `[None][feat] Add Qwen3.5 MTP support.`](https://github.com/NVIDIA/TensorRT-LLM/pull/12646)
  - MTP / speculative 系
  - `modeling_speculative.py` や `_torch/speculative/*` を触っており、今回の PR 1 とは別です

### 関連はあるが、対象が違う issue

- [#11947 `[Feature]: Support Dense Multi-Modal Qwen3.5 ... in trtllm-serve PyTorch/AutoDeploy backends`](https://github.com/NVIDIA/TensorRT-LLM/issues/11947)
  - `Qwen3_5ForConditionalGeneration` の route や `model_type: qwen3_5` 認識の話が出てきます
  - 症状は似ていますが、対象は `trtllm-serve --backend pytorch/_autodeploy` です
  - 今回のような `legacy tensorrt_llm.models` の real TensorRT engine build とは別の問題設定です

### この調査から言えること

- 概念的には `Qwen3.5` 周辺 PR と関連している
- ただし、現時点の open PR 群はほぼ `_torch` / AutoDeploy / multimodal / MTP / perf 側です
- 今回の `legacy tensorrt_llm.models` の dense TensorRT engine 対応は、公開されている open PR と直接の実装衝突は薄いと考えてよいです
- そのため upstream に出すときは、「既存の `_torch` 実装を参照しつつ、未着手の legacy TensorRT backend gap を埋める PR」と説明するのが適切です
- PR 本文では、非目標として `PyTorch backend`, `AutoDeploy`, `VL`, `MTP`, `MoE` を明示しておくとレビュー側が混乱しにくいです

## まず狙うべき最小スコープ

- 最初の対象は dense な text backbone のみ
- 入力 checkpoint は元の VLM checkpoint ではなく normalized 済みの text-only mirror
- dtype はまず BF16 のみ
- まずは tiny 設定だけを対象にする
  - `max_seq_len=127`
  - `max_batch_size=1`
  - `max_num_tokens=127`

ここでは「本当に 1 個でも TensorRT engine が作れるか」を最優先にして、MoE、FP8、VLM 全体対応は後回しにします。

## 実装が必要なもの

### 1. Legacy backend 側の architecture routing

- legacy `tensorrt_llm.models.MODEL_MAP` に `Qwen3_5ForCausalLM` を追加する
- 同じパスで MoE までやるなら `Qwen3_5MoeForCausalLM` も追加する
- normalized 済み `qwen3_5_text` config から、旧来の `QWenForCausalLM` 想定に落ちず、正しい model class に解決されるようにする

必要な理由:

- `_torch` 側ではすでに `qwen3_5` を理解している
- しかし legacy TensorRT model registry 側には dense Qwen3.5 用の実体クラスがまだない

### 2. `qwen3_5` 用の legacy config 対応

- `tensorrt_llm/models/qwen/config.py` の `QWenConfig.from_hugging_face` を拡張して、少なくとも以下を受けられるようにする
  - `qwen3_5`
  - `qwen3_5_text`
  - `qwen3_5_moe`
  - `qwen3_5_moe_text`
- VLM 由来 checkpoint の nested `text_config` 正規化を維持する
- Qwen3.5 text backbone に必要な以下の情報を config に残す
  - `layer_types`
  - `linear_key_head_dim`
  - `linear_value_head_dim`
  - `linear_num_key_heads`
  - `linear_num_value_heads`
  - `linear_conv_kernel_dim`
  - `full_attention_interval`
  - `partial_rotary_factor`
  - `rope_parameters` または正規化済みの `rope_scaling`
- text mirror で `mrope` 周りが壊れないようにする

必要な理由:

- 現在の legacy config path は `qwen3` と `qwen3_moe` までは受けますが、normalized checkpoint が使う Qwen3.5 の text model type を正しく扱えません
- config parse が通っても、hybrid layer を組み立てるための情報が今の legacy model path には足りません

### 3. Hybrid Qwen3.5 向けの legacy TensorRT model 実装

- 現在の dense `QWenDecoderLayer` をそのまま使う前提は捨てる
- `layer_types` を見て layer ごとに振る舞いを切り替えられる legacy TensorRT model path を実装する
- dense Qwen3.5 が使う 2 種類の layer をサポートする
  - `full_attention`
  - `linear_attention`
- layer index ごとに正しい block を選ぶ dispatch を入れる
- dense Qwen3.5 用の MLP path は維持する

必要な理由:

- 今の legacy `tensorrt_llm/models/qwen/model.py` は基本的に standard attention + MLP stack の前提で書かれている
- Qwen3.5 は hybrid architecture なので、旧来の Qwen/Qwen2/Qwen3 dense layer だけでは表現できません

### 4. Legacy 側の linear-attention module 対応

- Qwen3.5 の `linear_attn` を表現する TensorRT-side module 群を実装する
- 最低限、legacy backend で以下の weight を受けられる必要がある
  - `linear_attn.in_proj_qkvz`
  - `linear_attn.in_proj_ba`
  - `linear_attn.conv1d`
  - `linear_attn.dt_bias`
  - `linear_attn.A_log`
  - linear-attention 側の normalization
  - linear-attention 側の output projection
- 既存の lower-level 部品を流用するのはよいが、`layers/ssm.py` だけで足りる前提にはしない
- `_torch` 実行 path ではなく、engine build / runtime で必要な TensorRT backend の parameter layout に合わせる

必要な理由:

- 今詰まっているのは registration より深い層です
- legacy TensorRT model path には dense Qwen3.5 の hybrid linear-attention block がまだありません

### 5. Dense Qwen3.5 向けの HF weight loading / conversion

- `_torch` 側の Qwen3.5-specific な HF weight 正規化処理を legacy TensorRT の conversion / loading path に持ってくる
- VLM 由来 text weight の `model.language_model.` prefix を剥がす
- vision tensor を捨てる
- dense text mirror の layout を受けられるようにする
- 分割された linear-attention projection を TRT-LLM が期待する packed tensor に詰める
  - `q/k/v/z` または `qkv + z` -> `in_proj_qkvz`
  - `b + a` -> `in_proj_ba`
- dense MLP の key alias を維持する
  - `gate_proj`
  - `up_proj`
  - `down_proj`
- `ModelWeightsLoader` の postprocess で `None` が混ざらないよう、linear-attention weight mapping を追加する

必要な理由:

- 観測された TensorRT backend の失敗は config parse の前ではなく、legacy な HF weight loading の途中です
- `_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` に、legacy path にまだ移植されていない packing ルールがすでにあります

### 6. Engine build への統合

- `tensorrt_llm.commands.bench build` が新しい legacy TensorRT model path を実際に通るようにする
- text-only mirror から tiny BF16 engine を build する
- engine directory が実際に生成されることを確認する
- その engine に対して `trtllm-bench latency --backend tensorrt` が動くことを確認する

成功条件:

- workspace 配下に real engine が出力される
- 実行経路が PyTorch backend でも AutoDeploy の `torch-simple` でもなく、TensorRT になっている

## 後回しでよいもの

- MoE 対応
- FP8 / NVFP4 の loading
- VLM end-to-end 対応
- serve path の整理
- AutoDeploy 周りの整理
- tokenizer warning の解消
- tiny smoke test を超える long-context 対応

最初の目的が「real engine を 1 個通すこと」なら、これらは初回パスに入れない方がよいです。

## 1 本目の PR のゴール

1 本目は以下だけ満たせば十分です。

- Qwen3.5 dense text-only mirror を legacy TensorRT backend が読める
- HF weight conversion が通る
- tiny BF16 engine が build できる
- その engine に対して 1 サンプルの generate または latency smoke test が通る

1 本目では以下は非目標です。

- MoE
- VL
- FP8
- long context
- multi-node
- AutoDeploy の整理

## 実装順の提案

1. 既存の TensorRT-LLM source checkout から clean な worktree を切る
2. normalized 済み text-only mirror を引き続き入力 checkpoint として使う
3. 1 本目の PR は `dense only` に固定し、MoE / VL / FP8 をスコープから外す
4. dense `qwen3_5_text` 向けの legacy architecture registration と config 対応を入れる
5. Qwen3.5 hybrid layer を持つ dense な legacy TensorRT model class を追加する
6. legacy 側の linear-attention block と layer dispatch を実装する
7. Qwen3.5 の weight packing / normalization を legacy HF load path に移植する
8. tiny engine build を real engine が出るまで回す
9. tiny BF16 engine が通ってから、量子化、MoE、VL の順に後続 PR を検討する

## 触る可能性が高いファイル

- `tensorrt_llm/models/__init__.py`
- `tensorrt_llm/models/automodel.py`
- `tensorrt_llm/models/qwen/config.py`
- `tensorrt_llm/models/qwen/model.py`
- `tensorrt_llm/models/qwen/convert.py`
- `tensorrt_llm/models/model_weights_loader.py`
- `tensorrt_llm/models/` または `tensorrt_llm/layers/` 配下の新規 Qwen3.5 hybrid layer 実装ファイル

同じ source tree にある参照実装:

- `_torch` 側 model:
  `/media/masa/ssd_data/minipamayo_experiments/qwen35_trtllm_qwen35_commit/tensorrt_llm/_torch/models/modeling_qwen3_5.py`
- `_torch` 側の shared hybrid model:
  `/media/masa/ssd_data/minipamayo_experiments/qwen35_trtllm_qwen35_commit/tensorrt_llm/_torch/models/modeling_qwen3_next.py`
- `_torch` 側の Qwen3.5 HF mapper:
  `/media/masa/ssd_data/minipamayo_experiments/qwen35_trtllm_qwen35_commit/tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`

## 開発用作業ツリー

- 開発 clone:
  `/media/masa/ssd_data/minipamayo_experiments/qwen35_trtllm_fork`
- 作業ブランチ:
  `feat/qwen35-dense-minimal-trt`
- `origin`:
  `git@github.com:masahiro-kubota/TensorRT-LLM.git`
- `upstream`:
  `git@github.com:NVIDIA/TensorRT-LLM.git`

## PR 1 を実装タスクに落とすとこうなる

### 0. 先に固定する設計

- runtime 側は `layer_types` で実質 `attention` / `recurrent` しか見ていません。
- そのため、Qwen3.5 の生の `full_attention` / `linear_attention` をそのまま `layer_types` に入れるのではなく、build-time dispatch 用に別 field を持たせる方が安全です。
- 推奨は以下です。
  - `hybrid_layer_types`: `["full_attention", "linear_attention", ...]`
  - `layer_types`: `["attention", "recurrent", ...]`
- こうしておくと、build 時の layer dispatch と runtime の KV/recurrent state 管理を分離できます。
- recurrent state の各次元は `_torch/pyexecutor/_util.py` の `is_qwen3_hybrid(config)` ブランチに合わせる必要があります。
- ここは source からの推定を含むので、実装時に再確認する前提ですが、少なくとも以下の対応になります。
  - `conv_kernel = linear_conv_kernel_dim`
  - `state_size = linear_key_head_dim`
  - `rnn_head_size = linear_value_head_dim`
  - `rnn_hidden_size = linear_num_value_heads * linear_value_head_dim`
  - `rnn_conv_dim_size = 2 * linear_num_key_heads * linear_key_head_dim + linear_num_value_heads * linear_value_head_dim`
- PR 1 では unified loader を先に通すのが正解です。
- `TRTLLM_DISABLE_UNIFIED_CONVERTER=1` 側の完全対応は、最初の real engine build を通した後に詰めてもよいです。

### 1. `tensorrt_llm/models/automodel.py`

目的:

- `transformers.AutoConfig.from_pretrained(...)` だけでは `qwen3_5` を読めないので、Qwen3.5 専用の fallback loader を入れる

やること:

- raw の `config.json` を見て `model_type=qwen3_5*` または `Qwen3_5*` architecture を検知したら、通常の `AutoConfig` ではなく Qwen3.5 用の compat loader を通す
- nested `text_config` を展開した text-only config に正規化してから architecture 解決に進む
- `AutoConfig.from_hugging_face(...)` と `AutoModelForCausalLM.get_trtllm_model_class(...)` の両方で同じ判定が使われるようにする

完了条件:

- raw VLM config と text mirror config のどちらからでも、legacy 側で `Qwen3_5ForCausalLM` 相当の class 解決まで進める

### 2. `tensorrt_llm/models/__init__.py`

目的:

- legacy `MODEL_MAP` に Qwen3.5 dense 用の実体 class を載せる

やること:

- 新しい dense class を import する
- `MODEL_MAP` に `Qwen3_5ForCausalLM` を追加する
- PR 1 では MoE/VL は載せない

完了条件:

- `AutoModelForCausalLM.get_trtllm_model_class(...)` が Qwen3.5 dense に対して `NotImplementedError` を出さない

### 3. `tensorrt_llm/models/qwen/config.py`

目的:

- Qwen3.5 の config を legacy backend が理解できる形に正規化する

やること:

- `_torch/pyexecutor/config_utils.py` の `_Qwen35ConfigCompat.normalize(...)` を最小限移植する
- 少なくとも以下を正しく扱う
  - `qwen3_5`
  - `qwen3_5_text`
  - `Qwen3_5ForConditionalGeneration`
  - `Qwen3_5ForCausalLM`
- raw VLM config なら `text_config` を取り出す
- `rope_parameters` を flatten して legacy path でも使える形にする
- `layer_types` から runtime 用の generic `layer_types` と build 用の `hybrid_layer_types` を両方作る
- 以下の field を config に残す
  - `linear_key_head_dim`
  - `linear_value_head_dim`
  - `linear_num_key_heads`
  - `linear_num_value_heads`
  - `linear_conv_kernel_dim`
  - `full_attention_interval`
  - `partial_rotary_factor`
  - `rope_scaling`
  - `hybrid_layer_types`
  - `layer_types`
  - `conv_kernel`
  - `state_size`
  - `rnn_head_size`
  - `rnn_hidden_size`
  - `rnn_conv_dim_size`
- `qwen3` と同様に Q/K norm を使う path に入れるので、Qwen3.5 でも `attn_bias=False` と Q/K layernorm 前提を維持する

完了条件:

- `QWenConfig.from_hugging_face(...)` が Qwen3.5 text mirror から落ちずに config を返す
- 返った config に build 用と runtime 用の layer 情報が両方入っている

### 4. `tensorrt_llm/models/qwen/model.py`

目的:

- dense Qwen3.5 hybrid model を legacy TensorRT backend 上で表現する

推奨する切り方:

- 既存の `QWenForCausalLM` を大きく壊さず、Qwen3.5 用の class を追加する
- 例えば以下の単位に分ける
  - `QWen3_5LinearAttention`
  - `QWen3_5FullAttentionDecoderLayer`
  - `QWen3_5HybridDecoderLayer`
  - `QWen3_5Model`
  - `QWen3_5ForCausalLM`

やること:

- full-attention layer は現行 Qwen3/Qwen2 attention path を流用する
- linear-attention layer には最低限以下の parameter を持たせる
  - `in_proj_qkvz`
  - `in_proj_ba`
  - `conv1d`
  - `A_log`
  - `dt_bias`
  - gated RMS norm
  - `out_proj`
- linear-attention の TP layout は `_torch` 側に合わせる
  - `in_proj_qkvz`: column parallel
  - `in_proj_ba`: column parallel
  - `out_proj`: row parallel
- layer index ごとに `hybrid_layer_types[layer_idx]` を見て block を切り替える
- full-attention の `local_layer_idx` は「attention layer だけを数えた index」にする
  - これは `RecurrentGemma` と同じ罠で、単純な `layer_idx` を使うと KV cache index がずれる
- model の forward は KV cache と recurrent state の両方を扱う形にする
- `DecoderModelForCausalLM` をそのまま使うより、`RecurrentGemmaForCausalLM` や `MambaForCausalLM` に寄せて `prepare_inputs(...)` を自前で持つ方が安全
- `prepare_inputs(...)` では以下を同時に準備する
  - attention layers 向けの KV cache input
  - linear-attention layers 向けの `past_conv_state_*`
  - linear-attention layers 向けの `past_rnn_state_*`
  - `host_request_types`
  - `last_token_ids`
  - `slot_mapping` が必要ならその input
- output も以下を mark する
  - `present_key_value_*`
  - `present_conv_state_*`
  - `present_rnn_state_*`

完了条件:

- `network.set_named_parameters(model.named_parameters())` と `model.prepare_inputs(...)` が Qwen3.5 用 class で通る
- engine build 前に `prepare_inputs` 周りで詰まらない

### 5. Qwen3.5 専用 weight loader

候補ファイル:

- `tensorrt_llm/models/model_weights_loader.py`
- または新規に `tensorrt_llm/models/qwen/weight_loader.py`

推奨:

- いきなり global な `ModelWeightsLoader` を広くいじるより、Qwen3.5 専用 loader / helper を切った方が review しやすい

やること:

- `_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` のロジックを legacy path に移植する
- 少なくとも以下を入れる
  - `model.language_model.` prefix の剥離
  - `model.visual.*` tensor の破棄
  - split projection の pack
    - `in_proj_qkv + z` または `q/k/v/z` -> `in_proj_qkvz`
    - `b + a` -> `in_proj_ba`
  - dense MLP alias の維持
  - Q/K norm 関連 key の対応
- `weight_scale_inv` がある場合の扱いも Qwen3.5 用に揃える
- 最初の PR では BF16 を通すことが目的なので、FP8 packed qkvz の扱いは fallback 付きでもよい

完了条件:

- build log の `expected Tensor as element 0 in argument 0, but got NoneType` が消える
- `model.named_parameters()` をなめたとき、Qwen3.5 固有 weight で `None` が返らない

### 6. `tensorrt_llm/models/qwen/convert.py`

位置づけ:

- PR 1 の unblock という意味では最優先ではありません
- 現在の build failure は unified loader 側なので、まずはそちらを通すべきです

ただし:

- upstream test や将来の `TRTLLM_DISABLE_UNIFIED_CONVERTER=1` を考えるなら、最終的にはここにも同じ Qwen3.5 pack rule を入れる必要があります
- PR 1 で触るなら dense BF16 の最小 path のみに絞る

### 7. runtime で追加確認が要る点

- `runtime/generation.py` と `runtime/model_runner.py` は `layer_types` を `attention` / `recurrent` 前提で見ています
- そのため、config 側で generic `layer_types` を出せれば runtime 側の差分は最小化できます
- 逆に config 側で raw の `full_attention` / `linear_attention` を出す設計にすると runtime も広く直す必要があります
- PR 1 では runtime を大きく触らず済む設計に寄せる方がよいです

### 8. テストに落とす単位

最低限必要:

- config 正規化 unit test
  - raw VLM config から text config 抽出
  - `hybrid_layer_types` と generic `layer_types` の両立
  - recurrent state 用の各 dimension field の検証
- weight packing unit test
  - `q/k/v/z/b/a` -> `in_proj_qkvz` / `in_proj_ba`
  - `model.visual.*` の除外
- model class resolve test
  - `AutoModelForCausalLM.get_trtllm_model_class(...)` が Qwen3.5 dense class を返す
- TRT model smoke test
  - build 用の最小 config を組んで `prepare_inputs(...)` と network build が通る

置き場所の候補:

- `tests/unittest/trt/model/test_qwen35.py`
  - `test_mamba.py` と `test_nemotron_nas.py` を合わせたような形にする
- config 単体 test は `tests/unittest/others/test_pretrained_config.py` に寄せてもよいが、Qwen3.5 固有ロジックが多いので専用 test file の方が読みやすい

## 最初の smoke test コマンド

text mirror を使う前提で、最初の確認はこれで十分です。

```bash
python -m tensorrt_llm.commands.bench \
  --model Qwen/Qwen3.5-0.8B \
  --model_path /media/masa/ssd_data/minipamayo_experiments/qwen35_trtllm_main_runtime/artifacts/Qwen3.5-0.8B-text-mirror \
  --workspace /tmp/qwen35_bench_workspace \
  --log_level info \
  build \
  --max_seq_len 127 \
  --max_batch_size 1 \
  --max_num_tokens 127 \
  --trust_remote_code true
```

この段階で見るべきポイント:

- class resolve が Qwen3.5 dense 用 class に入るか
- config 正規化後の `layer_types` / `hybrid_layer_types` が想定通りか
- weight loader が `None` を返していないか
- `/tmp/qwen35_bench_workspace/.../tp_1_pp_1` に real engine が出るか

## 実装順のおすすめ

1. `automodel.py` と `qwen/config.py` を先に通して、class resolve と config parse を安定させる
2. `qwen/model.py` に Qwen3.5 hybrid class と `prepare_inputs(...)` を入れる
3. Qwen3.5 専用 weight loader を足して unified loader を通す
4. `tests/unittest/trt/model/test_qwen35.py` の最小版を先に書く
5. tiny build を回して engine directory が出るまで調整する
6. 最後に必要なら `qwen/convert.py` を整える

## 検証チェックリスト

- `transformers.AutoConfig` または fallback config loader で normalized text mirror を読める
- legacy `AutoModelForCausalLM` が新しい Qwen3.5 TensorRT model class を返す
- `ModelWeightsLoader` で `NoneType` failure を出さずに HF weight を load できる
- tiny BF16 engine build が成功する
- engine directory が空でない
- その engine に対して `trtllm-bench latency --backend tensorrt` が通る
- PR 1 の差分だけで dense single-GPU smoke test が再現できる

## 最初のパスでやらないこと

最初から元の `Qwen3_5ForConditionalGeneration` VLM checkpoint 全体を TensorRT で通そうとしない方がよいです。まず証明すべきなのは以下だけです。

- Qwen3.5 の text backbone を text-only checkpoint に正規化できる
- その checkpoint を legacy TensorRT backend で load できる
- real engine を build して実行できる
