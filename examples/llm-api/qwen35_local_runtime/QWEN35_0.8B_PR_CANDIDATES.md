# Qwen3.5-0.8B 向け upstream PR 候補メモ

更新日: 2026-04-02 JST

## 前提

- 対象モデル: `Qwen3.5-0.8B`
- 想定経路: `tensorrt_llm/_torch` / PyTorch backend / AutoDeploy / multimodal path
- 手元では `Qwen3.5-0.8B` を source tree 上で動かせている
- ただし、そのまま upstream に「Qwen3.5-0.8B VLA 完全対応」として出せる状態ではない

このメモでは、**手元での bring-up 経験から upstream に切り出しやすい PR を、現実的な粒度で整理する**。

## 結論

最初の 1 本は、**0.8B を動かすために必要だった 1 個の明確な不具合修正**に絞るのがよい。

特に現実的なのは次の 3 本。

1. `Qwen3.5` VLM checkpoint の config 認識 fallback fix
2. dense `Qwen3_5ForConditionalGeneration` の multimodal routing fix
3. dense Qwen3.5 の weight mapper 名寄せ fix

この順で、`1` が最も独立性が高く、最初の PR として出しやすい。

## PR 候補 1

### 概要

`Qwen3_5ForConditionalGeneration` のような VLM checkpoint を読んだとき、nested `text_config` を正しくたどって正規化済み text config として扱えるようにする。

### 変更候補ファイル

- `tensorrt_llm/models/automodel.py`
- `tensorrt_llm/llmapi/llm_args.py`

### なぜ通しやすいか

- 2 files の narrow bugfix に落とせる
- open の `#12203` や `#12611` と直接の diff 衝突が小さい
- 手元での `Qwen3.5-0.8B` bring-up に実際に効いた修正で、再現理由を書きやすい

### 何を主張する PR か

- top-level `Qwen3_5ForConditionalGeneration` でも config 判定を早期に落とさない
- `AutoConfig` / model format detection が `qwen3_5_text` 系の normalized config を受け取れるようにする

### PR タイトル案

- `[None][fix] Fallback to normalized Qwen3.5 text config for Qwen3.5 VLM checkpoints`

### 注意

- これは `Qwen3.5-0.8B` VLA 全体対応を意味しない
- 「config discovery / model routing の初期段を直す PR」として切るのが正しい

## PR 候補 2

### 概要

`Qwen3_5ForConditionalGeneration` を multimodal wrapper が正しく dense text backbone にルーティングできるようにする。

### 変更候補ファイル

- `tensorrt_llm/_torch/models/modeling_qwen3vl.py`
- `tests/unittest/_torch/modeling/test_modeling_qwen3vl.py`

### 背景

現状の `Qwen3VLModel` は、少なくとも手元 tree 上では次だけを明示的に扱っている。

- `Qwen3VLForConditionalGeneration`
- `Qwen3VLMoeForConditionalGeneration`

一方で、dense `Qwen3_5ForConditionalGeneration` はその分岐に入っていない。

### なぜ通しやすいか

- `#12611` の自然な follow-up になる
- multimodal routing の穴埋めとして意味が明確
- synthetic config ベースの unit test も書きやすい

### PR タイトル案

- `[None][fix] Route dense Qwen3.5 multimodal checkpoints in Qwen3VLModel`

### 注意

- `#12611` が未 merge の間は、実質的にその PR と競合しやすい
- 出すなら `#12611` の merge 後、または author と調整してからが安全

## PR 候補 3

### 概要

dense Qwen3.5 の HF checkpoint が持つ MLP weight 名を、loader が期待する形に narrow に吸収する。

### 変更候補ファイル

- `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`

### なぜ通しやすいか

- 1 file bugfix に落としやすい
- 「特定 checkpoint layout 差異の吸収」というレビューしやすい形になる
- 手元の 0.8B bring-up に直結している

### PR タイトル案

- `[None][fix] Normalize dense Qwen3.5 MLP names in HF weight mapper`

### 注意

- `#12203` も同系統ファイルを触っている
- そのため、最初の PR としては `PR 候補 1` より競合しやすい

## どれを最初に出すべきか

### 第一候補

**PR 候補 1: config 認識 fallback fix**

理由:

- 独立性が最も高い
- diff が小さい
- 手元再現を書きやすい
- upstream の active PR と重複しにくい

### 第二候補

**PR 候補 2: dense multimodal routing fix**

理由:

- `Qwen3.5-0.8B` の実利用に近い
- ただし `#12611` とタイミングが近い

### 第三候補

**PR 候補 3: weight mapper fix**

理由:

- 小さい bugfix としては良い
- ただし `#12203` と近い

## 批判的評価

ここからは、**「手元で役に立ったか」ではなく、「maintainer にとってありがたい小PRか」**という観点で評価する。

### PR 候補 1: config 認識 fallback fix

評価: **低め**

理由:

- `Qwen3.5-0.8B` の bring-up には効くが、速度改善や multimodal 実行に直接は効かない
- `automodel.py` と `llm_args.py` という global entrypoint を触るため、diff の見た目より責任範囲が広い
- 本質的には `Qwen3.5` config registry 周りの過渡的な互換 workaround に見えやすい
- `#12203` や `#12611` の本流が進むほど、価値が相対的に下がりやすい

判断:

- **最初の upstream 実績を作るための小PR**としては悪くない
- ただし、**maintainer が特に欲しがっている修正**とは言いにくい

### PR 候補 2: dense multimodal routing fix

評価: **中から高**

理由:

- `Qwen3.5-0.8B` の利用形に最も近い
- `#12611` の review で、reviewer が dense model gap を明示的に指摘している
- `modeling_qwen3vl.py` と unit test の narrow fix に落としやすい
- `Qwen3.5` multimodal の active work の穴埋めとして説明しやすい

判断:

- **maintainer にとってありがたい小PR**に最も近い
- ただし条件付きで、**`#12611` merge 後の follow-up**として出すのが筋がよい
- `#12611` が open の間は競合しやすいので、今すぐ独立 PR にする価値は下がる

### PR 候補 3: weight mapper 名寄せ fix

評価: **低から中**

理由:

- 1 file bugfix に見せやすいので、形としては悪くない
- ただし、その修正が `main` 上の public checkpoint で再現する failure に結び付かないと弱い
- 現状の説明だと、手元の `Qwen3.5-0.8B` bring-up 向け compatibility patch に見えやすい
- `#12203` が近いファイルを触っているため、別 PR の独立価値を出しにくい

判断:

- **public repro を最小で出せるなら** maintainer にとってありがたい小PRになる
- そこまで落とせないなら、優先度は高くない

## maintainer に喜ばれやすい小PRの型

今回の `Qwen3.5` 周辺で見る限り、通りやすいのは次の型。

- reviewer が open PR 上で既に指摘している gap を埋める
- 1 から 2 files と test で閉じる
- public checkpoint か synthetic config で再現できる
- local container 事情や環境依存の glue を持ち込まない

過去の前例もこの傾向に合っている。

- `#5650`: dense を先に切り出して merge、MoE は別扱い
- `#6344`: concrete な conversion failure の小さな bugfix
- `#11877`: active line 上の narrow fix が短期間で merge

## maintainer 観点での更新後の結論

**「独立して出しやすいPR」**と **「maintainer に喜ばれやすいPR」** は一致しない。

- 独立性だけで見ると、`PR 候補 1` はまだ出しやすい
- maintainer 価値で見ると、`PR 候補 2` が最も筋がよい
- `PR 候補 3` は public repro が作れるかどうかで価値が大きく変わる

このため、**信頼を作るための最初の小PR**という観点では、現時点では次の順で考えるのがよい。

1. `#12611` merge 後の dense multimodal routing + unittest
2. `#12611` review で見えている mRoPE / test-memory 周りの narrow follow-up
3. public repro を添えた weight mapper bugfix
4. config fallback fix

## 今は避けるべき PR

- `Qwen3.5-0.8B` VLA 全対応を 1 PR で出す
- local container 依存の import guard をまとめて upstream に持ち込む
- AutoDeploy config の local pruning を upstream に出す
- non-Ray fallback など、環境依存の glue code を最初の PR にする
- `#12203` や `#12611` と実質同じ diff を別 PR として出す

## upstream に出すときの書き方

最初の PR では、手元で 0.8B を動かしたことを次のように使うのがよい。

- `Qwen3.5-0.8B` で再現する具体的な failure point を提示する
- その failure point だけを直す
- 1 個以上の regression test を付ける
- PR 本文では「full support」ではなく「specific fix」として説明する

避けるべき書き方:

- `Add full Qwen3.5-0.8B support`
- `Enable Qwen3.5 VLA end-to-end`

推奨する書き方:

- `Fix config discovery for Qwen3.5 VLM checkpoints`
- `Fix dense Qwen3.5 multimodal routing`
- `Fix Qwen3.5 dense HF weight-name compatibility`

## 実務的なおすすめ

次にやるべきことはこれ。

1. `PR 候補 1` を branch に切り出す
2. その修正だけで再現できる minimal test を付ける
3. `#12611` / `#12203` の進行を見ながら、`PR 候補 2` か `3` を follow-up にする

最初の一本としては、**`automodel.py` と `llm_args.py` だけで完結する config 認識 fix** が最も現実的。
