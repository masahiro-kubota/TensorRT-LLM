# Qwen3.5-0.8B 向け upstream コントリビューション現実路線メモ

更新日: 2026-04-02 JST

## 調査対象

- repository: `NVIDIA/TensorRT-LLM`
- 対象モデル: `Qwen3.5-0.8B`
- 想定経路: **主流の `_torch` / PyTorch backend / AutoDeploy / multimodal path**
- 対象外: classic engine build 前提の整理

このメモでは、**「Qwen3.5-0.8B を upstream の主流路線で前に進めたいとき、どんな PR が現実的にマージされそうか」**を整理します。

## このメモでの作者分類ルール

作者の分類は、ユーザーが指定したルールに合わせます。

- **NVIDIA とみなす**
  - login に `nv` が入っている
  - GitHub profile の `company` に `@NVIDIA` または `NVIDIA` が入っている
- **非 NVIDIA とみなす**
  - 上記に当てはまらず、profile の `company` が別組織になっている
- **不明**
  - login に `nv` がなく、profile の `company` も空

補足:
- `company` が空の人は、実際の所属までは断定しません。
- ここでは **公開プロフィールから見える範囲**だけで分類します。

## 対象モデルを明確にする

ここで主に見ているのは **`Qwen3.5-0.8B` = dense VLM / VLA** です。

- top-level architecture: `Qwen3_5ForConditionalGeneration`
- 0.8B は **MoE ではなく dense**
- したがって、`Qwen3.5-35B-A3B` や `Qwen3.5-397B-A17B` の MoE/perf work と完全には一致しません

## まず結論

- **Qwen3.5 の core 実装は、かなり NVIDIA 側主導で進んでいる**と見てよいです。
- ただし、**非 NVIDIA の merge 実績はちゃんとあります**。
- その merge 実績を見ると、非 NVIDIA で通っているのは主に:
  - **スコープが非常に明確な model onboarding**
  - **再現性の高い小さい bugfix**
- あなたが今狙うべきなのは、**「Qwen3.5-0.8B 全対応の大PR」ではなく、主流の multimodal path の上で 0.8B を前に進める小さい fix / test PR** です。

## 0.8B そのものの upstream 状況

- 2026-04-02 時点で、`Qwen3.5-0.8B` を明示した open issue / open PR は確認できませんでした。
- `Qwen3.5-0.8B` で検索すると、実質的に近いのは以下の一般化された dense multimodal PR です。
  - [#12203 Support Qwen3.5 Dense and MoE Models in Pytorch Backend](https://github.com/NVIDIA/TensorRT-LLM/pull/12203)
  - [#12611 Add the Qwen3.5 multimodal support](https://github.com/NVIDIA/TensorRT-LLM/pull/12611)

結論:
- **0.8B 専用の upstream work item はまだ立っていない**
- なので、最初の貢献は **generic な Qwen3.5 dense multimodal path の小さい穴埋め**として出すのが自然です

## 今 open の Qwen3.5 関連 PR と作者

### core に近い open PR

| PR | 内容 | 作者 | 公開プロフィール上の分類 | コメント |
| --- | --- | --- | --- | --- |
| [#12203](https://github.com/NVIDIA/TensorRT-LLM/pull/12203) | Qwen3.5 dense + MoE の PyTorch backend support | `keddyjin` | NVIDIA | profile `company=@NVIDIA` |
| [#12611](https://github.com/NVIDIA/TensorRT-LLM/pull/12611) | Qwen3.5 multimodal support | `nv-guomingz` | NVIDIA | login に `nv`、profile `company=NVIDIA` |
| [#12646](https://github.com/NVIDIA/TensorRT-LLM/pull/12646) | Qwen3.5 MTP support | `nv-guomingz` | NVIDIA | 同上 |

### perf / infra / docs / registry 側

| PR | 内容 | 作者 | 公開プロフィール上の分類 | コメント |
| --- | --- | --- | --- | --- |
| [#12265](https://github.com/NVIDIA/TensorRT-LLM/pull/12265) | AutoDeploy: Optimize Qwen3.5 perf | `taylor-yb-lee` | 不明 | login に `nv` なし、profile `company` 空 |
| [#12557](https://github.com/NVIDIA/TensorRT-LLM/pull/12557) | GDN / BF16 TRTLLM MoE perf | `rosenrodt` | 不明 | login に `nv` なし、profile `company` 空 |
| [#12340](https://github.com/NVIDIA/TensorRT-LLM/pull/12340) | support matrix / AD docs | `bmarimuthu-nv` | NVIDIA | login に `-nv` |
| [#12221](https://github.com/NVIDIA/TensorRT-LLM/pull/12221) | BTK benchmark registry に Qwen3.5 NVFP4 追加 | `edenfunf` | 非 NVIDIA | profile `company=NFU` |
| [#12419](https://github.com/NVIDIA/TensorRT-LLM/pull/12419) | new sharding infra | `greg-kwasniewski1` | 不明 | login に `nv` なし、profile `company` 空 |

読み方:
- **Qwen3.5 dense multimodal の core 部分は、公開プロフィール上は NVIDIA 側がかなり握っている**
- 一方で、**registry / docs / 周辺整備は非 NVIDIA や所属不明の人も入っている**

## merge 済み Qwen3.5 PR が示していること

### 1. Qwen3.5 の大きい流れは core 実装 -> 小さい follow-up fix

| PR | merge 日 | 作者 | 分類 | changed files | 内容 |
| --- | --- | --- | --- | --- | --- |
| [#11728](https://github.com/NVIDIA/TensorRT-LLM/pull/11728) | 2026-02-26 | `bmarimuthu-nv` | NVIDIA | 1 file | Qwen3.5 cookbook |
| [#11877](https://github.com/NVIDIA/TensorRT-LLM/pull/11877) | 2026-03-04 | `bmarimuthu-nv` | NVIDIA | 1 file | text-only の `position_ids` plumbing fix |
| [#12242](https://github.com/NVIDIA/TensorRT-LLM/pull/12242) | 2026-03-20 | `rosenrodt` | 不明 | 22 files | initial Qwen3.5 text model support |
| [#12302](https://github.com/NVIDIA/TensorRT-LLM/pull/12302) | 2026-03-24 | `nv-guomingz` | NVIDIA | 9 files | Qwen3.5 support (NVFP4) |
| [#12114](https://github.com/NVIDIA/TensorRT-LLM/pull/12114) | 2026-03-25 | `bmarimuthu-nv` | NVIDIA | multimodal fix | Qwen3.5 MoE の 3D position ID fix |

読み方:
- Qwen3.5 系は、
  - **大きめの staff-driven onboarding**
  - その後の **小さい fix**
  という流れで進んでいます。
- つまり、外から入るなら **follow-up fix の形**が一番現実的です。

### 2. 小さい fix は実際にすぐ merge されている

特に分かりやすいのが [#11877](https://github.com/NVIDIA/TensorRT-LLM/pull/11877) です。

- 1 file
- 11 行追加
- 内容は `Qwen3.5` の `position_ids` plumbing を直すだけ
- 作成 2026-03-03
- merge 2026-03-04

これは、**「既存実装の上に載る小さい不具合修正」は非常に通りやすい**ことを示しています。

## 非 NVIDIA の merge 前例

### [#5650](https://github.com/NVIDIA/TensorRT-LLM/pull/5650) Add TensorRT-Engine Qwen3 (dense) model support

- merge 日: 2025-07-10
- 作者: `gkswns0531`
- profile: `Hanjun Cho`, `company=Allganize Korea`
- 分類: **非 NVIDIA**
- changed files:
  - `tensorrt_llm/models/__init__.py`
  - `tensorrt_llm/models/qwen/config.py`
  - `tensorrt_llm/models/qwen/convert.py`
  - `tensorrt_llm/models/qwen/model.py`

読み方:
- **非 NVIDIA でも model onboarding は merge される**
- ただし、この PR も
  - dense only
  - 4 files
  - 目的が明確
 という、かなりレビューしやすい形でした

### [#6344](https://github.com/NVIDIA/TensorRT-LLM/pull/6344) fix bugs caused by None attention_bias during Qwen3 model convert engine

- merge 日: 2025-07-29
- 作者: `Fan-Yunfan`
- profile: `Fan - Yunfan`, `company=Peking University`
- 分類: **非 NVIDIA**
- changed files:
  - `tensorrt_llm/models/qwen/config.py`
  - `tensorrt_llm/models/qwen/convert.py`

読み方:
- **非 NVIDIA の小さい具体的 bugfix も merge される**
- しかもこれは 2 files の修正で、典型的な「通りやすい PR」です

## ここから何が言えるか

- **非 NVIDIA の merge 実績はある**
- ただし、成功パターンはかなり明確です
  - `#5650`: 小さく切った onboarding
  - `#6344`: 再現性の高い bugfix
- 逆に、**Qwen3.5-0.8B VLA を end-to-end で大きく足す PR** は、現状の開発体制を見ると最初の一手としては重すぎます

## `Qwen3.5-0.8B` で現実的にマージされそうな PR

### A. 一番現実的

**既存の Qwen3.5 dense multimodal path に対する小さい bugfix**

候補の種類:
- `Qwen3_5ForConditionalGeneration` の routing / normalization bug
- multimodal wrapper と `Qwen3.5-0.8B` config の食い違い修正
- `position_ids` / `mRoPE` / placeholder metadata の不整合修正
- HF checkpoint の名前空間差異に対する narrow な修正

候補ファイル:
- `tensorrt_llm/_torch/pyexecutor/config_utils.py`
- `tensorrt_llm/_torch/models/modeling_qwen3vl.py`
- `tensorrt_llm/_torch/models/modeling_qwen3_5.py`

なぜ現実的か:
- パターンが [#11877](https://github.com/NVIDIA/TensorRT-LLM/pull/11877) と [#12114](https://github.com/NVIDIA/TensorRT-LLM/pull/12114) に近い
- 0.8B の実問題に直結する
- 大きい構造変更ではない

### B. merge しやすく、今後の土台にもなる

**`Qwen3.5-0.8B` の smoke / regression test 追加**

候補:
- `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`
- `tests/integration/defs/accuracy/references/*.yaml`

なぜ現実的か:
- maintainers は新しい code path にテストを求めている
- 既存 open PR [#12611](https://github.com/NVIDIA/TensorRT-LLM/pull/12611) も multimodal accuracy test を触っている
- `0.8B` 固有の再現を upstream に残せる

注意:
- これは **機能 fix が前提**です
- path がまだ壊れているなら、test-only ではなく fix + test が良いです

### C. 周辺整備として通しやすい

**support matrix / cookbook / registry の follow-up**

候補:
- `docs/source/models/supported-models.md`
- cookbook
- AutoDeploy registry config

参考:
- [#11728](https://github.com/NVIDIA/TensorRT-LLM/pull/11728)
- [#12221](https://github.com/NVIDIA/TensorRT-LLM/pull/12221)
- [#12340](https://github.com/NVIDIA/TensorRT-LLM/pull/12340)

なぜ現実的か:
- コア実装ほど重くない
- ただし **自分の目的である 0.8B の高速実行には直結しない**

## 最初の PR として現実的ではないもの

- `Qwen3.5-0.8B` VLA 全対応を 1 PR で出す
- multimodal + VLA + perf + docs + registry をまとめて出す
- GDN kernel / shared-stack / MTP / sharding infra に手を出す
- すでに open の [#12203](https://github.com/NVIDIA/TensorRT-LLM/pull/12203) や [#12611](https://github.com/NVIDIA/TensorRT-LLM/pull/12611) と実質同じ diff を並行で出す

理由:
- 今の mainline Qwen3.5 開発は core 実装がすでに動いている
- 外からの最初の PR としては、**大きい横断変更より、active work の隙間を埋める方が圧倒的に通しやすい**

## 今の状況からのおすすめ順

### 1. merge 確率を優先するなら

- **`Qwen3.5-0.8B` で再現する narrow bug を 1 つ潰す**
- fix 対象は 1 から 3 files
- 必ず regression test を付ける

この型が一番良いです。

### 2. 目的とのバランスを取るなら

- [#12611](https://github.com/NVIDIA/TensorRT-LLM/pull/12611) の延長として
- `Qwen3.5-0.8B` だけで落ちる dense multimodal bug を直す
- その bug に対応する smoke test を足す

これは **あなたの目的にも効くし、レビューもしやすい** です。

### 3. まず足場を作るなら

- docs / support matrix / cookbook / registry の PR

これは通しやすいですが、0.8B 実行には直接効きません。

## 実務的なおすすめ

最初の狙いはこれです。

**`Qwen3.5-0.8B` を current `_torch` / multimodal path で実際に流して、最小の failing point を 1 個見つけ、その点だけ直す PR を出す**

理想的な粒度:
- 1 から 3 files
- 1 つの明確な不具合
- 1 つ以上の test
- PR description に再現手順あり

この型なら、
- あなたの目的に効く
- maintainers にとってレビューしやすい
- 非 NVIDIA contributor の成功パターンにも合っています

## ひとことで言うと

**Qwen3.5-0.8B の主流路線で最初に狙うべきなのは、「全対応 PR」ではなく、「active な Qwen3.5 multimodal 実装の上で 0.8B だけが落ちる narrow bugfix + test」**です。

今の upstream の実態を見る限り、これが一番現実的です。
