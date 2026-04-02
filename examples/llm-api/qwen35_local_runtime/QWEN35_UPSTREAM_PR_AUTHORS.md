# Qwen3.5 Upstream PR と作者メモ

## 目的

このメモは、`TensorRT-LLM` における `Qwen3.5` 関連 PR が「誰によって」「どのような流れで」出されているかを、公開情報ベースで整理するためのものです。

注意:

- ここに書くのは GitHub 上で公開されている情報に基づく観察です
- `-nv` のようなアカウント名から NVIDIA 関係者らしいと推測できるものはありますが、雇用関係そのものを断定するものではありません
- 現時点では、Qwen / Alibaba 公式の人だと明示できる PR は確認できていません

## 結論

- `Qwen3.5` 関連 PR は、少なくとも 2026-02-20 以降 upstream に merge され始めています
- 最近の `Qwen3.5` 系 PR は、公開アカウント名の見え方としては NVIDIA 側の人が主導しているように見えます
- 一方で、`Qwen3` dense の追加には community contributor の merge 実績もあります
- したがって、`Qwen3.5 dense` の legacy TensorRT backend 対応も、粒度を合わせれば upstream proposal として十分現実的です

## 確認した PR

### merge 済みの Qwen3.5 関連 PR

#### [#11394 `[#11440] [feat] AutoDeploy : Support Qwen3.5`](https://github.com/NVIDIA/TensorRT-LLM/pull/11394)

- merge 日: 2026-02-20
- 作者: `bmarimuthu-nv`
- 公開アカウント名の見え方としては NVIDIA-affiliated と考えるのが自然です
- 内容:
  - AutoDeploy 側の Qwen3.5 対応
  - Gated Delta / hybrid linear-attention
  - cached prefill / decode
  - MoE とテスト

#### [#12302 `[TRTLLM-11544][feat] Add Qwen 3.5 supporting(NVFP4).`](https://github.com/NVIDIA/TensorRT-LLM/pull/12302)

- merge 日: 2026-03-24
- 作者: `nv-guomingz`
- これも公開アカウント名の見え方としては NVIDIA 側アカウントに見えます
- 内容:
  - `_torch` 側の Qwen3.5 dense / MoE 対応
  - NVFP4 含む Qwen3.5 サポート
  - 今ローカルで見ている `qwen35` 系 source tree のベースに相当

#### [#12114 `[#12290][fix] Qwen 3.5 fix 3d position ID handling`](https://github.com/NVIDIA/TensorRT-LLM/pull/12114)

- merge 日: 2026-03-25
- 作者: `bmarimuthu-nv`
- 内容:
  - Qwen3.5 MoE / multimodal path の mRoPE / 3D position 修正
  - `Qwen3.5` 系が upstream で継続的に保守され始めていることを示す follow-up fix

### 2026-04-01 時点で open な Qwen3.5 関連 PR

#### [#12265 `[#11548][feat] AutoDeploy: Optimize Qwen3.5 perf`](https://github.com/NVIDIA/TensorRT-LLM/pull/12265)

- state: open
- 作者: `taylor-yb-lee`
- 内容:
  - Qwen3.5 の perf 最適化

#### [#12611 `[None][feat] Add the Qwen3.5 multimodal support.`](https://github.com/NVIDIA/TensorRT-LLM/pull/12611)

- state: open
- 作者: `nv-guomingz`
- 内容:
  - Qwen3.5 multimodal 対応

#### [#12646 `[None][feat] Add Qwen3.5 MTP support.`](https://github.com/NVIDIA/TensorRT-LLM/pull/12646)

- state: open
- 作者: `nv-guomingz`
- 内容:
  - Qwen3.5 MTP 対応

## community contributor の前例

### [#5650 `[feat] Add TensorRT-Engine Qwen3 (dense) model support`](https://github.com/NVIDIA/TensorRT-LLM/pull/5650)

- merge 日: 2025-07-10
- 作者: `gkswns0531`
- これは `Qwen3` であって `Qwen3.5` ではありません
- ただし、`legacy TensorRT engine` 向けの dense model onboarding が community contributor から merge された重要な前例です

関連 issue:

- [#5673 `Support for Qwen3 model in TensorRT-LLM Engine`](https://github.com/NVIDIA/TensorRT-LLM/issues/5673)
  - issue 本文で、作者自身が「PR を出した後に contribution guideline に従って issue を作った」と書いています
  - そのため、少なくともこのケースは NVIDIA 内製だけでなく community contribution の流れとして読めます

## このメモから言えること

- `Qwen3.5` は upstream に受け入れられ始めている
- 最近の `Qwen3.5` 実装 PR は NVIDIA 側の人が主導しているように見える
- しかし、model onboarding 自体は community contributor の merge 実績がある
- よって、こちらが狙う `legacy tensorrt_llm.models` の `Qwen3.5 dense` 対応も、PR の粒度を小さく保てば upstream として不自然ではありません

## 今回の PR 1 への示唆

- 実装内容の型は `#5650` を最も強く参考にする
- `Qwen3.5` 固有の実装知見は `#11394` と `#12302` を参照する
- 説明の仕方としては「Qwen3.5 全体対応」ではなく「legacy TensorRT backend の dense 最小対応」に寄せる
- community contributor の既存成功パターンに沿って、issue と PR を review しやすい単位で出す
