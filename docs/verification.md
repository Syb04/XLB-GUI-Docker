# 拡張版の検証記録

この記録は熱・LESのGPU化前の検証です。GPU化後の最新実行・性能比較は [gpu-pipeline.md](gpu-pipeline.md) を参照してください。

2026-09-17、Windows + WSL2 Ubuntu 24.04。初版のCPU計算記録は [verification-initial.md](verification-initial.md) に保存しています。

## 実行環境

- NVIDIA GeForce RTX 3060 12 GB、Windowsドライバー595.79。
- Docker Engine 29.8.1、Compose 5.5.1、NVIDIA Container Toolkit 1.20.0。
- Dockerベース: Python 3.12 slim-bookworm。CPU/GPU両イメージを実際にビルドし、Gmshのネイティブ依存とコンテナ内CUDA演算を確認。
- XLB 0.3.2、JAX/JAXlib 0.11.1、CUDA 12用JAXプラグイン0.11.1。実際に解決された依存は [requirements-resolved-gpu.txt](requirements-resolved-gpu.txt)。
- 入力・ソルバー実装SHA-256・ライブラリバージョン・格子・計算場を各runに保存。GPU指定時には実際の分布関数の配置デバイスも記録。

## 数値精度の設定

最初のCPU/GPU比較では最大約5×10⁻⁴ m/sの速度差を検出しました。XLBのJAX内積がGPUで低精度化されないよう `jax_default_matmul_precision=highest` を明示した後、同一条件の再比較に合格しました。修正前の2実行は `data/development_archive/pre_fp32_gpu_comparison/` に保存しています。

## 再実行可能な確認

```sh
python -m unittest discover -s tests -v
node --check static/app.js
python -m scripts.verify_archives
python scripts/verify_gpu.py --url http://127.0.0.1:8766
python scripts/verify_cad_gpu.py --url http://127.0.0.1:8766
```

`verify_gpu.py` は3,000セル（流体2,600、固体400）のLES・共役熱伝達をCPU/CUDA各100ステップ実行し、速度・圧力・温度・渦動粘度を比較します。固体の熱伝導率は温度表です。

`verify_cad_gpu.py` は23度傾けたSTL流路をインポートし、異なるCAD面に速度入口・圧力出口・熱流束を与えてCUDAで60ステップ実行します。

機械可読な実測値とrun IDは [gpu-verification.json](gpu-verification.json) と [cad-gpu-verification.json](cad-gpu-verification.json)。条件・元CAD・計算結果を含むZIPは `examples/les_conjugate_gpu.xlb.zip` と `examples/rotated_cad_gpu.xlb.zip` です。

## 確認の範囲

- 最終ソースの自動テスト43件がすべて成功（WSL環境15.668秒、Docker CPUイメージ16.342秒、いずれもスキップなし）。JavaScript構文チェックも成功。
- 熱伝導の解析解、材料界面の熱抵抗、閉じた系の熱量保存、CAD面の熱流束の上書きと面積を自動テストで確認します。
- Smagorinsky係数の零ひずみ・正値性、CAD入口の温度移流、固体温度と流体専用場のマスク、ZIP保存、旧CADアセット互換性を検証します。
- GUIのLES設定、固体材料設定、固体を含む温度断面、渦動粘度表示をブラウザで確認。最終Docker版では、CAD面を直接クリックすると、その面の強調表示・ツリー選択・個別境界編集が一致することも確認。
- 3種類のサンプルZIPを独立した一時保存領域へ復元し、結果場、固体セル、CAD面参照、実装履歴を確認。

## 最終実行

|項目|結果|
|---|---|
|CPU LES・CHT run|`3ec689a39cec4b65803793992b0cc562`、100ステップ完了|
|CUDA LES・CHT run|`39c80d14f6db45eead976250b267464a`、100ステップ完了、実デバイス `cuda:0`|
|CUDA 斜めCAD run|`98ea89672eb84a0ebbdefb811013bf79`、60ステップ完了、実デバイス `cuda:0`|
|CPU/CUDA最大差|速度2.1083×10⁻⁷ m/s、温度3.0518×10⁻⁵ K、圧力6.3539×10⁻⁵ Pa|
|CADの実測場|最大速度0.00999999 m/s、最高温度293.2033 K（初期293.15 K）|

上記3実行の `provenance.json` に保存した4モジュールのSHA-256が最終ソースと一致することを確認しました。

`start-docker.ps1 -Gpu -LocalData` をWindows PowerShellから実行し、GPU版コンテナ `xlb_workbench-workbench-1` のhealth check成功と `http://127.0.0.1:8766/` のGUI表示を確認しました。保存データは既存の `./data` を引き継いでいます。検証用8767サーバーは停止済みです。

これは実装・実行・基本的な数値整合性の確認です。短時間の計算は未収束で、乱流の統計量、実験との一致、任意CADでの格子収束は検証していません。LBMはGPU、熱輸送・物性評価・前後処理はCPUです。LESは固定Csで、壁関数・壁面減衰モデルは含みません。
