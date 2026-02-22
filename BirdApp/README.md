# BirdApp - 鳥のさえずりビジュアライザー

## 🐦 概要

本アプリは、鳥のさえずり（WAVファイル）を読み込み、周波数解析とMIDI変換によって可視化・再生するツールです。生徒への自然観察やICT活用の教材としても有効じゃないかなと思います。

---

## 🔧 使用方法

### ① Python環境の準備

以下のいずれかを使ってPythonの環境を準備してください：

* Anaconda（推奨）
* Python 3.9 以上（Windows/macOS/Linux対応）

### ② 必要なライブラリをインストール（初回のみ）

以下の `install_env.bat` をダブルクリックすると自動でライブラリをインストールできます：

```
BirdApp/
├─ install_env.bat
```

または、手動で以下のコマンドを実行してください：

```
pip install -r requirements.txt
```

### ③ フォルダ構成

以下のようにファイルを配置してください：

```
BirdApp/
├─ bird_app.py              ← アプリ本体
├─ sample.wav               ← サンプル音声
├─ sample.json              ← 設定ファイル（オプション）
├─ start_bird_app.bat       ← ダブルクリックで起動
├─ install_env.bat          ← ライブラリ一括インストール
├─ requirements.txt         ← 必要なライブラリ一覧
├─ output/                  ← 処理結果の保存フォルダ（自動生成）
└─ README.md                ← この説明書
```

### ④ アプリの起動

`start_bird_app.bat` をダブルクリックすると、アプリがブラウザで開きます。

---

## 📝 補足事項

* `sample.json` を使うと各種パラメータの事前設定が可能です。
* 処理後の結果（フィルター後WAV、MIDI、設定）は `output/` フォルダにまとめて保存されます。
* `.bat` ファイルは Windows 環境用です。macOS/Linux の場合は `streamlit run bird_app.py` をターミナルで実行してください。

---

## 🧑‍💻 開発者

* 根崎健太郎（宮城県 富谷高校）
* 初版：2025年7月19日

---

## 🐤 ライセンス

教育目的での利用・改変は自由です。再配布の際はクレジットを明記してください。
