# birdsongs-pitch-visualizer

鳥のさえずり音声（.wav）からピッチ（基本周波数）を抽出し、  
スペクトログラムとともに可視化する Streamlit アプリケーションです。

教育用途・研究用途の双方を想定し、  
鳥類行動観察や音響解析の入門ツールとして活用できます。

---

## 🎯 Features

- WAV 音声のアップロード
- 波形表示
- スペクトログラム表示
- ピッチ（基本周波数）抽出
- 時間変化の可視化
- （拡張予定）MIDI 変換

---

## 🖥 Demo

<img src="docs/screenshot.png" width="600">

※ スクリーンショットは後で差し替え

---

## 🚀 Getting Started

### 1. Clone

```bash
git clone https://github.com/<yourname>/birdsongs-pitch-visualizer.git
cd birdsongs-pitch-visualizer

2. Install

pip install -r requirements.txt

3. Run

streamlit run app.py

📂 Input Format

    .wav

    Mono / Stereo 可

    推奨：44.1 kHz / 48 kHz

📊 Output

    波形プロット

    スペクトログラム

    ピッチ曲線

🐦 Educational Applications

本ツールは以下の教育活動での利用を想定しています。

    高校物理：音波・周波数解析

    情報科：信号処理入門

    科学部：野鳥観察研究

    STEAM 教育：フィールド録音 × データ解析

🔬 Research Context

本プロジェクトは以下の研究テーマの一部として開発しています。

    鳥のさえずりにおける周波数文法解析

    個体識別と音響特徴量抽出

    音源定位デバイス（Sound Umbrella）との統合

🛠 Tech Stack

    Python

    Streamlit

    Librosa

    NumPy

    Matplotlib

📌 Roadmap

    MIDI 変換

    自動フレーズ分割

    種類分類モデル連携

    リアルタイム録音対応

📷 Sample Data

サンプル音声は sample_data/ に格納予定。

※ フィールド録音データ使用
👤 Author

Kentaro Negishi
Physics Educator / Birdsong Researcher / Maker
