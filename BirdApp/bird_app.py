# ============================================
#  🐦 BirdApp - 鳥のさえずりビジュアライザー
#  開発者: 根崎健太郎 (Tomiya High School)
#  初版: 2025年7月19日
# ============================================
#        ＿＾　フ
#        | 　_　_|
#      ＿` ミ＿xノ
#     /　　　　　　|
#    /　　 ノ　　ノ
#    ╊　　　|　|　|
# ＿‾|　　　|　|　|
# | (‾‾ノ∥_∥)__)__)
# ＿二つ
# ============================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.signal import butter, lfilter, spectrogram
from midiutil import MIDIFile
import io, zipfile, datetime, json

# ========= 基本関数 ========= #
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

def moving_average(arr, w):
    ret = np.copy(arr)
    for i in range(len(arr)):
        start = max(0, i - w//2)
        end = min(len(arr), i + w//2)
        valid = arr[start:end][~np.isnan(arr[start:end])]
        ret[i] = np.nan if len(valid) == 0 else np.mean(valid)
    return ret

def freq_to_midi(freq):
    return 69 + 12 * np.log2(freq / 440.0)

def midi_to_note_name(midi_num):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = int(midi_num // 12) - 1
    name = note_names[int(midi_num % 12)]
    return f"{name}{octave}"

def generate_wav_from_midi(freqs, bins, time_per_bin, rate):
    total_duration = bins[-1] + time_per_bin
    t = np.linspace(0, total_duration, int(rate * total_duration))
    signal = np.zeros_like(t)
    for i, freq in enumerate(freqs):
        if np.isnan(freq): continue
        start = int(bins[i] * rate)
        end = int((bins[i] + time_per_bin) * rate)
        end = min(len(t), end)
        signal[start:end] += 0.5 * np.sin(2 * np.pi * freq * t[start:end])
    if np.max(np.abs(signal)) > 0:
        signal /= np.max(np.abs(signal))
    return np.int16(signal * 32767)

# ここから Streamlit UI
st.title(":bird: 鳥のさえずりビジュアライザー (MIDI付き)")

uploaded_file = st.file_uploader("WAVファイルをアップロード", type=["wav"])
config = None
config_filename_warning = None

config_file = st.file_uploader("設定ファイル(JSON)", type=["json"])
if config_file:
    config = json.load(config_file)

if uploaded_file:
    rate, data = wavfile.read(uploaded_file)
    if len(data.shape) > 1:
        data = data[:, 0]

    duration = len(data) / rate
    time = np.linspace(0., duration, len(data))
    st.audio(uploaded_file, format="audio/wav")

    # input_filename チェック
    if config:
        expected_filename = config.get("input_filename")
        if uploaded_file.name != expected_filename:
            config_filename_warning = (
                f"⚠️ 設定ファイルには '{expected_filename}' を想定していますが、現在アップロードされたファイルは '{uploaded_file.name}' です。"
            )
        else:
            config_filename_warning = None  # 一致したら警告クリア

    if config_filename_warning:
        st.warning(config_filename_warning)

    dt_now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    location = st.text_input("場所", config.get("location") if config else "Sendai")
    species = st.text_input("鳥の種類", config.get("species") if config else "unknown")
    situation = st.text_input("状況", config.get("situation") if config else "dawn")
    base_filename = f"{dt_now}_{location}_{species}_{situation}"

    lowcut = st.slider("Low Cut (Hz)", 0, rate // 2, config.get("lowcut") if config else 5000)
    highcut = st.slider("High Cut (Hz)", 0, rate // 2, config.get("highcut") if config else 7000)
    t_range = st.slider("表示範囲 (秒)", 0.0, duration,
        tuple(config.get("t_range")) if config else (0.0, duration), step=0.1)
    cmap = st.selectbox("Colormap",
        ['gray', 'bone', 'cividis', 'viridis', 'plasma', 'magma', 'inferno', 'monochrome'],
        index=['gray', 'bone', 'cividis', 'viridis', 'plasma', 'magma', 'inferno', 'monochrome'].index(config.get("cmap")) if config else 0)
    threshold_ratio = st.slider("振幅しきい値", 0.0, 2.0, config.get("threshold_ratio") if config else 0.3, 0.05)
    smoothing_sec = st.slider("平滑化時間 (秒)", 0.01, 5.0, config.get("smoothing_sec") if config else 0.5, step=0.01)

    # monothre は常に表示し、有効/無効切り替え
    monothre_default = config.get("monothre") if config and "monothre" in config else -60
    monothre_disabled = (cmap != 'monochrome')
    monothre = st.slider("モノクロしきい値 (dB)", -100, 30, monothre_default, step=2, disabled=monothre_disabled, key="monothre_slider")

    # ===== 設定情報のエクスポート =====
    if st.button("設定をJSONでエクスポート"):
        config_out = {
            "input_filename": uploaded_file.name,
            "location": location,
            "species": species,
            "situation": situation,
            "lowcut": lowcut,
            "highcut": highcut,
            "t_range": [t_range[0], t_range[1]],
            "cmap": cmap,
            "threshold_ratio": threshold_ratio,
            "smoothing_sec": smoothing_sec
        }
        if cmap == 'monochrome':
            config_out["monothre"] = monothre

        config_json = json.dumps(config_out, indent=2)
        st.download_button(
            label="📂 JSONとして保存",
            data=config_json,
            file_name=base_filename + ".json",
            mime="application/json"
        )

    start_idx, end_idx = int(t_range[0] * rate), int(t_range[1] * rate)
    scoped_data = data[start_idx:end_idx]

    filtered = bandpass_filter(scoped_data, lowcut, highcut, rate)
    filtered /= np.max(np.abs(filtered))
    filtered_int16 = np.int16(filtered * 32767)

    time_f = np.linspace(0., len(filtered_int16)/rate, len(filtered_int16))

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(time, data)
    ax[0].set_title("Original Waveform")
    if cmap == 'monochrome':
        ax[1].specgram(data, Fs=rate, NFFT=1024, noverlap=512, cmap='gray')
    else :
        ax[1].specgram(data, Fs=rate, NFFT=1024, noverlap=512, cmap=cmap)
    ax[1].axhline(lowcut, color='red', linestyle='--')
    ax[1].axhline(highcut, color='orange', linestyle='--')
    ax[0].axvline(t_range[0], color='red', linestyle='--')
    ax[0].axvline(t_range[1], color='orange', linestyle='--')
    ax[1].axvline(t_range[0], color='red', linestyle='--')
    ax[1].axvline(t_range[1], color='orange', linestyle='--')
    ax[0].set_ylabel("Amplitude")
    ax[1].set_ylabel("Frequency (Hz)")
    ax[1].set_title("Original Spectrogram")
    st.pyplot(fig)

    # Filtered waveform 再生復活
    st.markdown("### フィルタ後の音声再生")
    filtered_buf = io.BytesIO()
    wavfile.write(filtered_buf, rate, filtered_int16)
    filtered_buf.seek(0)
    st.audio(filtered_buf.getvalue(), format="audio/wav")

    # 周波数追跡
    fig2, ax2 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax2[0].plot(time_f, filtered_int16)
    ax2[0].set_title("Filtered Waveform")
    ax2[0].set_ylabel("Amplitude")

    NFFT, noverlap = 1024, 512
    if cmap == 'monochrome':
        from scipy.signal import spectrogram
        freqs, bins, Sxx = spectrogram(filtered_int16, fs=rate, nperseg=NFFT, noverlap=noverlap)
        Sxx_dB = 10 * np.log10(Sxx + 1e-10)
        masked = np.ma.masked_less(Sxx_dB, monothre)
        ax2[1].imshow(masked, aspect='auto', extent=[bins[0], bins[-1], freqs[0], freqs[-1]], origin='lower', cmap='gray_r')
        Pxx = Sxx  # 後続の max_amps や np.argmax のために定義
    else :
        Pxx, freqs, bins, _ = ax2[1].specgram(filtered_int16, Fs=rate, NFFT=NFFT, noverlap=noverlap, cmap=cmap)
    max_amps = np.max(Pxx, axis=0)
    threshold = threshold_ratio * np.mean(max_amps)
    dominant_raw = np.where(max_amps >= threshold, freqs[np.argmax(Pxx, axis=0)], np.nan)

    time_per_bin = bins[1] - bins[0]
    win_size = max(1, int(smoothing_sec / time_per_bin))
    smoothed_freqs = moving_average(dominant_raw, win_size)

    ax2[1].plot(bins, dominant_raw, color='green', linewidth=1.0, label='Dominant Freq')
    ax2[1].plot(bins, smoothed_freqs, color='red', linewidth=1.5, label='Smoothed Freq')
    ax2[1].set_ylim(lowcut * 0.8, highcut * 1.2)
    ax2[1].set_ylabel("Frequency (Hz)")
    ax2[1].set_xlabel("Time (s)")
    ax2[1].set_title("Filtered Spectrogram")
    ax2[1].legend()
    st.pyplot(fig2)

    # 音階可視化（fig3）
    fig3, ax3 = plt.subplots(figsize=(12, 3))
    midi_pitches = np.array([freq_to_midi(f) if not np.isnan(f) else np.nan for f in smoothed_freqs])
    valid = ~np.isnan(midi_pitches)
    times_valid = bins[valid]
    midi_valid = midi_pitches[valid]
    note_names = [midi_to_note_name(int(round(p))) for p in midi_valid]

    ax3.scatter(times_valid, midi_valid, marker='|', color='blue', s=80)
    unique_midis = sorted(set(int(round(p)) for p in midi_valid))
    ax3.set_yticks(unique_midis)
    ax3.set_yticklabels([midi_to_note_name(p) for p in unique_midis])
    ax3.set_ylim(min(unique_midis) - 2, max(unique_midis) + 2)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Note")
    ax3.set_title("Detected Notes Over Time")
    ax3.grid(True)
    st.pyplot(fig3)

    # MIDI生成・再生
    st.markdown("### MIDI生成・保存・再生")
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 120)
    for i, freq in enumerate(smoothed_freqs):
        if np.isnan(freq): continue
        pitch = int(np.round(freq_to_midi(freq)))
        midi.addNote(0, 0, pitch, bins[i], time_per_bin, 100)

    midi_bytes = io.BytesIO()
    midi.writeFile(midi_bytes)
    midi_bytes.seek(0)

    # 簡易WAV生成（リアルタイム再生）
    midi_audio = generate_wav_from_midi(smoothed_freqs, bins, time_per_bin, rate)
    midi_buf = io.BytesIO()
    wavfile.write(midi_buf, rate, midi_audio)
    midi_buf.seek(0)
    st.audio(midi_buf.getvalue(), format="audio/wav")

    #st.download_button("🎵 MIDIファイルをダウンロード", midi_bytes.getvalue(), file_name=base_filename + "_midi.mid", mime="audio/midi")

    # 保存用zip作成
    raw_buf = io.BytesIO()
    wavfile.write(raw_buf, rate, data)
    raw_buf.seek(0)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w') as zf:
        zf.writestr(base_filename + '_raw.wav', raw_buf.read())
        zf.writestr(base_filename + '_filtered.wav', filtered_buf.getvalue())
        zf.writestr(base_filename + '_midi.mid', midi_bytes.getvalue())
        zf.writestr(base_filename + '_midi.wav', midi_buf.getvalue())
    zip_buf.seek(0)

    st.download_button("📦 ZIP一括保存", zip_buf.getvalue(), file_name=base_filename + ".zip", mime="application/zip")
