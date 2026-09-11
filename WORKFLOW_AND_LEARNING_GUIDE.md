# Meeting Intelligence System — Full Workflow & Learning Guide

This document is a comprehensive technical breakdown of the **Meeting Intelligence System**. It covers the complete end-to-end architecture, mathematical and algorithmic foundations, step-by-step pipeline execution, and a curated roadmap for learning and mastering Audio AI, Speech Processing, and LLM Engineering.

---

## 1. System Architecture & End-to-End Workflow

```
               [ Meeting Audio: WAV / MP3 / M4A / FLAC ]
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │ 1. Audio Ingestion &     │
                     │    Validation            │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │ 2. Signal Preprocessing  │
                     │    (16kHz, Mono, Norm)   │
                     └────────────┬─────────────┘
                                  │
               ┌──────────────────┴──────────────────┐
               ▼                                     ▼
┌──────────────────────────────┐       ┌──────────────────────────────┐
│ 3. Voice Activity Detection  │       │ 5. Speaker Diarization       │
│    (Silero VAD)              │       │    (pyannote.audio)          │
│    Detects Speech Segments   │       │    Embeddings + Clustering   │
└──────────────┬───────────────┘       └──────────────┬───────────────┘
               │                                      │
               ▼                                      │
┌──────────────────────────────┐                      │
│ 4. Automatic Speech Rec.     │                      │
│    (faster-whisper)          │                      │
│    Words + Timestamps        │                      │
└──────────────┬───────────────┘                      │
               │                                      │
               └──────────────────┬───────────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │ 6. Temporal Speaker      │
                     │    Alignment (Overlap)   │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │ 7. Speaker-Attributed    │
                     │    Transcript            │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │ 8. LLM Intelligence     │
                     │    (NVIDIA Llama 3.1)    │
                     │    Map-Reduce & Extract  │
                     └────────────┬─────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│ Summary       │         │ Action Items  │         │ Key Decisions │
│ & Topics      │         │ (Owner/Due)   │         │ & Insights    │
└───────────────┘         └───────────────┘         └───────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │ 9. UI & Export           │
                     │    (Streamlit / JSON /   │
                     │     Markdown / TXT)      │
                     └──────────────────────────┘
```

---

## 2. Detailed Pipeline Stages (What Happens Under the Hood)

### Stage 1: Audio Ingestion & Validation (`src/audio/metadata.py`)
- **Input:** Raw audio file (WAV, MP3, M4A, FLAC, OGG).
- **Process:** Reads metadata using `soundfile` without decoding the entire audio into memory.
- **Validation Checks:**
  - File size must not exceed the limit (e.g., 200MB).
  - Duration must meet the minimum threshold (e.g., > 1.0 second).
  - Format integrity check: rejects corrupt headers.
- **Output:** `AudioMetadata` object containing `duration_seconds`, `sample_rate`, `channels`, `format`, and `bit_depth`.

### Stage 2: Signal Preprocessing (`src/audio/preprocessing.py`)
Audio recorded in conference rooms or Zoom calls comes in varying sample rates (44.1kHz, 48kHz) and stereo channels. Preprocessing standardizes the signal:
1. **Stereo to Mono Conversion:**
   $$x_{\text{mono}}[n] = \frac{x_{\text{left}}[n] + x_{\text{right}}[n]}{2}$$
   Prevents phase-cancellation issues and halves computational memory.
2. **Resampling to 16,000 Hz (16 kHz):**
   - Both OpenAI Whisper and pyannote are pre-trained on 16kHz audio. Resampling avoids pitch shifting or model degradation.
   - Uses polyphase or Fourier filter resampling via `scipy.signal` / `librosa`.
3. **Peak Normalization:**
   - Scales the waveform so the peak amplitude reaches $-1.0 \text{ dBFS}$ to prevent clipping while maintaining a uniform dynamic range.

### Stage 3: Voice Activity Detection (VAD) (`src/audio/vad.py`)
- **Model:** Silero VAD (a lightweight PyTorch neural network operating on 30ms audio windows).
- **Purpose:** Identifies contiguous chunks where human speech is present, discarding background static, air conditioner hum, and long silences.
- **Output:** List of `SpeechSegment(start_seconds, end_seconds)`.

### Stage 4: Automatic Speech Recognition (ASR) (`src/transcription/asr.py`)
- **Engine:** `faster-whisper` (reimplementation of OpenAI Whisper using CTranslate2).
- **Why faster-whisper instead of vanilla Whisper?**
  - Up to **4x faster** with **8x less memory** via 8-bit quantization (`int8` on CPU, `float16` on GPU).
- **Decoding Strategy:**
  - Audio is converted into an 80-channel log-Mel spectrogram.
  - Encoder extracts acoustic representations.
  - Autoregressive Decoder generates text tokens with timestamp predictions (`<|0.00|>`, `<|0.02|>`).
- **Output:** List of `TranscriptSegment(start, end, text, confidence)`.

### Stage 5: Speaker Diarization ("Who Spoke When") (`src/diarization/speaker_diarization.py`)
- **Engine:** `pyannote.audio 3.1`.
- **Pipeline:**
  1. **Segmentation:** SincNet + convolutional layers segment audio into short frames.
  2. **Speaker Embedding:** Generates high-dimensional acoustic embeddings (e.g., 512-dim d-vectors) capturing unique vocal tract features.
  3. **Clustering:** Agglomerative Hierarchical Clustering (AHC) groups similar vectors into distinct clusters (`SPEAKER_00`, `SPEAKER_01`).
- **Output:** Time intervals mapped to speaker labels: `[start, end, speaker_label]`.

### Stage 6: Temporal Alignment Algorithm (`src/pipeline.py` -> `align_speakers`)
ASR produces text with boundaries $[t_{w,\text{start}}, t_{w,\text{end}}]$, while Diarization produces speaker intervals $[t_{d,\text{start}}, t_{d,\text{end}}]$. Because these models run independently, their timestamps do not align on exact boundaries.

**The Solution — Temporal Overlap Intersection:**
For each transcription segment $W$, we find all diarization segments $D_i$ that intersect with it:
$$\text{Overlap}(W, D_i) = \max(0, \min(W_{\text{end}}, D_{i,\text{end}}) - \max(W_{\text{start}}, D_{i,\text{start}}))$$

The speaker with the **maximum accumulated overlap duration** wins the segment.
- If no diarization segment overlaps, the speaker defaults to `"Unknown"`.
- **Output:** `AlignedSegment(start, end, speaker, text)`.

### Stage 7: LLM Intelligence Layer (`src/meeting/summarizer.py`)
- **Provider:** **NVIDIA NIM API** (`integrate.api.nvidia.com/v1`) using model **`meta/llama-3.1-70b-instruct`** (or OpenAI fallback).
- **Token Handling & Map-Reduce:**
  - Short meetings (<12,000 words) are processed in a single prompt.
  - Long meetings are split into chunks. A **Map** step summarizes each section, followed by a **Reduce** step synthesizing all chunks into a coherent executive report.
- **Structured JSON Extraction:**
  The LLM is prompted with strict JSON schema instructions to return:
  1. **Executive Summary**: Core meeting purpose and outcomes.
  2. **Key Discussion Points**: Grouped by topic.
  3. **Action Items**: Array of `{task, owner, deadline, confidence}`.
  4. **Decisions Made**: Explicit agreements reached during the meeting.

### Stage 8: Evaluation Metrics (`evaluation/`)
1. **Word Error Rate (WER):**
   $$\text{WER} = \frac{S + D + I}{N}$$
   Where:
   - $S$ = Substitutions (wrong word transcribed)
   - $D$ = Deletions (word omitted)
   - $I$ = Insertions (extra word added)
   - $N$ = Total words in reference ground truth
   Computed using dynamic programming (Levenshtein distance algorithm in `evaluation/wer.py`).
2. **Real-Time Factor (RTF):**
   $$\text{RTF} = \frac{\text{Wall-Clock Processing Time (seconds)}}{\text{Total Audio Duration (seconds)}}$$
   - An RTF of $0.15$ means a 60-minute meeting is processed in just 9 minutes.

---

## 3. How to Master This Field (Step-by-Step Learning Roadmap)

To build and explain systems like this in top-tier AI/Audio engineering interviews, follow this targeted learning journey:

### Step 1: Foundations of Digital Audio & Signal Processing
- **Concepts to master:**
  - **Sample Rate & Nyquist-Shannon Theorem:** Why 16kHz preserves frequencies up to 8kHz (sufficient for human speech).
  - **Bit Depth & Dynamic Range:** 16-bit PCM vs 32-bit float.
  - **Time-Domain vs Frequency-Domain:** Fourier Transform (FFT) and Short-Time Fourier Transform (STFT).
  - **Log-Mel Spectrograms:** How human auditory perception (Mel scale) maps onto audio feature representations.
- **Hands-on practice:**
  - Load audio with `soundfile` and `librosa`.
  - Plot waveforms and Mel spectrograms using `matplotlib`.

### Step 2: Voice Activity Detection (VAD)
- **Concepts to master:**
  - Energy thresholding vs Machine Learning VAD.
  - How Silero VAD operates on small window frames (30ms/60ms) with minimal latency.
- **Hands-on practice:**
  - Benchmark Silero VAD against noisy audio recordings.

### Step 3: Speech-to-Text & Automatic Speech Recognition (ASR)
- **Concepts to master:**
  - **Acoustic Models:** Connectionist Temporal Classification (CTC) vs Encoder-Decoder Transformers.
  - **OpenAI Whisper Architecture:** 80-channel Mel spectrogram input $\to$ Conv1D feature extractor $\to$ Transformer Encoder $\to$ Autoregressive Decoder with timestamp tokens.
  - **CTranslate2 & faster-whisper:** INT8/FP16 quantization, KV caching, beam search decoding, temperature scheduling.
  - **Evaluation:** Word Error Rate (WER) and Character Error Rate (CER).
- **Hands-on practice:**
  - Compare transcription speed and accuracy across `tiny`, `base`, and `small` models.

### Step 4: Speaker Diarization
- **Concepts to master:**
  - **Voice Biometrics & Speaker Embeddings:** X-vectors, d-vectors, and ResNet embeddings.
  - **Metric Learning:** Cosine similarity and angular loss functions.
  - **Clustering Algorithms:** Agglomerative Hierarchical Clustering (AHC) vs Spectral Clustering.
  - **Diarization Error Rate (DER):** Missed speech, false alarm, speaker confusion.
- **Hands-on practice:**
  - Run `pyannote/speaker-diarization-3.1` on two-speaker conversations.

### Step 5: Audio-Text Alignment & Fusion
- **Concepts to master:**
  - Temporal overlap logic and greedy intersection.
  - Forced alignment models (e.g., WhisperX using wav2vec2 cross-attention phoneme alignment).
- **Hands-on practice:**
  - Implement and test custom alignment algorithms handling overlaps and speaker interruptions.

### Step 6: LLM Engineering & Information Extraction
- **Concepts to master:**
  - **Prompt Engineering for Long Contexts:** System prompts, role enforcement, few-shot formatting.
  - **Structured Outputs:** Enforcing valid JSON schemas for entity extraction (action items, deadlines, owners).
  - **Context-Length Strategies:** Map-Reduce summarization vs Refine vs Hierarchical Chunking.
  - **Model Serving & NIM APIs:** NVIDIA NIM endpoints, token rate limits, and latency optimization.
- **Hands-on practice:**
  - Experiment with prompts extracting implicit decisions vs explicit decisions from raw transcripts.

### Step 7: System Performance & Production Deployment
- **Concepts to master:**
  - **Real-Time Factor (RTF) optimization:** Profiling GPU vs CPU bottlenecks.
  - **Batching & Thread Pools:** Decoupling ASR and Diarization into parallel workers.
  - **Modern UI Design:** Building responsive, black-themed analytical dashboards with Streamlit.

---

## 4. How to Create and Test This Project Locally

```bash
# 1. Clone & enter repository
cd "e:\Audio Project"

# 2. Activate virtual environment
.\venv\Scripts\activate

# 3. Configure .env file
# Add:
# NVIDIA_API_KEY=nvapi-...
# HUGGINGFACE_TOKEN=hf_...

# 4. Run automated unit test suite
pytest -v

# 5. Launch dashboard
streamlit run app/app.py
```

---

## 5. Interview Cheat Sheet: 5 Common Questions & Concise Answers

1. **Why do we resample all audio to 16 kHz mono?**
   > *Answer:* Whisper and pyannote are pre-trained on 16 kHz single-channel audio. Converting to 16 kHz mono eliminates channel phase discrepancies, halves memory consumption, and guarantees alignment with the models' input feature shapes.

2. **Why use `faster-whisper` over Hugging Face or OpenAI's original implementation?**
   > *Answer:* `faster-whisper` leverages CTranslate2's custom C++ inference engine, supporting 8-bit quantization and efficient KV caching. It yields 4x faster throughput and 8x reduced RAM usage with zero loss in WER.

3. **How do you assign speaker tags to text when ASR and Diarization run independently?**
   > *Answer:* We perform interval overlap intersection between word/phrase timestamps from ASR and speaker time-segments from Diarization. The speaker occupying the longest temporal overlap within that interval is assigned to the segment.

4. **How do you handle meetings exceeding LLM context windows?**
   > *Answer:* We apply a Map-Reduce pipeline: the transcript is chunked into logical temporal segments, each chunk is summarized independently in parallel (Map), and the summaries are aggregated into a final global synthesis (Reduce).

5. **What is Real-Time Factor (RTF)?**
   > *Answer:* RTF measures execution efficiency: $\text{Processing Time} / \text{Audio Duration}$. An RTF $< 1.0$ indicates faster-than-real-time throughput.
