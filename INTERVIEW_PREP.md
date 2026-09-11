# Interview Preparation Guide — Meeting Intelligence System

> Use this guide to answer questions in AI / Audio AI / MLSys interviews.
> For each question: **Simple explanation → Technical explanation → Project example.**

---

## AUDIO FUNDAMENTALS

### Q1. What is sampling rate?

**Simple:** Sampling rate is how many times per second we measure the height (amplitude) of a sound wave.

**Technical:** Audio is an analogue continuous signal. To store it digitally, we sample its amplitude at regular intervals. The sampling rate (in Hz or kHz) tells us how many samples are taken per second. By the Nyquist theorem, to accurately represent frequencies up to F Hz, we need a sampling rate of at least 2F Hz.

**Project:** We resample all audio to 16,000 Hz (16 kHz) in `src/audio/preprocessing.py`. Whisper was trained on 16 kHz; different rates would reduce accuracy.

---

### Q2. Why convert audio to 16 kHz?

**Simple:** Whisper was trained on 16 kHz audio. Feeding it a different rate is like giving it text in a different font — it can handle it, but it may make more mistakes.

**Technical:** Human speech contains most of its energy below 8 kHz. By Nyquist, 16 kHz sampling captures all frequencies up to 8 kHz — sufficient for speech. Higher sample rates (44.1 kHz, 48 kHz) just add unnecessary computation. Whisper's internal feature extractor (80-channel mel spectrogram) was designed for 16 kHz, so passing a different rate causes librosa to resample internally, which adds non-determinism and possible quality loss.

**Project:** `librosa.load(path, sr=16000)` handles resampling in a single call using a high-quality resampler.

---

### Q3. What is mono vs stereo?

**Simple:** Mono has one audio channel (one speaker playing from one direction). Stereo has two channels (left and right), giving a sense of spatial direction.

**Technical:** Meeting recordings often come in stereo. ASR models typically expect mono input. We convert to mono by averaging the two channels. This halves the array size and avoids ambiguity about which channel to process.

**Project:** `librosa.load(path, mono=True)` averages channels automatically. This is done in `preprocessing.py`.

---

### Q4. What is PCM?

**Simple:** PCM (Pulse Code Modulation) is the raw, uncompressed digital audio format. It's just a list of amplitude values.

**Technical:** In PCM, each sample is stored as a fixed-size integer (e.g., 16-bit = 65,536 possible values). It has no compression, no encoding overhead, and is the universal intermediate format. WAV is a container that typically holds PCM audio. Most ML audio pipelines work with float32 internally (-1.0 to 1.0) converted from 16-bit PCM.

**Project:** We save processed audio as PCM_16 WAV using `soundfile.write()`. The NumPy array used internally is float32.

---

### Q5. What is VAD?

**Simple:** VAD (Voice Activity Detection) is a classifier that looks at short windows of audio and decides: "Is someone speaking here, or is this silence/noise?"

**Technical:** VAD outputs binary labels (speech / non-speech) for each audio frame. Silero VAD uses a small RNN trained on diverse audio. It processes 512-sample windows at 16 kHz and outputs a speech probability for each window. We threshold at 0.5.

**Project:** `src/audio/vad.py` uses `torch.hub.load("snakers4/silero-vad")`. Returns a list of `SpeechSegment(start, end)` objects.

---

### Q6. Why use VAD?

**Simple:** A meeting might have 20 minutes of silence in a 60-minute recording. We skip those silent portions so Whisper doesn't have to process them.

**Technical:** VAD reduces ASR computation proportionally to the silence ratio. It also prevents Whisper from hallucinating repeated words or filler text on silence (a known Whisper quirk). Additionally, diarization models can confuse long silence with speaker changes.

**Important caveat:** VAD does NOT improve accuracy on speech segments — it only removes non-speech. Lowering the threshold causes more false positives (noise labelled as speech); raising it causes false negatives (quiet speech dropped).

---

## SPEECH RECOGNITION

### Q7. What is ASR?

**Simple:** ASR (Automatic Speech Recognition) converts spoken audio into written text.

**Technical:** Modern ASR systems are sequence-to-sequence neural networks. They take audio as input (represented as a spectrogram) and output a sequence of text tokens. Whisper is an encoder-decoder transformer trained on 680,000 hours of labelled audio.

**Project:** We use faster-whisper, which wraps Whisper's weights in CTranslate2 for 2–4× faster CPU inference. See `src/transcription/asr.py`.

---

### Q8. How does Whisper work at a high level?

**Simple:** Whisper converts audio to a picture (spectrogram), a neural network reads that picture and generates text word by word.

**Technical:**
1. **Feature extraction:** Audio → 80-channel log-mel spectrogram (time × frequency).
2. **Encoder:** 32 transformer layers process the spectrogram into embeddings (one per 20ms frame).
3. **Decoder:** Autoregressively generates text tokens, using cross-attention over encoder embeddings.
4. **Special tokens:** `<|transcribe|>`, `<|en|>`, `<|0.00|>` encode task, language, and timestamps.
5. **Language detection:** First 30 seconds → run through encoder → pick language with highest logit.

---

### Q9. Why use faster-whisper?

**Simple:** It's the same model as OpenAI's Whisper but 2–4× faster on CPU, using less memory.

**Technical:** faster-whisper uses CTranslate2, a C++ inference engine that applies int8 quantisation, fused operations, and memory layout optimisations. The model weights are the same (converted to CTranslate2 format). int8 quantisation reduces memory by 4× with minimal accuracy loss.

---

### Q10. What are transcription timestamps?

**Simple:** Each segment of text comes with a start and end time telling you when in the audio that text was spoken.

**Technical:** Whisper inserts special timestamp tokens (`<|0.00|>`, `<|0.02|>`, ...) during decoding. These tokens correspond to positions in the audio spectrogram. faster-whisper converts these to (start_seconds, end_seconds) per segment.

---

### Q11. What affects transcription accuracy?

- **Model size:** larger = more accurate, slower
- **Audio quality:** background noise, echo, reverb all degrade performance
- **Accent:** Whisper handles many accents; very strong regional accents are harder
- **Speaking rate:** fast speech is harder; long pauses sometimes cause hallucinations
- **Language:** English is most accurate; Indian languages vary
- **Overlapping speech:** Whisper sees a single audio stream — it can't separate concurrent speakers

---

## DIARIZATION

### Q12. What is speaker diarization?

**Simple:** Diarization answers: "Who was speaking at each moment in time?"

**Technical:** Diarization is a two-stage process:
1. **Speaker embedding:** Each audio frame is encoded into a fixed-size vector (x-vector or d-vector) that captures vocal characteristics.
2. **Clustering:** Frames with similar embeddings are grouped together → each cluster = one speaker. Common algorithms: agglomerative hierarchical clustering, spectral clustering.

Output: a set of intervals `(speaker_id, start, end)`.

---

### Q13. Diarization vs. speaker identification?

| | Diarization | Speaker Identification |
|---|---|---|
| Output | "Person A, Person B" (arbitrary labels) | "Alice, Bob" (actual names) |
| Requires | Just the meeting audio | Pre-built voice profiles |
| Use case | Unknown speakers | Known employee database |

**Project:** We use diarization. Labels are `SPEAKER_00`, `SPEAKER_01` → converted to `Speaker 1`, `Speaker 2`. The user knows who was on the call.

---

### Q14. How do you align diarization with ASR?

**Simple:** We find which speaker was talking during most of each transcribed segment.

**Technical:** For each Whisper segment `(asr_start, asr_end, text)`, we compute the time overlap with every diarization interval `(speaker, dia_start, dia_end)`:

```
overlap = min(asr_end, dia_end) - max(asr_start, dia_start)
```

The speaker with maximum overlap is assigned. This is implemented in `src/pipeline.py: align_speakers()`.

---

### Q15. What happens with overlapping speech?

When two speakers talk simultaneously:
- ASR hears a mixture — one or both voices may be transcribed, sometimes garbled
- Diarization assigns one speaker per interval (cannot label two simultaneously)
- The alignment assigns the majority speaker to the ASR segment

This is a fundamental limitation. Handling overlapping speech well requires source separation (e.g., SepFormer) — out of scope for this project.

---

## LLM ANALYSIS

### Q16. Why use an LLM after transcription?

**Simple:** Whisper gives us the words. An LLM understands what those words *mean* — it can extract decisions, action items, and write summaries.

**Technical:** LLMs are trained on vast text corpora and can perform instruction-following, information extraction, and text summarisation. Whisper is a sequence-to-sequence model specialised for audio → text; it has no semantic understanding. The LLM operates on the transcript text and returns structured JSON.

---

### Q17. How do you prevent hallucination?

1. **Explicit prohibition in the system prompt:** "NEVER invent information not present in the transcript."
2. **Null over guessing:** Prompt says use `null` if a deadline isn't mentioned.
3. **Evidence requirement:** Every action item must include an `evidence` field (verbatim quote).
4. **Low temperature:** `temperature=0.2` makes the model more conservative and deterministic.
5. **Schema validation:** Pydantic validates the JSON — if a field is wrong type, it's caught immediately.

---

### Q18. How do you handle long transcripts?

**Map-Reduce pattern:**
1. **Map:** Split transcript into chunks of N words. Summarise each chunk independently (plain text).
2. **Reduce:** Feed all chunk summaries to a final LLM call that produces structured JSON.

This avoids truncating context or feeding incomplete context to the model.

**Project:** `src/meeting/summarizer.py`, triggered when `word_count > settings.max_chunk_words` (default 3000 words).

---

### Q19. Why structured outputs?

**Simple:** We want JSON, not an essay. Structured output lets us display specific fields (action items table, decisions list) without parsing free text.

**Technical:** We instruct the LLM to return a specific JSON schema via the system prompt. The JSON is parsed by Python's `json.loads()` and then validated by Pydantic. Any type mismatch raises an exception, preventing corrupt data from reaching the UI.

---

### Q20. Why use Pydantic?

**Simple:** Pydantic validates data types at runtime. If the LLM accidentally returns a string where we expect a float, Pydantic catches it instead of silently breaking downstream code.

**Technical:** Pydantic v2 is built in Rust, extremely fast, and provides:
- Runtime type validation
- JSON serialisation/deserialisation (`.model_dump_json()`)
- Automatic documentation (schema introspection)
- Field validation with custom validators
- Settings management from environment variables (`pydantic-settings`)

---

## EVALUATION

### Q21. What is WER?

**Simple:** WER (Word Error Rate) measures what percentage of words were transcribed incorrectly.

```
WER = (Substitutions + Deletions + Insertions) / Total Reference Words
```

WER = 0.0 is perfect. WER = 0.1 means 10% of words were wrong.

---

### Q22. Substitutions, deletions, insertions?

- **Substitution:** wrong word transcribed ("morning" → "mourning")
- **Deletion:** word was missed entirely
- **Insertion:** extra word added that wasn't said

These are computed using the Levenshtein edit distance algorithm on word sequences.

---

### Q23. What is RTF?

```
RTF = processing_time / audio_duration
```

- RTF < 1.0 → faster than real-time (good)
- RTF = 0.2 → 5× faster than real-time
- RTF > 1.0 → slower than real-time (can't do live transcription)

**Project:** `evaluation/latency.py` computes RTF. A typical result on CPU with the base model: RTF ≈ 0.3–0.5.

---

### Q24. How do you measure system latency?

We use Python's `time.perf_counter()` (nanosecond precision). Each stage is wrapped in a `timed()` context manager from `src/utils.py` that records wall-clock time. Results saved to `data/outputs/<id>/timing.json`.

---

### Q25. Major failure cases?

1. **Background noise** → increased WER
2. **Overlapping speech** → wrong speaker assignment, garbled text
3. **Fast speech** → missed words (deletions)
4. **Code-switching** → language detection may pick wrong language
5. **Accents** → substitution errors on uncommon pronunciations
6. **Very quiet audio** → VAD false negatives, missed segments
7. **LLM hallucination** → invented deadlines/owners despite anti-hallucination prompting
8. **Long meetings** → map-reduce summaries may lose nuance

---

## SYSTEM DESIGN

### Q26. How would you make this real-time?

1. **Streaming ASR:** Use Whisper with streaming mode (process audio in chunks as it arrives).
2. **WebSocket server:** Replace Streamlit with FastAPI + WebSocket to push transcript updates live.
3. **VAD gating:** Only send frames with detected speech to the ASR model.
4. **Chunked diarization:** Run diarization on fixed sliding windows.

Trade-off: real-time mode typically has 2–5× higher WER than offline mode because the model can't look ahead.

---

### Q27. How would you support 1,000 simultaneous meetings?

1. **Horizontal scaling:** Deploy multiple ASR workers. ASR is compute-bound; add GPU instances.
2. **Queue:** Use a task queue (Celery + Redis or AWS SQS) between upload and processing.
3. **Separate services:** ASR, diarization, and LLM in separate microservices for independent scaling.
4. **Batching:** Batch multiple short audio segments into one Whisper call.
5. **Async LLM:** LLM calls are I/O-bound — use `asyncio` + `openai.AsyncClient`.

---

### Q28. How would you reduce inference cost?

1. **Smaller Whisper model:** `base` is 4× cheaper than `large-v3` with acceptable quality.
2. **GPU quantisation:** int8 on GPU reduces memory 4× and speeds up inference.
3. **VAD pre-filtering:** Skip silence → fewer audio frames → fewer compute cycles.
4. **LLM model selection:** GPT-4o-mini is 15× cheaper than GPT-4o for most extractions.
5. **Caching:** Cache diarization results for repeated uploads of the same file.

---

### Q29. How would you store transcripts?

- **Short term:** File system (current approach) → JSON in `data/outputs/`
- **Medium term:** PostgreSQL for transcripts + metadata; S3 for audio files
- **Large scale:** Elasticsearch for searchable transcript index

---

### Q30. How would you secure meeting recordings?

1. **Encryption at rest:** Encrypt audio files and transcripts with AES-256.
2. **Encryption in transit:** HTTPS only (TLS 1.3).
3. **Access control:** Per-meeting access tokens; only meeting participants can view results.
4. **Retention policy:** Auto-delete audio after processing (only keep transcripts).
5. **No external upload:** Process locally — never send audio to cloud without consent.

---

### Q31. How would you handle sensitive meeting data?

1. **On-premise deployment:** Don't use cloud LLM APIs for sensitive meetings. Use a local LLM (Ollama + Llama 3).
2. **Data minimisation:** Delete raw audio after transcription if only the transcript is needed.
3. **Audit logs:** Log who accessed which meeting transcript.
4. **GDPR compliance:** Provide delete endpoints. Transcripts containing personal data are PII.
5. **PII redaction:** Before storing, run a NER model to identify and redact names, phone numbers, etc.

---

## RESUME BULLETS

*(Use only bullets for features actually implemented and measured)*

```
• Built an end-to-end AI meeting intelligence pipeline converting multi-speaker audio
  into structured reports using faster-whisper ASR (auto-detected timestamps), Silero
  VAD (silence removal), and pyannote.audio speaker diarization.

• Implemented LLM-based meeting analysis with anti-hallucination prompting (null over
  guessing, evidence fields, Pydantic validation) to extract summaries, decisions,
  and action items from timestamped transcripts.

• Evaluated transcription and processing performance using WER (jiwer) and real-time
  factor across multilingual (English, Hindi, Marathi) and noisy meeting recordings;
  achieved RTF < 0.5 on CPU with the base Whisper model.
```
