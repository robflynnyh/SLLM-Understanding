# Audio Judge Benchmark Shortlist

This note lists benchmarks that are useful for evaluating speech/audio language
models as judges for conversational agents. The focus is not generic ASR or
sound classification. The useful judge should hear and rate naturalness, speech
quality, prosody, emotion, speaking style, and conversational interaction.

## Recommended Core Set

| Benchmark | What it measures | Why it is useful for judge evaluation |
| --- | --- | --- |
| [VoiceMOS Challenge](https://sites.google.com/view/voicemos-challenge/voicemos-challenge-2026) | MOS prediction for speech naturalness; the 2026 challenge includes emotion similarity, perceived emotion, and optional valence, arousal, and dominance ratings. | Closest match to "predict human perceptual ratings from speech." Useful for calibrating a judge against MOS-style human scores. |
| [SOMOS](https://innoetics.github.io/publications/somos-dataset/index.html) | Naturalness MOS for neural TTS samples. | Clean naturalness benchmark for synthetic speech. Good first sanity check for whether an SLLM can reproduce human naturalness judgments. |
| [BVCC / VoiceMOS 2022 data](https://zenodo.org/records/6572573) | MOS ratings for Blizzard Challenge, Voice Conversion Challenge, and ESPnet-TTS samples. | More diverse than SOMOS; useful for robustness across TTS and voice conversion systems. |
| [NISQA](https://github.com/gabrielmittag/NISQA) | Overall speech quality plus noisiness, coloration, discontinuity, and loudness. | Useful control benchmark for separating acoustic degradation from higher-level style, prosody, or emotion issues. |
| [MSP-Podcast SER Benchmark](https://lab-msp.com/MSP-Podcast_Competition/SERB/) | Naturalistic speech emotion recognition with categorical emotions and continuous affective attributes. | Strong naturalistic affect benchmark for judging emotion, valence, arousal, and dominance in conversational speech. |
| [ADU-Bench](https://adu-bench.github.io/) | Open-ended audio dialogue understanding, including ambiguity from intonation, pause, and emphasis. | Tests whether the judge hears when prosody changes the intended meaning of an utterance. |
| [SpeechRole-Eval](https://arxiv.org/html/2508.02013v7) | Speech role-play with dimensions such as fluency, naturalness, prosody consistency, emotion appropriateness, and role fidelity. | Very close to conversational-agent judging if the benchmark artifacts are usable. |
| [VoiceAssistant-Eval](https://github.com/mathllm/VoiceAssistant-Eval) | Voice assistant listening and speaking tasks, including audio naturalness, fluency, and emotion analysis. | Broad voice-agent benchmark with judge-relevant speaking dimensions. |
| [Full-Duplex-Bench](https://arxiv.org/html/2503.04721v3) | Pause handling, backchanneling, turn-taking, and interruption management. | Useful for judging interaction behavior rather than just single-turn audio quality. |
| [EmphAssess](https://aclanthology.org/2024.emnlp-main.30.pdf) | Prosodic emphasis preservation and recognition. | Narrow but valuable for testing whether an SLLM judge can detect emphasis and prosodic focus. |

## Secondary Benchmarks

| Benchmark | What it adds |
| --- | --- |
| [IEMOCAP](https://sail.usc.edu/iemocap/) | Classic acted emotional dialogue baseline. Useful, but less naturalistic than MSP-Podcast. |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | Controlled acted emotion and intensity clips from many speakers. Good for sanity checks. |
| [MELD](https://affective-meld.github.io/) | Conversational emotion with audio, video, and text. Useful when dialogue context matters. |
| [Mandarin Speech Prosody Benchmark](https://www.microsoft.com/en-us/research/publication/can-ai-understand-mandarin-speech-prosody-a-framework-and-benchmark-showcase/) | Prosody understanding in Mandarin across linguistically grounded tasks. Useful for non-English prosody, but language-specific. |

## Suggested Evaluation Axes

Use separate judge prompts and scores for each axis rather than one global
"quality" score:

- Naturalness: does the speech sound human-like and fluent?
- Audio quality: is the signal clean, intelligible, and free of artifacts?
- Prosody appropriateness: are rhythm, stress, pauses, pitch movement, and emphasis appropriate for the utterance?
- Emotion appropriateness: does the perceived emotion match the target conversational context?
- Style or persona consistency: does the speaking style match the requested role, persona, or brand?
- Conversation timing: does the agent handle pauses, interruptions, backchannels, and turn-taking naturally?
- Semantic-pragmatic fit: does the response answer appropriately given the spoken cues, not just the transcript?

## Suggested Metrics

For MOS-style and continuous ratings:

- Pearson correlation with mean human ratings.
- Spearman correlation with mean human ratings.
- MAE and RMSE on the target rating scale.
- System-level Pearson and Spearman when multiple samples come from the same TTS or agent system.
- Calibration by score bucket, especially for high-quality samples where most systems cluster.

For categorical or preference data:

- Accuracy against majority human label.
- Macro-F1 for emotion categories.
- Pairwise preference accuracy.
- Krippendorff alpha or weighted kappa for human agreement, so model performance can be interpreted against label reliability.

## Practical Stack

For this repo, a pragmatic initial stack would be:

1. VoiceMOS / BVCC / SOMOS for naturalness and MOS calibration.
2. MSP-Podcast or EmoNet for affective perception.
3. ADU-Bench for prosody-dependent meaning.
4. SpeechRole-Eval or VoiceAssistant-Eval for conversational-agent style and naturalness.
5. NISQA for acoustic-quality confound control.
6. EmphAssess for targeted prosody and emphasis testing.

The main caution is to avoid using only an emotion classifier benchmark. A
conversational-agent judge needs to separate acoustic quality, naturalness,
prosody, affect, persona, and interaction timing.
