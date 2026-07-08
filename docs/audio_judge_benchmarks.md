# Audio Judge Benchmark Shortlist

This note lists benchmarks that are useful for evaluating speech/audio language
models as judges for conversational agents. The focus is not generic ASR or
sound classification. The useful judge should hear and rate naturalness, speech
quality, prosody, emotion, speaking style, and conversational interaction.

## Recommended Core Set

Scale numbers are approximate. "Not reported" means I could not find an
official public number on the linked project page or paper; for Hugging Face
repositories, storage is the hosted artifact size, not necessarily an expanded
local checkout.

| Benchmark | What it measures | Public split/set scale | Approx hours | Download/storage size | Why it is useful for judge evaluation |
| --- | --- | --- | --- | --- | --- |
| [VoiceMOS Challenge](https://sites.google.com/view/voicemos-challenge/voicemos-challenge-2026) | MOS prediction for speech naturalness; the 2026 challenge includes emotion similarity, perceived emotion, and optional valence, arousal, and dominance ratings. | Varies by challenge year. The 2022/BVCC release has main train/dev/test = 4,974/1,066/1,066 samples plus OOD labeled/unlabeled/dev/test = 136/540/136/540 samples. | Not reported. | 2022 Zenodo seed package is 286.9 MB; full recreation may be larger because Blizzard samples are not redistributed. | Closest match to "predict human perceptual ratings from speech." Useful for calibrating a judge against MOS-style human scores. |
| [SOMOS](https://innoetics.github.io/publications/somos-dataset/index.html) | Naturalness MOS for neural TTS samples. | 20,000 synthetic utterances plus 100 natural utterances; official train/validation/test split is 70%/15%/15%. | Not reported. | [Zenodo archive](https://zenodo.org/records/7378801) is 4.0 GB. | Clean naturalness benchmark for synthetic speech. Good first sanity check for whether an SLLM can reproduce human naturalness judgments. |
| [BVCC / VoiceMOS 2022 data](https://zenodo.org/records/6572573) | MOS ratings for Blizzard Challenge, Voice Conversion Challenge, and ESPnet-TTS samples. | Main train/dev/test = 4,974/1,066/1,066 samples; OOD labeled/unlabeled/dev/test = 136/540/136/540 samples. | Not reported. | 286.9 MB seed package; full recreation may be larger because Blizzard samples are not redistributed. | More diverse than SOMOS; useful for robustness across TTS and voice conversion systems. |
| [NISQA](https://github.com/gabrielmittag/NISQA) | Overall speech quality plus noisiness, coloration, discontinuity, and loudness. | 14,672 files total: TRAIN_SIM 10,000; VAL_SIM 2,500; TRAIN_LIVE 1,020; VAL_LIVE 200; TEST_LIVETALK 232; TEST_FOR 240; TEST_NSC 240; TEST_P501 240. | Official total hours not reported; using the published 6-12s segment range gives ~24.5-48.9 h. | [DepositOnce ZIP](https://depositonce.tu-berlin.de/items/b8908103-b0e8-4912-8144-aea65098fa1f) is 8.89 GB. | Useful control benchmark for separating acoustic degradation from higher-level style, prosody, or emotion issues. |
| [MSP-Podcast SER Benchmark](https://lab-msp.com/MSP-Podcast_Competition/SERB/) | Naturalistic speech emotion recognition with categorical emotions and continuous affective attributes. | Version 2.0: train/dev/test1/test2/test3 = 169,190/34,399/46,294/14,822/3,200 speaking turns. | 409 h total. | GB not reported on public page; academic-license download. | Strong naturalistic affect benchmark for judging emotion, valence, arousal, and dominance in conversational speech. |
| [ADU-Bench](https://adu-bench.github.io/) | Open-ended audio dialogue understanding, including ambiguity from intonation, pause, and emphasis. | Paper reports 20,715 dialogues across ADU-General, ADU-Skill, ADU-Multilingual, and ADU-Ambiguity; the public HF viewer currently exposes a 210-row train view. | Not reported. | [HF artifact](https://huggingface.co/datasets/KuofengGao/ADU-Bench) is 2.21 GB. | Tests whether the judge hears when prosody changes the intended meaning of an utterance. |
| [SpeechRole-Eval](https://arxiv.org/html/2508.02013v7) | Speech role-play with dimensions such as fluency, naturalness, prosody consistency, emotion appropriateness, and role fidelity. | Eval benchmark over 98 roles, with single-turn and multi-turn test data; SpeechRole-Data has train/test splits and 111k-112k dialogues. | Not reported. | [SpeechRole-Data](https://huggingface.co/datasets/yuhui1038/SpeechRole-Data) is 499 GB; SpeechRole-Eval hosted size is not cleanly reported by the public API/page. | Very close to conversational-agent judging if the benchmark artifacts are usable. |
| [VoiceAssistant-Eval](https://github.com/mathllm/VoiceAssistant-Eval) | Voice assistant listening and speaking tasks, including audio naturalness, fluency, and emotion analysis. | Test-only configs across 13 tasks: 4 listening, 8 speaking, and 1 viewing config; 10,497 examples. | Not reported. | [HF artifact](https://huggingface.co/datasets/MathLLMs/VoiceAssistant-Eval) is 9.49 GB: listening 5.29 GB, speaking 2.74 GB, viewing 1.46 GB. | Broad voice-agent benchmark with judge-relevant speaking dimensions. |
| [Full-Duplex-Bench](https://arxiv.org/html/2503.04721v3) | Pause handling, backchanneling, turn-taking, and interruption management. | Benchmark versions vary; v1.5 reports a 99-sample test set, and v3 reports 21 real-human-speech scenarios. | Not reported. | GB not reported on public page/papers. | Useful for judging interaction behavior rather than just single-turn audio quality. |
| [EmphAssess](https://aclanthology.org/2024.emnlp-main.30.pdf) | Prosodic emphasis preservation and recognition. | Test benchmark: 3,652 speech samples from 913 transcripts rendered in 4 voices. | Not reported. | Dataset tarball is ~0.21 GB. | Narrow but valuable for testing whether an SLLM judge can detect emphasis and prosodic focus. |

## Secondary Benchmarks

| Benchmark | What it adds | Public split/set scale | Approx hours | Download/storage size |
| --- | --- | --- | --- | --- |
| [IEMOCAP](https://sail.usc.edu/iemocap/) | Classic acted emotional dialogue baseline. Useful, but less naturalistic than MSP-Podcast. | 5 dyadic sessions; common evaluation is leave-one-session-out rather than a fixed official train/test split. | ~12 h. | GB not reported on official page; full audiovisual package includes video, speech, mocap, and transcripts. |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | Controlled acted emotion and intensity clips from many speakers. Good for sanity checks. | 7,442 clips from 91 actors; no official train/dev/test split. | Not reported. | Full original repo checkout is large because of media/LFS; an [audio-only HF mirror](https://huggingface.co/datasets/myleslinder/crema-d) is 471 MB. |
| [MELD](https://affective-meld.github.io/) | Conversational emotion with audio, video, and text. Useful when dialogue context matters. | Train/dev/test = 9,989/1,109/2,610 utterances, 13,708 utterances total across 1,433 dialogues. | Not reported. | GB not reported on public page. |
| [Mandarin Speech Prosody Benchmark](https://www.microsoft.com/en-us/research/publication/can-ai-understand-mandarin-speech-prosody-a-framework-and-benchmark-showcase/) | Prosody understanding in Mandarin across linguistically grounded tasks. Useful for non-English prosody, but language-specific. | 8 prosody-understanding tasks; no public train/test counts found. | Not reported. | GB not reported on public page/paper. |

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
