# In-Context ASR Results

This note records the corrected in-context-asr evaluation for
`OpenMOSS-Team/MOSS-Audio-4B-Instruct` as a normal transcription task.

## Setup

- Dataset: `robflynnyh/in-context-asr`
- Local checkout: `/exp/exp4/acp21rjf/in-context-asr`
- Items: 20
- Requests: 40, one `sentence_without_repeat.wav` and one
  `sentence_with_repeat.wav` request per item
- Model: `OpenMOSS-Team/MOSS-Audio-4B-Instruct`
- Local model path: `/store/store5/acp21rjf/models/MOSS-Audio-4B-Instruct`
- Generation: deterministic, `temperature=0`, `max_new_tokens=256`
- Prompt mode: plain transcription

The same prompt is used for every request:

```text
Transcribe the speech in this audio. Return only the transcript.
```

Targets and repeat separators are not included in the prompt. They are stored
only as metadata for scoring after generation.

Scoring follows the original repo's logic: normalize the model transcript and
search for the target word or accepted variant. For `sentence_with_repeat.wav`,
the normalized transcript is first split on the first matching repeat separator,
then the target is searched before and after that separator.

## Results

| Parsed | Separators found | Correct in clear repeat | Correct in corrupt repeat | Correct without repeat |
| ---: | ---: | ---: | ---: | ---: |
| 40 / 40 | 18 / 20 | 90.000 | 80.000 | 45.000 |

## Notes

The earlier target-aware JSON-probe run is superseded by this result. That run
leaked the target word and asked for target-presence booleans, so it was not a
normal ASR evaluation.

MOSS often transcribed the conversational repeat-request phrase even for
`sentence_without_repeat.wav`, for example:

```text
in_context_asr__row-001__without_repeat:
The professor's pedagogical approach mythologized critical thinking over rote memorization. I think the line broke up. Could you repeat that? Sure thing.

in_context_asr__row-001__with_repeat:
The professor's pedagogical approach emphasized critical thinking over rote memorization. I think the line broke up. Could you repeat that? Sure thing. The professor's pedagogical approach emphasized critical thinking over rote memorization.

in_context_asr__row-002__without_repeat:
His proclivity for collecting to a lucrative business venture. I think the line broke up. Could you repeat that? Sure thing.
```

Miss pattern:

- `without_repeat`: 11 / 20 target misses
- `with_repeat`: 2 / 20 separator misses
- `with_repeat`: 4 / 20 before-repeat target misses after separator splitting
- `with_repeat`: 2 / 20 after-repeat target misses after separator splitting

## Artifacts

The raw run outputs are intentionally left under gitignored `runs/` paths:

- `runs/in_context_asr_moss4b_transcription_requests.jsonl`
- `runs/moss4b_in_context_asr_transcription_raw.jsonl`
- `runs/moss4b_in_context_asr_transcription_summary.txt`

Superseded target-aware diagnostic artifacts:

- `runs/in_context_asr_moss4b_target_probe_requests.jsonl`
- `runs/moss4b_in_context_asr_target_probe_raw.jsonl`
- `runs/moss4b_in_context_asr_target_probe_summary.txt`
