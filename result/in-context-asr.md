# In-Context ASR Results

This note records the completed in-context-asr probe for
`OpenMOSS-Team/MOSS-Audio-4B-Instruct`.

## Setup

- Dataset: `robflynnyh/in-context-asr`
- Local checkout: `/exp/exp4/acp21rjf/in-context-asr`
- Items: 20
- Requests: 40, one `sentence_without_repeat.wav` and one
  `sentence_with_repeat.wav` request per item
- Model: `OpenMOSS-Team/MOSS-Audio-4B-Instruct`
- Local model path: `/store/store5/acp21rjf/models/MOSS-Audio-4B-Instruct`
- Generation: deterministic, `temperature=0`, `max_new_tokens=192`
- Prompt mode: target-word JSON probe with transcript fields

The comparable metric follows the original repo's scoring style: normalize the
model transcript text and search for the target word or accepted variant. The
boolean-field metric is also retained because the prompt explicitly requested
target-presence booleans.

Without-repeat prompt:

```text
You are evaluating speech recognition on a short audio clip. Transcribe only what is clearly audible; do not infer missing words from context.

Target word or accepted variant: {target}

Return exactly one JSON object with this schema:
{"target_present": true, "transcript": "..."}

Set target_present to true only if the target word is clearly audible and recognizable.
```

With-repeat prompt:

```text
You are evaluating speech recognition on an audio clip that may contain an initial unclear sentence, a spoken request to repeat it, and then a clearer repeated sentence. Transcribe only what is clearly audible; do not infer missing words from context.

Target word or accepted variant: {target}
Possible repeat-request phrases: {separators}

Return exactly one JSON object with this schema:
{"target_before_repeat": false, "target_after_repeat": true, "before_repeat_transcript": "...", "after_repeat_transcript": "..."}

Set each boolean to true only if the target word is clearly audible and recognizable in that segment.
```

## Results

| Metric source | Parsed | Correct in clear repeat | Correct in corrupt repeat | Correct without repeat |
| --- | ---: | ---: | ---: | ---: |
| Transcript search | 40 / 40 | 100.000 | 75.000 | 100.000 |
| Boolean fields | 40 / 40 | 100.000 | 0.000 | 100.000 |

## Notes

The two scoring views disagree on corrupt-repeat recognition. MOSS returned
`target_before_repeat: false` for all 20 with-repeat requests, but in 15 of
those requests its own `before_repeat_transcript` contained the target. For
comparison with the original in-context-asr runner, use the transcript-search
row as the headline result.

Example raw responses:

```text
in_context_asr__row-001__without_repeat:
{"target_present": true, "transcript": "The professor's pedagogical approach"}

in_context_asr__row-001__with_repeat:
{"target_before_repeat": false, "target_after_repeat": true, "before_repeat_transcript": "The professor's pedagogical approach emphasized critical thinking over rote memorization.", "after_repeat_transcript": "The professor's pedagogical approach emphasized critical thinking over rote memorization."}

in_context_asr__row-002__with_repeat:
{"target_before_repeat": false, "target_after_repeat": true, "before_repeat_transcript": "His proclivity for collecting...", "after_repeat_transcript": "His proclivity for collecting antique watches turned into a lucrative business venture."}
```

## Artifacts

The raw run outputs are intentionally left under gitignored `runs/` paths:

- `runs/in_context_asr_moss4b_target_probe_requests.jsonl`
- `runs/moss4b_in_context_asr_target_probe_raw.jsonl`
- `runs/moss4b_in_context_asr_target_probe_summary.txt`
