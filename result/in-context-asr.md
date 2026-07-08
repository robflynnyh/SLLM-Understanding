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
`sentence_without_repeat.wav`; the samples below show this pattern.

Miss pattern:

- `without_repeat`: 11 / 20 target misses
- `with_repeat`: 2 / 20 separator misses
- `with_repeat`: 4 / 20 before-repeat target misses after separator splitting
- `with_repeat`: 2 / 20 after-repeat target misses after separator splitting

## Output Samples

The full raw JSONL is in `runs/moss4b_in_context_asr_transcription_raw.jsonl`.
These are the paired model transcript outputs for each item.

### Row 1: target `pedagogical`

`without_repeat`:

```text
The professor's pedagogical approach mythologized critical thinking over rote memorization. I think the line broke up. Could you repeat that? Sure thing.
```

`with_repeat`:

```text
The professor's pedagogical approach emphasized critical thinking over rote memorization. I think the line broke up. Could you repeat that? Sure thing. The professor's pedagogical approach emphasized critical thinking over rote memorization.
```

### Row 2: target `antique`

`without_repeat`:

```text
His proclivity for collecting to a lucrative business venture. I think the line broke up. Could you repeat that? Sure thing.
```

`with_repeat`:

```text
His proclivity for collecting antique watches turned into a lucrative business venture. I think the line broke up. Could you repeat that? Sure thing. His proclivity for collecting antique watches turned into a lucrative business venture.
```

### Row 3: target `quintessential`

`without_repeat`:

```text
The experience includes watching leaves change color while sipping hot cider. I think the line broke up. Could you repeat that? Sure thing.
```

`with_repeat`:

```text
The quintessential autumn experience includes watching leaves change color while sipping hot cider. I think the line broke up. Could you repeat that? Sure thing. The quintessential autumn experience includes watching leaves change color while sipping hot cider.
```

### Row 4: target `renaissance`

`without_repeat`:

```text
The confluence of art and technology created an unprecedented cultural phenomenon. I think the line broke up. Could you repeat that? Sure thing.
```

`with_repeat`:

```text
The confluence of art and technology created an unprecedented cultural renaissance. I think the line broke up. Could you repeat that? Sure thing. The confluence of art and technology created an unprecedented cultural renaissance.
```

### Row 5: target `finally`

`without_repeat`:

```text
His propensity for procrastination finally caught up with him during finals week. Sorry, I didn't catch that. Could you say that again? Sure thing.
```

`with_repeat`:

```text
His propensity for procrastination finally caught up with him during finals week. Sorry, I didn't catch that. Could you say that again? Sure thing. His propensity for procrastination finally caught up with him during finals week.
```

### Row 6: target `respect`

`without_repeat`:

```text
Her pragmatic approach to problem solving earned her respect among colleagues. Sorry, I didn't catch that. Could you say that again? Sure thing.
```

`with_repeat`:

```text
Her pragmatic approach to problem solving earned her respect among colleagues. Sorry, I didn't catch that. Could you say that again? Sure thing. Her pragmatic approach to problem solving earned her respect among colleagues.
```

### Row 7: target `vernacular`

`without_repeat`:

```text
The vernacular architecture of the region reflected centuries of local tradition. Sorry, I didn't catch that. Could you say that again? Sure thing.
```

`with_repeat`:

```text
The vernacular architecture of the region reflected centuries of local tradition. Sorry, I didn't catch that. Could you say that again? Sure thing. The vernacular architecture of the region reflected centuries of local tradition.
```

### Row 8: target `therapist`

`without_repeat`:

```text
Her perspicacious observations about human nature made her an excellent therapist. Sorry, I didn't catch that. Could you say that again? Sure thing.
```

`with_repeat`:

```text
Her perspicacious observations about human nature made her an excellent therapist. Sorry, I didn't catch that. Could you say that again? Sure thing. Her perspicacious observations about human nature made her an excellent therapist.
```

### Row 9: target `flamboyant`

`without_repeat`:

```text
The Wambiant decorations transformed the ordinary room into a spectacular space. Sorry, I didn't catch that. Could you say that again? Sure thing.
```

`with_repeat`:

```text
The flamboyant decorations transformed the ordinary room into a spectacular space. Sorry, I didn't catch that. Could you say that again? Sure thing. The flamboyant decorations transformed the ordinary room into a spectacular space.
```

### Row 10: target `frost`

`without_repeat`:

```text
The winter mornings were harsh, especially before sunrise when the frost was thickest. Sorry, I didn't catch that. Could you say that again? Sure thing.
```

`with_repeat`:

```text
The winter mornings were harsh, especially before sunrise when the frost was thickest. Sorry, I didn't catch that. Could you say that again? Sure thing. The winter mornings were harsh, especially before sunrise when the frost was thickest.
```

### Row 11: target `computation`

`without_repeat`:

```text
The biggest lesson that can be read from seventy years of AI research is that general methods that leverage computation are ultimately the most effective and by a large margin. Sorry, I didn't catch that. Could you say that again? Sure thing.
```

`with_repeat`:

```text
The biggest lesson that can be read from seventy years of AI research is that general methods that leverage computation are ultimately the most effective and by a large margin. Sorry, I didn't catch that. Could you say that again? Sure thing. The biggest lesson that can be read from seventy years of AI research is that general methods that leverage computation are ultimately the most effective and by a large margin.
```

### Row 12: target `general purpose`

`without_repeat`:

```text
One thing that I should mention is that I think there's something wrong with the line. Could you repeat that? Sure thing.
```

`with_repeat`:

```text
One thing that should be learned from the bitter lesson is the great power of general purpose methods.
```

### Row 13: target `guitar`

`without_repeat`:

```text
I actually just started taking guitar lessons last month. It's quite harder than I expected. Sorry, the line is a bit funny. Could you repeat that? Sure thing.
```

`with_repeat`:

```text
I actually just started taking guitar lessons last month. It's quite harder than I expected. Sorry, the line is a bit funny. Could you repeat that? Sure thing. I was just mentioning how I started taking guitar lessons last month.
```

### Row 14: target `main street`

`without_repeat`:

```text
Did you hear they're finally renovating the old theater on Main Street? Sorry, the line is a bit funny. Could you repeat that? Sure thing.
```

`with_repeat`:

```text
Did you hear they're finally renovating the old theater on Main Street? Sorry, the line is a bit funny. Could you repeat that? Sure thing. Did you hear they're finally renovating the old theater on Main Street?
```

### Row 15: target `sourdough`

`without_repeat`:

```text
You didn't see me getting a hand to the face. You didn't see me getting a hand to the face. Sorry, the line is a bit funny. Could you repeat that? Sure thing.
```

`with_repeat`:

```text
Sorry, the line is a bit funny. Could you repeat that? Sure thing. I think I'm finally getting the hang of sourdough baking. After like five failed attempts.
```

### Row 16: target `innate`

`without_repeat`:

```text
Is enabling to new situations, makes them actually. Sorry, the line is a bit funny. Could you repeat that? Sure thing.
```

`with_repeat`:

```text
His inability to diffuse tense situations makes him an excellent mediator. Sorry, the line is a bit funny. Could you repeat that? Sure thing. His innate ability to diffuse tense situations makes him an excellent mediator.
```

### Row 17: target `eloquent`

`without_repeat`:

```text
So I can see that the bad location for the name moves everything here. Sorry, the line is a bit funny. Could you repeat that? Sure thing.
```

`with_repeat`:

```text
Her eloquent speech at the graduation ceremony moved everyone to tears. Sorry, the line is a bit funny. Could you repeat that? Sure thing. Her eloquent speech at the graduation ceremony moved everyone to tears.
```

### Row 18: target `landmark`

`without_repeat`:

```text
This is the art exhibition. It is the only way to see it. Oh dear, the line is quite noisy. Could you repeat that last sentence? Sure thing.
```

`with_repeat`:

```text
The distinctive architecture of the building makes it a prominent landmark downtown.
```

### Row 19: target `profound`

`without_repeat`:

```text
Have you shared this? Have an ounce of the lips until the final days. I don't know why I'm in a handish. Hey, I can't hear much. Can you turn that music down and repeat what you said?
```

`with_repeat`:

```text
Evolutionary processes have endowed us with a glimpse into the profound mechanisms underlying learning and adaptation. Hey, I can't hear much. Can you turn that music down and repeat what you said? Sure, no worries. I said that. Evolutionary processes have endowed us with a glimpse into the profound mechanisms underlying learning and adaptation.
```

### Row 20: target `ships`

`without_repeat`:

```text
They lifted people off the ship. I don't know where they're going. Hey, I didn't get that last bit. Can you repeat it, John?
```

`with_repeat`:

```text
They lifted people up into the ships. I don't know where they're going. Hey, I didn't get that last bit. Can you repeat it, John? I said, they lifted people up into the ships with some kind of beam. I don't know where they're going.
```

## Artifacts

The raw run outputs are intentionally left under gitignored `runs/` paths:

- `runs/in_context_asr_moss4b_transcription_requests.jsonl`
- `runs/moss4b_in_context_asr_transcription_raw.jsonl`
- `runs/moss4b_in_context_asr_transcription_summary.txt`

Superseded target-aware diagnostic artifacts:

- `runs/in_context_asr_moss4b_target_probe_requests.jsonl`
- `runs/moss4b_in_context_asr_target_probe_raw.jsonl`
- `runs/moss4b_in_context_asr_target_probe_summary.txt`
