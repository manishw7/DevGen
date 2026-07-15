# Devanagari error analysis — `lora_r4_attn`

Samples: 12869 — total reference aksharas: 42374 — total akshara-level edits: 58535

## Error category breakdown

| Category | Count | % of errors |
|---|---:|---:|
| Insertion | 29291 | 50.0% |
| Base consonant substitution | 19053 | 32.5% |
| Conjunct substitution | 4473 | 7.6% |
| Matra (dependent vowel) error | 3551 | 6.1% |
| Deletion | 1854 | 3.2% |
| Nasalization/visarga sign error | 190 | 0.3% |
| Deletion (conjunct) | 122 | 0.2% |
| Insertion (conjunct) | 1 | 0.0% |

## Conjunct vs simple akshara error rate

| Akshara type | Occurrences | Errored | Error rate |
|---|---:|---:|---:|
| Conjunct (contains virama-joined consonants) | 4595 | 4595 | 1.0000 |
| Simple | 37779 | 24648 | 0.6524 |

## Top akshara confusion pairs (reference → prediction)

| Reference | Prediction | Count |
|---|---|---:|
| र | ा | 219 |
| ना | न | 167 |
| न | ा | 157 |
| ने | न | 154 |
| ता | ा | 145 |
| ता | त | 123 |
| या | ा | 114 |
| री | ी | 109 |
| र | ् | 108 |
| ने | े | 97 |
| यों | ं | 96 |
| ना | ा | 95 |
| त | ि | 95 |
| ते | े | 91 |
| क | ् | 90 |
| नी | ी | 90 |
| ला | ल | 88 |
| का | क | 86 |
| ती | ी | 83 |
| ल | ् | 83 |
| क | ि | 83 |
| मा | ा | 78 |
| न | ् | 76 |
| ला | ा | 74 |
| वा | ा | 72 |

## Akshara error rate by word length (aksharas)

| Length | Words | Mean AER |
|---|---:|---:|
| 1-2 | 3372 | 1.6176 |
| 3-4 | 7804 | 1.3261 |
| 5-6 | 1483 | 1.3511 |
| 7+ | 210 | 1.4890 |

## Akshara error rate by conjunct count

| Conjuncts in word | Words | Mean AER |
|---|---:|---:|
| 0 conjuncts | 8870 | 1.1935 |
| 1 conjunct | 3432 | 1.8308 |
| 2+ conjuncts | 567 | 2.2047 |
