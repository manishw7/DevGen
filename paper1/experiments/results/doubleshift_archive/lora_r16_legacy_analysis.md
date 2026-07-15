# Devanagari error analysis — `lora_r16_legacy`

Samples: 12869 — total reference aksharas: 42374 — total akshara-level edits: 55628

## Error category breakdown

| Category | Count | % of errors |
|---|---:|---:|
| Insertion | 26999 | 48.5% |
| Base consonant substitution | 18030 | 32.4% |
| Conjunct substitution | 4477 | 8.0% |
| Matra (dependent vowel) error | 4156 | 7.5% |
| Deletion | 1627 | 2.9% |
| Nasalization/visarga sign error | 221 | 0.4% |
| Deletion (conjunct) | 118 | 0.2% |

## Conjunct vs simple akshara error rate

| Akshara type | Occurrences | Errored | Error rate |
|---|---:|---:|---:|
| Conjunct (contains virama-joined consonants) | 4595 | 4595 | 1.0000 |
| Simple | 37779 | 24034 | 0.6362 |

## Top akshara confusion pairs (reference → prediction)

| Reference | Prediction | Count |
|---|---|---:|
| र | ा | 191 |
| ने | न | 162 |
| ता | त | 161 |
| ना | न | 142 |
| न | ा | 134 |
| ता | ा | 125 |
| या | ा | 108 |
| ना | ा | 104 |
| यों | ं | 104 |
| री | ी | 102 |
| ला | ल | 93 |
| का | क | 89 |
| रा | र | 89 |
| ने | े | 88 |
| री | र | 85 |
| ते | े | 85 |
| नी | ी | 78 |
| वा | ा | 76 |
| या | य | 75 |
| ती | ी | 74 |
| ते | त | 69 |
| का | ा | 67 |
| ल | ा | 67 |
| ला | ा | 63 |
| ती | त | 62 |

## Akshara error rate by word length (aksharas)

| Length | Words | Mean AER |
|---|---:|---:|
| 1-2 | 3372 | 1.4631 |
| 3-4 | 7804 | 1.2834 |
| 5-6 | 1483 | 1.2944 |
| 7+ | 210 | 1.2789 |

## Akshara error rate by conjunct count

| Conjuncts in word | Words | Mean AER |
|---|---:|---:|
| 0 conjuncts | 8870 | 1.1187 |
| 1 conjunct | 3432 | 1.7480 |
| 2+ conjuncts | 567 | 2.1433 |
