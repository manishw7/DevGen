# Devanagari error analysis — `lora_r8_attn`

Samples: 12869 — total reference aksharas: 42374 — total akshara-level edits: 53469

## Error category breakdown

| Category | Count | % of errors |
|---|---:|---:|
| Insertion | 24555 | 45.9% |
| Base consonant substitution | 18515 | 34.6% |
| Conjunct substitution | 4420 | 8.3% |
| Matra (dependent vowel) error | 3647 | 6.8% |
| Deletion | 2000 | 3.7% |
| Deletion (conjunct) | 169 | 0.3% |
| Nasalization/visarga sign error | 155 | 0.3% |
| Insertion (conjunct) | 8 | 0.0% |

## Conjunct vs simple akshara error rate

| Akshara type | Occurrences | Errored | Error rate |
|---|---:|---:|---:|
| Conjunct (contains virama-joined consonants) | 4595 | 4588 | 0.9985 |
| Simple | 37779 | 24318 | 0.6437 |

## Top akshara confusion pairs (reference → prediction)

| Reference | Prediction | Count |
|---|---|---:|
| र | ा | 231 |
| ता | ा | 171 |
| न | ा | 166 |
| ने | न | 163 |
| ना | न | 140 |
| री | ी | 124 |
| ता | त | 118 |
| र | ् | 115 |
| ना | ा | 110 |
| या | ा | 105 |
| वा | ा | 100 |
| ने | े | 89 |
| का | क | 88 |
| ते | े | 88 |
| ला | ा | 87 |
| ती | ी | 87 |
| क | ् | 86 |
| का | ा | 84 |
| स | ा | 84 |
| यों | ं | 83 |
| ल | ा | 82 |
| क | ि | 81 |
| ला | ल | 79 |
| त | ि | 79 |
| नी | ी | 79 |

## Akshara error rate by word length (aksharas)

| Length | Words | Mean AER |
|---|---:|---:|
| 1-2 | 3372 | 1.4837 |
| 3-4 | 7804 | 1.2276 |
| 5-6 | 1483 | 1.2098 |
| 7+ | 210 | 1.1960 |

## Akshara error rate by conjunct count

| Conjuncts in word | Words | Mean AER |
|---|---:|---:|
| 0 conjuncts | 8870 | 1.0985 |
| 1 conjunct | 3432 | 1.6874 |
| 2+ conjuncts | 567 | 1.9291 |
