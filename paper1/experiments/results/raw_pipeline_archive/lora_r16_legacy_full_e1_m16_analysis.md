# Devanagari error analysis — `lora_r16_legacy_full_e1_m16`

Samples: 12869 — total reference aksharas: 42374 — total akshara-level edits: 55687

## Error category breakdown

| Category | Count | % of errors |
|---|---:|---:|
| Insertion | 26892 | 48.3% |
| Base consonant substitution | 18115 | 32.5% |
| Matra (dependent vowel) error | 4559 | 8.2% |
| Conjunct substitution | 4506 | 8.1% |
| Deletion | 1278 | 2.3% |
| Nasalization/visarga sign error | 247 | 0.4% |
| Deletion (conjunct) | 90 | 0.2% |

## Conjunct vs simple akshara error rate

| Akshara type | Occurrences | Errored | Error rate |
|---|---:|---:|---:|
| Conjunct (contains virama-joined consonants) | 4595 | 4595 | 1.0000 |
| Simple | 37779 | 24200 | 0.6406 |

## Top akshara confusion pairs (reference → prediction)

| Reference | Prediction | Count |
|---|---|---:|
| र | ा | 177 |
| ने | न | 166 |
| ता | त | 141 |
| ता | ा | 137 |
| ना | न | 132 |
| न | ा | 108 |
| का | क | 104 |
| रा | र | 99 |
| यों | ं | 97 |
| ना | ा | 96 |
| या | ा | 90 |
| ते | े | 90 |
| री | ी | 88 |
| ला | ल | 87 |
| री | र | 87 |
| या | य | 82 |
| ने | े | 76 |
| ती | ी | 73 |
| वा | ा | 72 |
| नी | न | 68 |
| नी | ी | 64 |
| ल | ा | 63 |
| वा | व | 62 |
| ती | त | 62 |
| ले | ल | 60 |

## Akshara error rate by word length (aksharas)

| Length | Words | Mean AER |
|---|---:|---:|
| 1-2 | 3372 | 1.5316 |
| 3-4 | 7804 | 1.2975 |
| 5-6 | 1483 | 1.2368 |
| 7+ | 210 | 1.1117 |

## Akshara error rate by conjunct count

| Conjuncts in word | Words | Mean AER |
|---|---:|---:|
| 0 conjuncts | 8870 | 1.1458 |
| 1 conjunct | 3432 | 1.7608 |
| 2+ conjuncts | 567 | 2.0309 |
