# Devanagari error analysis — `lora_r16_attn_full_e1_m16`

Samples: 12869 — total reference aksharas: 42374 — total akshara-level edits: 55598

## Error category breakdown

| Category | Count | % of errors |
|---|---:|---:|
| Insertion | 26588 | 47.8% |
| Base consonant substitution | 18689 | 33.6% |
| Conjunct substitution | 4498 | 8.1% |
| Matra (dependent vowel) error | 4470 | 8.0% |
| Deletion | 1050 | 1.9% |
| Nasalization/visarga sign error | 200 | 0.4% |
| Deletion (conjunct) | 83 | 0.1% |
| Insertion (conjunct) | 20 | 0.0% |

## Conjunct vs simple akshara error rate

| Akshara type | Occurrences | Errored | Error rate |
|---|---:|---:|---:|
| Conjunct (contains virama-joined consonants) | 4595 | 4579 | 0.9965 |
| Simple | 37779 | 24411 | 0.6462 |

## Top akshara confusion pairs (reference → prediction)

| Reference | Prediction | Count |
|---|---|---:|
| र | ा | 230 |
| ने | न | 199 |
| ना | न | 167 |
| ता | त | 149 |
| ता | ा | 136 |
| न | ा | 118 |
| या | य | 101 |
| री | ी | 97 |
| ला | ल | 97 |
| री | र | 91 |
| रा | र | 87 |
| ती | ी | 86 |
| र | ् | 86 |
| या | ा | 83 |
| का | क | 82 |
| ते | त | 79 |
| स | ा | 79 |
| ना | ा | 76 |
| ते | े | 76 |
| नी | न | 74 |
| ले | ल | 72 |
| का | ा | 70 |
| यों | य | 70 |
| क | ा | 68 |
| यों | ं | 67 |

## Akshara error rate by word length (aksharas)

| Length | Words | Mean AER |
|---|---:|---:|
| 1-2 | 3372 | 1.5446 |
| 3-4 | 7804 | 1.2954 |
| 5-6 | 1483 | 1.2281 |
| 7+ | 210 | 1.0767 |

## Akshara error rate by conjunct count

| Conjuncts in word | Words | Mean AER |
|---|---:|---:|
| 0 conjuncts | 8870 | 1.1635 |
| 1 conjunct | 3432 | 1.7263 |
| 2+ conjuncts | 567 | 1.9759 |
