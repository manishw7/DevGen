# Devanagari error analysis — `lora_r16_attn`

Samples: 12869 — total reference aksharas: 42374 — total akshara-level edits: 5761

## Error category breakdown

| Category | Count | % of errors |
|---|---:|---:|
| Base consonant substitution | 2319 | 40.3% |
| Conjunct substitution | 1635 | 28.4% |
| Matra (dependent vowel) error | 614 | 10.7% |
| Deletion | 598 | 10.4% |
| Insertion | 300 | 5.2% |
| Nasalization/visarga sign error | 199 | 3.5% |
| Insertion (conjunct) | 49 | 0.9% |
| Deletion (conjunct) | 47 | 0.8% |

## Conjunct vs simple akshara error rate

| Akshara type | Occurrences | Errored | Error rate |
|---|---:|---:|---:|
| Conjunct (contains virama-joined consonants) | 4595 | 895 | 0.1948 |
| Simple | 37779 | 4517 | 0.1196 |

## Top akshara confusion pairs (reference → prediction)

| Reference | Prediction | Count |
|---|---|---:|
| म | स | 16 |
| अ | आ | 15 |
| ता | त | 14 |
| रा | र | 13 |
| स | रा | 12 |
| श | रा | 11 |
| न | म | 11 |
| स | र | 11 |
| ब | बा | 10 |
| त | ल | 10 |
| मि | सि | 10 |
| क | का | 10 |
| का | क | 10 |
| ए | रा | 10 |
| त | न | 10 |
| स | सा | 9 |
| आ | अ | 9 |
| मा | ग्रा | 9 |
| मा | सा | 9 |
| म | मा | 9 |
| ना | न | 8 |
| ग | गा | 8 |
| ता | ना | 7 |
| रू | रु | 7 |
| न | ल | 7 |

## Akshara error rate by word length (aksharas)

| Length | Words | Mean AER |
|---|---:|---:|
| 1-2 | 3372 | 0.1655 |
| 3-4 | 7804 | 0.1192 |
| 5-6 | 1483 | 0.1541 |
| 7+ | 210 | 0.2099 |

## Akshara error rate by conjunct count

| Conjuncts in word | Words | Mean AER |
|---|---:|---:|
| 0 conjuncts | 8870 | 0.1230 |
| 1 conjunct | 3432 | 0.1598 |
| 2+ conjuncts | 567 | 0.2133 |
