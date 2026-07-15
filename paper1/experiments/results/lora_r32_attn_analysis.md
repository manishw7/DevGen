# Devanagari error analysis — `lora_r32_attn`

Samples: 12869 — total reference aksharas: 42374 — total akshara-level edits: 5596

## Error category breakdown

| Category | Count | % of errors |
|---|---:|---:|
| Base consonant substitution | 2251 | 40.2% |
| Conjunct substitution | 1576 | 28.2% |
| Matra (dependent vowel) error | 622 | 11.1% |
| Deletion | 569 | 10.2% |
| Insertion | 292 | 5.2% |
| Nasalization/visarga sign error | 198 | 3.5% |
| Insertion (conjunct) | 49 | 0.9% |
| Deletion (conjunct) | 39 | 0.7% |

## Conjunct vs simple akshara error rate

| Akshara type | Occurrences | Errored | Error rate |
|---|---:|---:|---:|
| Conjunct (contains virama-joined consonants) | 4595 | 852 | 0.1854 |
| Simple | 37779 | 4403 | 0.1165 |

## Top akshara confusion pairs (reference → prediction)

| Reference | Prediction | Count |
|---|---|---:|
| ता | त | 14 |
| म | स | 13 |
| अ | आ | 13 |
| श | रा | 11 |
| रा | र | 11 |
| स | र | 11 |
| का | क | 10 |
| आ | अ | 9 |
| न | म | 9 |
| मा | सा | 9 |
| त | ल | 9 |
| म | मा | 9 |
| ए | रा | 9 |
| स | रा | 9 |
| ता | ना | 8 |
| स | सा | 8 |
| ब | बा | 8 |
| ज | न | 8 |
| मा | ग्रा | 8 |
| मि | सि | 8 |
| ग | गा | 8 |
| रू | रु | 7 |
| हा | ह | 7 |
| शा | श | 7 |
| क | का | 7 |

## Akshara error rate by word length (aksharas)

| Length | Words | Mean AER |
|---|---:|---:|
| 1-2 | 3372 | 0.1594 |
| 3-4 | 7804 | 0.1167 |
| 5-6 | 1483 | 0.1434 |
| 7+ | 210 | 0.2165 |

## Akshara error rate by conjunct count

| Conjuncts in word | Words | Mean AER |
|---|---:|---:|
| 0 conjuncts | 8870 | 0.1201 |
| 1 conjunct | 3432 | 0.1535 |
| 2+ conjuncts | 567 | 0.2019 |
