# Devanagari error analysis — `lora_r16_attn_ffn`

Samples: 12869 — total reference aksharas: 42374 — total akshara-level edits: 5560

## Error category breakdown

| Category | Count | % of errors |
|---|---:|---:|
| Base consonant substitution | 2217 | 39.9% |
| Conjunct substitution | 1596 | 28.7% |
| Matra (dependent vowel) error | 605 | 10.9% |
| Deletion | 578 | 10.4% |
| Insertion | 282 | 5.1% |
| Nasalization/visarga sign error | 203 | 3.7% |
| Insertion (conjunct) | 45 | 0.8% |
| Deletion (conjunct) | 34 | 0.6% |

## Conjunct vs simple akshara error rate

| Akshara type | Occurrences | Errored | Error rate |
|---|---:|---:|---:|
| Conjunct (contains virama-joined consonants) | 4595 | 876 | 0.1906 |
| Simple | 37779 | 4357 | 0.1153 |

## Top akshara confusion pairs (reference → prediction)

| Reference | Prediction | Count |
|---|---|---:|
| म | स | 16 |
| ता | त | 13 |
| स | र | 13 |
| रा | र | 12 |
| क | का | 12 |
| स | रा | 11 |
| म | मा | 11 |
| अ | आ | 11 |
| श | रा | 10 |
| आ | अ | 9 |
| ब | बा | 9 |
| त | ल | 9 |
| ए | रा | 9 |
| ता | ना | 8 |
| स | सा | 8 |
| न | म | 8 |
| मि | सि | 8 |
| ग | गा | 7 |
| र | स | 7 |
| का | ना | 7 |
| सि | रि | 7 |
| हा | ह | 7 |
| बा | वा | 7 |
| रा | श | 7 |
| वि | नि | 7 |

## Akshara error rate by word length (aksharas)

| Length | Words | Mean AER |
|---|---:|---:|
| 1-2 | 3372 | 0.1555 |
| 3-4 | 7804 | 0.1168 |
| 5-6 | 1483 | 0.1438 |
| 7+ | 210 | 0.2085 |

## Akshara error rate by conjunct count

| Conjuncts in word | Words | Mean AER |
|---|---:|---:|
| 0 conjuncts | 8870 | 0.1180 |
| 1 conjunct | 3432 | 0.1529 |
| 2+ conjuncts | 567 | 0.2133 |
