# Devanagari error analysis — `lora_r8_attn`

Samples: 12869 — total reference aksharas: 42374 — total akshara-level edits: 5914

## Error category breakdown

| Category | Count | % of errors |
|---|---:|---:|
| Base consonant substitution | 2387 | 40.4% |
| Conjunct substitution | 1641 | 27.7% |
| Matra (dependent vowel) error | 643 | 10.9% |
| Deletion | 638 | 10.8% |
| Insertion | 309 | 5.2% |
| Nasalization/visarga sign error | 197 | 3.3% |
| Insertion (conjunct) | 50 | 0.8% |
| Deletion (conjunct) | 49 | 0.8% |

## Conjunct vs simple akshara error rate

| Akshara type | Occurrences | Errored | Error rate |
|---|---:|---:|---:|
| Conjunct (contains virama-joined consonants) | 4595 | 902 | 0.1963 |
| Simple | 37779 | 4653 | 0.1232 |

## Top akshara confusion pairs (reference → prediction)

| Reference | Prediction | Count |
|---|---|---:|
| म | स | 17 |
| स | र | 14 |
| रा | र | 13 |
| क | का | 13 |
| अ | आ | 13 |
| ता | त | 12 |
| स | रा | 11 |
| आ | अ | 10 |
| श | रा | 10 |
| मि | सि | 10 |
| स | सा | 9 |
| ग | गा | 9 |
| मा | ग्रा | 9 |
| मा | सा | 9 |
| त | ल | 9 |
| का | क | 9 |
| म | मा | 9 |
| त | न | 9 |
| ब | बा | 8 |
| ज | न | 8 |
| ए | रा | 8 |
| ता | ना | 7 |
| रू | रु | 7 |
| न | म | 7 |
| र | स | 7 |

## Akshara error rate by word length (aksharas)

| Length | Words | Mean AER |
|---|---:|---:|
| 1-2 | 3372 | 0.1738 |
| 3-4 | 7804 | 0.1221 |
| 5-6 | 1483 | 0.1557 |
| 7+ | 210 | 0.2195 |

## Akshara error rate by conjunct count

| Conjuncts in word | Words | Mean AER |
|---|---:|---:|
| 0 conjuncts | 8870 | 0.1282 |
| 1 conjunct | 3432 | 0.1626 |
| 2+ conjuncts | 567 | 0.2128 |
