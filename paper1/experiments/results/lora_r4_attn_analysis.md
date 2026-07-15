# Devanagari error analysis — `lora_r4_attn`

Samples: 12869 — total reference aksharas: 42374 — total akshara-level edits: 6131

## Error category breakdown

| Category | Count | % of errors |
|---|---:|---:|
| Base consonant substitution | 2481 | 40.5% |
| Conjunct substitution | 1722 | 28.1% |
| Deletion | 677 | 11.0% |
| Matra (dependent vowel) error | 642 | 10.5% |
| Insertion | 307 | 5.0% |
| Nasalization/visarga sign error | 198 | 3.2% |
| Deletion (conjunct) | 54 | 0.9% |
| Insertion (conjunct) | 50 | 0.8% |

## Conjunct vs simple akshara error rate

| Akshara type | Occurrences | Errored | Error rate |
|---|---:|---:|---:|
| Conjunct (contains virama-joined consonants) | 4595 | 937 | 0.2039 |
| Simple | 37779 | 4837 | 0.1280 |

## Top akshara confusion pairs (reference → prediction)

| Reference | Prediction | Count |
|---|---|---:|
| म | स | 19 |
| श | रा | 13 |
| ता | त | 13 |
| अ | आ | 12 |
| स | रा | 11 |
| आ | अ | 11 |
| ग | गा | 11 |
| ज | न | 11 |
| त | ल | 11 |
| मा | सा | 11 |
| क | का | 11 |
| म | मा | 11 |
| रा | र | 10 |
| ता | ना | 9 |
| र | रू | 9 |
| मि | सि | 9 |
| स | र | 9 |
| स | सा | 8 |
| व | वा | 8 |
| ब | बा | 8 |
| न | म | 8 |
| मा | ग्रा | 8 |
| मा | म | 8 |
| न | त | 8 |
| हा | ह | 8 |

## Akshara error rate by word length (aksharas)

| Length | Words | Mean AER |
|---|---:|---:|
| 1-2 | 3372 | 0.1836 |
| 3-4 | 7804 | 0.1256 |
| 5-6 | 1483 | 0.1581 |
| 7+ | 210 | 0.2522 |

## Akshara error rate by conjunct count

| Conjuncts in word | Words | Mean AER |
|---|---:|---:|
| 0 conjuncts | 8870 | 0.1339 |
| 1 conjunct | 3432 | 0.1678 |
| 2+ conjuncts | 567 | 0.2165 |
