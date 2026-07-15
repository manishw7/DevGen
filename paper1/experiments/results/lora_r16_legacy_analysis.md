# Devanagari error analysis — `lora_r16_legacy`

Samples: 12869 — total reference aksharas: 42374 — total akshara-level edits: 5531

## Error category breakdown

| Category | Count | % of errors |
|---|---:|---:|
| Base consonant substitution | 2220 | 40.1% |
| Conjunct substitution | 1580 | 28.6% |
| Matra (dependent vowel) error | 586 | 10.6% |
| Deletion | 583 | 10.5% |
| Insertion | 280 | 5.1% |
| Nasalization/visarga sign error | 197 | 3.6% |
| Insertion (conjunct) | 45 | 0.8% |
| Deletion (conjunct) | 40 | 0.7% |

## Conjunct vs simple akshara error rate

| Akshara type | Occurrences | Errored | Error rate |
|---|---:|---:|---:|
| Conjunct (contains virama-joined consonants) | 4595 | 875 | 0.1904 |
| Simple | 37779 | 4331 | 0.1146 |

## Top akshara confusion pairs (reference → prediction)

| Reference | Prediction | Count |
|---|---|---:|
| म | स | 17 |
| ता | त | 12 |
| रा | र | 12 |
| स | र | 11 |
| श | रा | 11 |
| अ | आ | 11 |
| क | का | 11 |
| म | मा | 11 |
| स | रा | 10 |
| आ | अ | 9 |
| ब | बा | 9 |
| ए | रा | 9 |
| ता | ना | 8 |
| र | स | 8 |
| मा | सा | 8 |
| स | सा | 7 |
| न | म | 7 |
| सि | रि | 7 |
| त | ल | 7 |
| मि | सि | 7 |
| हा | ह | 7 |
| का | क | 7 |
| बा | वा | 7 |
| वि | नि | 7 |
| ए | गु | 7 |

## Akshara error rate by word length (aksharas)

| Length | Words | Mean AER |
|---|---:|---:|
| 1-2 | 3372 | 0.1550 |
| 3-4 | 7804 | 0.1157 |
| 5-6 | 1483 | 0.1440 |
| 7+ | 210 | 0.2064 |

## Akshara error rate by conjunct count

| Conjuncts in word | Words | Mean AER |
|---|---:|---:|
| 0 conjuncts | 8870 | 0.1172 |
| 1 conjunct | 3432 | 0.1524 |
| 2+ conjuncts | 567 | 0.2100 |
