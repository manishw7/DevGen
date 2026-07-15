# Devanagari error analysis — `base_prep`

Samples: 12869 — total reference aksharas: 42374 — total akshara-level edits: 8385

## Error category breakdown

| Category | Count | % of errors |
|---|---:|---:|
| Base consonant substitution | 3203 | 38.2% |
| Conjunct substitution | 2207 | 26.3% |
| Deletion | 1337 | 15.9% |
| Matra (dependent vowel) error | 792 | 9.4% |
| Insertion | 462 | 5.5% |
| Nasalization/visarga sign error | 158 | 1.9% |
| Deletion (conjunct) | 115 | 1.4% |
| Insertion (conjunct) | 111 | 1.3% |

## Conjunct vs simple akshara error rate

| Akshara type | Occurrences | Errored | Error rate |
|---|---:|---:|---:|
| Conjunct (contains virama-joined consonants) | 4595 | 1269 | 0.2762 |
| Simple | 37779 | 6543 | 0.1732 |

## Top akshara confusion pairs (reference → prediction)

| Reference | Prediction | Count |
|---|---|---:|
| म | स | 22 |
| ल | त | 21 |
| ता | त | 18 |
| न | त | 17 |
| श | रा | 16 |
| स | र | 16 |
| का | वा | 15 |
| म | ग | 15 |
| ना | न | 14 |
| गा | ग | 12 |
| आ | अ | 11 |
| रा | र | 11 |
| मा | सा | 11 |
| बा | वा | 11 |
| ला | ता | 11 |
| रू | रु | 10 |
| अ | आ | 10 |
| अ | स | 10 |
| स | म | 10 |
| ल | न | 10 |
| र | रा | 10 |
| क | का | 9 |
| क | व | 9 |
| शा | श | 9 |
| गा | वा | 9 |

## Akshara error rate by word length (aksharas)

| Length | Words | Mean AER |
|---|---:|---:|
| 1-2 | 3372 | 0.3729 |
| 3-4 | 7804 | 0.1681 |
| 5-6 | 1483 | 0.1552 |
| 7+ | 210 | 0.2314 |

## Akshara error rate by conjunct count

| Conjuncts in word | Words | Mean AER |
|---|---:|---:|
| 0 conjuncts | 8870 | 0.2101 |
| 1 conjunct | 3432 | 0.2514 |
| 2+ conjuncts | 567 | 0.2151 |
