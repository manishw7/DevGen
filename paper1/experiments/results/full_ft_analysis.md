# Devanagari error analysis — `full_ft`

Samples: 12869 — total reference aksharas: 42374 — total akshara-level edits: 5152

## Error category breakdown

| Category | Count | % of errors |
|---|---:|---:|
| Base consonant substitution | 2015 | 39.1% |
| Conjunct substitution | 1608 | 31.2% |
| Deletion | 530 | 10.3% |
| Matra (dependent vowel) error | 512 | 9.9% |
| Insertion | 230 | 4.5% |
| Nasalization/visarga sign error | 174 | 3.4% |
| Insertion (conjunct) | 50 | 1.0% |
| Deletion (conjunct) | 33 | 0.6% |

## Conjunct vs simple akshara error rate

| Akshara type | Occurrences | Errored | Error rate |
|---|---:|---:|---:|
| Conjunct (contains virama-joined consonants) | 4595 | 781 | 0.1700 |
| Simple | 37779 | 4091 | 0.1083 |

## Top akshara confusion pairs (reference → prediction)

| Reference | Prediction | Count |
|---|---|---:|
| क | का | 12 |
| ता | ना | 11 |
| श | रा | 11 |
| ब | बा | 10 |
| आ | अ | 9 |
| मा | ग्रा | 9 |
| त | ल | 9 |
| ना | न | 9 |
| म | मा | 9 |
| स | र | 9 |
| का | क | 8 |
| म | स | 8 |
| त | न | 8 |
| स | रा | 8 |
| ती | नी | 8 |
| त | स्त | 7 |
| ब | बं | 7 |
| हा | ह | 7 |
| का | क्रा | 7 |
| सा | खा | 7 |
| ए | गु | 7 |
| मू | मु | 7 |
| ना | न्या | 6 |
| स | सा | 6 |
| ग | गा | 6 |

## Akshara error rate by word length (aksharas)

| Length | Words | Mean AER |
|---|---:|---:|
| 1-2 | 3372 | 0.1299 |
| 3-4 | 7804 | 0.1079 |
| 5-6 | 1483 | 0.1437 |
| 7+ | 210 | 0.1927 |

## Akshara error rate by conjunct count

| Conjuncts in word | Words | Mean AER |
|---|---:|---:|
| 0 conjuncts | 8870 | 0.1086 |
| 1 conjunct | 3432 | 0.1333 |
| 2+ conjuncts | 567 | 0.1995 |
