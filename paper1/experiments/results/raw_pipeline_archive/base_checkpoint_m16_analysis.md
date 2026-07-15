# Devanagari error analysis — `base_checkpoint_m16`

Samples: 12869 — total reference aksharas: 42374 — total akshara-level edits: 32047

## Error category breakdown

| Category | Count | % of errors |
|---|---:|---:|
| Deletion | 14993 | 46.8% |
| Base consonant substitution | 8427 | 26.3% |
| Conjunct substitution | 5578 | 17.4% |
| Deletion (conjunct) | 1837 | 5.7% |
| Matra (dependent vowel) error | 1086 | 3.4% |
| Nasalization/visarga sign error | 115 | 0.4% |
| Insertion | 11 | 0.0% |

## Conjunct vs simple akshara error rate

| Akshara type | Occurrences | Errored | Error rate |
|---|---:|---:|---:|
| Conjunct (contains virama-joined consonants) | 4595 | 4025 | 0.8760 |
| Simple | 37779 | 28011 | 0.7414 |

## Top akshara confusion pairs (reference → prediction)

| Reference | Prediction | Count |
|---|---|---:|
| ल | त | 74 |
| न | त | 72 |
| न | ग | 51 |
| ना | ग | 37 |
| ज | ग | 37 |
| ना | न | 37 |
| ता | त | 36 |
| म | ग | 32 |
| र | ग | 31 |
| र | रा | 23 |
| गा | ग | 23 |
| य | ग | 23 |
| त | ति | 22 |
| क | व | 22 |
| ने | गे | 21 |
| स | ग | 21 |
| र | ष | 20 |
| न | व | 18 |
| ते | गे | 17 |
| ल | न | 17 |
| ल | ग | 17 |
| र | श | 16 |
| ने | ते | 16 |
| र | त | 16 |
| क | द | 15 |

## Akshara error rate by word length (aksharas)

| Length | Words | Mean AER |
|---|---:|---:|
| 1-2 | 3372 | 0.4263 |
| 3-4 | 7804 | 0.7612 |
| 5-6 | 1483 | 0.9372 |
| 7+ | 210 | 0.9680 |

## Akshara error rate by conjunct count

| Conjuncts in word | Words | Mean AER |
|---|---:|---:|
| 0 conjuncts | 8870 | 0.6587 |
| 1 conjunct | 3432 | 0.7648 |
| 2+ conjuncts | 567 | 0.8891 |
