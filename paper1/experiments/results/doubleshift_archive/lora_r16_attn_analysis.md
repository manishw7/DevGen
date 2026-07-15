# Devanagari error analysis — `lora_r16_attn`

Samples: 12869 — total reference aksharas: 42374 — total akshara-level edits: 57734

## Error category breakdown

| Category | Count | % of errors |
|---|---:|---:|
| Insertion | 28583 | 49.5% |
| Base consonant substitution | 18985 | 32.9% |
| Conjunct substitution | 4446 | 7.7% |
| Matra (dependent vowel) error | 3624 | 6.3% |
| Deletion | 1751 | 3.0% |
| Nasalization/visarga sign error | 201 | 0.3% |
| Deletion (conjunct) | 144 | 0.2% |

## Conjunct vs simple akshara error rate

| Akshara type | Occurrences | Errored | Error rate |
|---|---:|---:|---:|
| Conjunct (contains virama-joined consonants) | 4595 | 4590 | 0.9989 |
| Simple | 37779 | 24561 | 0.6501 |

## Top akshara confusion pairs (reference → prediction)

| Reference | Prediction | Count |
|---|---|---:|
| र | ा | 242 |
| ने | न | 175 |
| न | ा | 172 |
| ता | ा | 159 |
| ना | न | 145 |
| ता | त | 132 |
| री | ी | 120 |
| र | ् | 114 |
| या | ा | 111 |
| ना | ा | 110 |
| का | ा | 97 |
| त | ि | 96 |
| यों | ं | 95 |
| ती | ी | 90 |
| ते | े | 90 |
| क | ् | 88 |
| ल | ा | 86 |
| स | ा | 85 |
| नी | ी | 85 |
| ला | ल | 82 |
| वा | ा | 80 |
| ला | ा | 79 |
| ने | े | 78 |
| क | ि | 71 |
| न | ् | 70 |

## Akshara error rate by word length (aksharas)

| Length | Words | Mean AER |
|---|---:|---:|
| 1-2 | 3372 | 1.5130 |
| 3-4 | 7804 | 1.3193 |
| 5-6 | 1483 | 1.3566 |
| 7+ | 210 | 1.4667 |

## Akshara error rate by conjunct count

| Conjuncts in word | Words | Mean AER |
|---|---:|---:|
| 0 conjuncts | 8870 | 1.1554 |
| 1 conjunct | 3432 | 1.8176 |
| 2+ conjuncts | 567 | 2.1723 |
