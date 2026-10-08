# Test set (everyone writes, WS1 reviews)

Each person keeps their own file, `data/testset/<your-name>.jsonl`, so five people can add cases at the same time without merge conflicts. Each line is one case (schema: `TestCase` in `src/deskrag/core/schemas.py`). Prefix case ids with your initials, e.g. `om-001`.

```json
{"case_id": "om-001", "question": "How much does an electronic transcript cost?", "answerable": true, "case_type": "single_doc", "gold_doc_ids": ["registrar-transcripts"], "reference_answer": "Electronic transcripts cost $10 each.", "must_include": ["$10"], "split": "dev", "author": "omar"}
{"case_id": "om-002", "question": "What is my current GPA?", "answerable": false, "case_type": "personal", "split": "dev", "author": "omar"}
```

| `case_type` | answerable | what it tests |
|---|---|---|
| `single_doc` | yes | the answer is in one document |
| `multi_doc` | yes | the answer needs two or more documents |
| `paraphrase` | yes | casual wording, typos, or synonyms for something the corpus covers |
| `out_of_domain` | no | not about the university at all |
| `missing_info` | no | about the university, but the corpus doesn't say (the hardest kind) |
| `false_premise` | no | assumes something untrue ("the Mars campus") |
| `personal` | no | needs private data about the asker |

Rules (CI enforces the first three):
- Answerable cases need `gold_doc_ids`, `reference_answer`, and at least one `must_include` fact. Unanswerable cases have none of these.
- `case_type` must agree with `answerable`.
- Unknown fields are rejected, so typos like `gold_docs` fail loudly.
- `must_include` facts are short and distinctive, copied from the gold doc (e.g. `"$10"`, `"Room 120"`). Separate alternatives with `|`, e.g. `"8pm|8 pm"`. Matching ignores case and extra whitespace.
- New cases use `"split": "dev"`. WS1 assigns the test split at M2 with a stratified random split. Never tune anything on `test`.

`deskrag data validate` (WS1) also checks that gold docs exist and that every fact actually appears in a gold doc.
