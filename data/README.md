# Data — University of Toledo RAG Evaluation

## 1. Overview

This directory contains the document corpus and labeled evaluation dataset for our **Domain-Specific RAG Evaluation** project.

Our system is designed to answer university-related questions using information retrieved from official University of Toledo webpages.

The data supports three main evaluation objectives:

1. **Retrieval Quality:** Can the system retrieve the correct documents for a question?
2. **Answer Quality:** Can the language model generate accurate, evidence-supported answers?
3. **Abstention Quality:** Can the system correctly decline to answer questions when sufficient information is unavailable?

We use a frozen document corpus to ensure that different retrieval methods, prompts, and language models are evaluated against the same information.

## 2. Directory Structure

```text
data/
├── README.md
├── corpus/
│   ├── sources.csv
│   ├── MANIFEST
│   ├── docs/
│   │   ├── it001.md
│   │   ├── it002.md
│   │   ├── library001.md
│   │   └── ...
│   ├── corpus-quality-report.csv
│   └── ...
├── testset/
│   ├── README.md
│   ├── faeghe.jsonl
│   ├── question-review.csv
│   └── drafts/
│       ├── faeghe-draft.jsonl
│       └── faeghe-qa-proposals.jsonl
└── samples/
    └── ...
```

The `samples/` directory belongs to the original project and contains synthetic data for development and testing.

## 3. Document Corpus

### Data Source

All collected documents originate from publicly accessible official University of Toledo webpages.

Official university website:

https://www.utoledo.edu/

The current WS1 deliverable contains:

- **177 university documents**
- Approximately **100,000 extracted words**
- Multiple university service categories
- Source URLs and document metadata
- Quality assessment and review reports

The documents cover topics such as:

- Information Technology (IT)
- Registrar and Academic Records
- Financial Aid
- Student Housing
- Library Services
- Parking and Transportation
- Dining and Meal Plans
- Student Accounts
- Academic and Student Support Services

### `corpus/sources.csv`

This file lists the original source URL for each university document.

Format:

```csv
doc_id,url,section
it001,https://www.utoledo.edu/...,IT
library001,https://www.utoledo.edu/...,Library
```

Each `doc_id` must be unique and must correspond to the associated Markdown document.

### `corpus/docs/`

This directory stores the extracted webpage content as Markdown files.

Each document contains source metadata and the information extracted from the corresponding university webpage.

For example:

```text
data/corpus/docs/it001.md
```

The RAG retrieval system uses these documents as its knowledge source.

### `corpus/MANIFEST`

The manifest records the corpus fingerprint.

It helps ensure that experiments use the intended frozen document collection.

If the corpus changes, its fingerprint must be regenerated using the project's existing corpus hashing implementation.

### Corpus Quality

WS1 includes quality checks for:

- Missing or incomplete content
- Duplicate documents
- Short or low-information pages
- Extraction problems
- Metadata inconsistencies
- Outdated information
- Source-document accuracy

See the corpus quality reports and `docs/ws1-final-quality-report.md` for details.

Some documents may still require human review. Automatically passing validation does not guarantee that every university policy is current.

## 4. Evaluation Dataset

### Purpose

The test dataset is used to evaluate whether the RAG system retrieves appropriate documents, generates correct answers, and abstains when answers are unsupported.

Each member of the team should contribute approximately **20 labeled evaluation questions**.

The full team target is approximately **150 questions**.

The initial WS1 contribution is stored in:

```text
data/testset/faeghe.jsonl
```

It currently contains 21 primary evaluation cases.

Additional draft questions are stored separately in `data/testset/drafts/`.

### Question Categories

We use seven question types.

| Case Type | Answerable | Description |
|---|---|---|
| `single_doc` | Yes | The answer is supported by one document. |
| `multi_doc` | Yes | The answer requires information from multiple documents. |
| `paraphrase` | Yes | The question uses different wording from the source. |
| `out_of_domain` | No | The question is unrelated to the corpus domain. |
| `missing_info` | No | The topic is relevant, but the required information is absent from the corpus. |
| `false_premise` | No | The question relies on a false or unsupported assumption. |
| `personal` | No | The question requires unavailable personal information. |

### Example Answerable Question

```json
{
  "case_id": "ab-001",
  "question": "How can students reset their university password?",
  "answerable": true,
  "case_type": "single_doc",
  "gold_doc_ids": ["it001"],
  "reference_answer": "A source-supported answer goes here.",
  "must_include": ["a verified fact"],
  "split": "dev",
  "author": "your-name"
}
```

This is a format example only. The reference answer and required facts must be replaced with information verified against the actual gold document.

### Example Unanswerable Question

```json
{
  "case_id": "ab-002",
  "question": "What is my current GPA?",
  "answerable": false,
  "case_type": "personal",
  "split": "dev",
  "author": "your-name"
}
```

Unanswerable questions must not include reference answers, gold-document IDs, or `must_include` facts.

## 5. Instructions for All Team Members

**Please add your own evaluation questions to this directory.**

To avoid merge conflicts, each team member should create a separate JSONL file.

For example:

```text
data/testset/
├── faeghe.jsonl
├── member2.jsonl
├── member3.jsonl
├── member4.jsonl
└── member5.jsonl
```

Please follow these steps:

1. Read `data/testset/README.md` before creating questions.
2. Review the available documents in `data/corpus/docs/`.
3. Create approximately 20 questions using the seven case types.
4. Include both answerable and unanswerable questions.
5. Use a unique case ID prefix based on your initials.
6. Verify reference answers against the corpus.
7. Include valid gold document IDs for answerable cases.
8. Use `"split": "dev"` for new questions.
9. Save your questions in your own `.jsonl` file.
10. Validate your questions before committing changes.

Each line of a JSONL file must contain one complete JSON object.

Do not put all questions inside a single JSON array.

**Please do not modify another team member's question file without coordinating with that person.**

The dataset will be reviewed and finalized before the official evaluation experiments.

## 6. Dataset Quality Requirements

All submitted questions should be:

- Clear and understandable
- Relevant to realistic user scenarios
- Factually correct when answerable
- Supported by the frozen corpus
- Free from duplicates
- Consistent with their assigned case type
- Compatible with the project's `TestCase` schema

For answerable cases, verify that the reference answer and required facts are present in the gold documents.

For unanswerable cases, verify that the required information cannot be established from the frozen corpus.

Do not classify a question as unanswerable simply because one retrieved document does not contain the answer.

The question may be answerable using another document in the corpus.

Machine-generated questions and labels should be reviewed before final acceptance.

## 7. Data Collection and Validation

Run commands from the repository root.

### Install Dependencies

```bash
uv sync --group scrape
```

### Collect University Documents

```bash
uv run deskrag data snapshot
```

The snapshot command reads `sources.csv` and processes the listed webpages.

Document collection should respect public access restrictions, `robots.txt`, and polite request rates.

Do not rerun or modify the frozen corpus during official experiments without team coordination.

### Validate the Dataset

```bash
uv run deskrag data validate
```

This command checks the consistency of the corpus and labeled questions, including document references and required facts.

### Run Automated Tests

```bash
uv run pytest
```

All relevant checks should pass before submitting changes.

Validation warnings should be reviewed rather than automatically ignored.

## 8. Dataset Split and Reproducibility

The project uses two evaluation splits:

- **Development (`dev`)**: Used for implementation, model selection, and tuning.
- **Test (`test`)**: Held out for final evaluation.

New questions should initially use the development split.

The WS1 team will coordinate the stratified split according to the project workplan.

The test split should not be used for prompt tuning, retrieval optimization, or threshold selection.

Once the dataset is frozen, changes to documents or question labels must be documented and coordinated with the team.

## 9. GitHub Contribution Guidelines

When adding or updating data:

1. Work on the appropriate branch.
2. Keep files in their designated directories.
3. Avoid modifying unrelated project components.
4. Run the available validation checks.
5. Review the files changed before committing.
6. Use a descriptive commit message.
7. Open a Pull Request for review.

Suggested commit message:

```text
Add labeled RAG evaluation questions
```

Example Pull Request description:

```text
Added 20 labeled evaluation questions covering
answerable and unanswerable cases.

Verified gold document references and required facts.
Ran the data validation command.
```

Do not commit downloaded caches, virtual environments, private information, or temporary files.

## 10. Collaboration With Other Workstreams

The WS1 data supports the rest of the project:

**WS2 — Retrieval**

Uses `data/corpus/docs/` to build BM25, Dense, and Hybrid retrieval indexes.

**WS3 — Generation**

Uses the retrieved information to generate source-grounded answers through Ollama.

**WS4 — Evaluation**

Uses the labeled questions, reference answers, and gold documents to calculate retrieval, answer quality, and abstention metrics.

**WS5 — Regression Detection**

Uses evaluation results and fixed datasets to detect performance regressions between system versions.

Maintaining stable document IDs, correct labels, and reproducible data is essential for all workstreams.

## 11. Important Notes

- This project uses public University of Toledo webpages, not private university records.
- Some documents may contain historical dates or policies. Verify time-sensitive information against the original source.
- All evaluation answers must be grounded in the collected documents.
- Corpus changes may affect existing question labels.
- Do not modify the frozen dataset during final experiments without coordination.
- Human review remains necessary for quality assurance.

## 12. References

- [University of Toledo](https://www.utoledo.edu/)
- `docs/design.md` — System architecture and evaluation design
- `docs/workplan.md` — Project milestones and responsibilities
- `docs/authoring-guide.md` — Test-case authoring guidance
- `data/testset/README.md` — Exact question format and labeling requirements
- `docs/ws1-final-quality-report.md` — Final WS1 quality review

---

**WS1 Status:** Initial document collection, dataset preparation, and automated validation have been completed. The data is available for team review and integration. Final human verification and dataset freezing should be coordinated with the team.
