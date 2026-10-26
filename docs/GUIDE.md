# compliance-rag: A Beginner's Guide

This guide explains what the project does and every step taken so far, in plain
English. You don't need a background in AI or law to follow it.

---

## 1. What are we building?

A program that answers questions about the **GDPR**, the European Union law about
personal data and privacy.

You ask something like *"Can I ask a company to delete my data?"* The program:

1. **Searches** the law for the parts that matter (here, Article 17).
2. **Writes an answer** using only those parts, and says where it found them.
3. **Checks its own answer** against the law and says how confident it is.

This design is called **RAG** (Retrieval-Augmented Generation):

- *Retrieval* = finding the right text.
- *Generation* = an AI writing the answer from that text.

The AI answers from the actual law instead of from memory, so it is less likely to
make things up.

> **Why search first?** An AI model on its own can sound confident and still be
> wrong. Giving it the exact paragraphs of the law to work from, and asking it to
> point to them, makes answers far more trustworthy.

---

## 2. The plan (roadmap)

| Phase | What | Status |
|---|---|---|
| 1 | Get the GDPR text and cut it into pieces | ✅ Done |
| 2 | Write 50 test questions and build a basic search, then score it | ✅ Done |
| 3 | Make the search smarter (keyword search, combining, reranking) | Next |
| 4 | Add the AI that writes answers and the AI that checks them | Later |
| 5 | Run tests automatically on GitHub whenever code changes | Later |
| 6 | Optional extras (bigger data tools, Azure cloud) | Optional |

---

## 3. Tools we use

| Tool | What it is | Why we need it |
|---|---|---|
| **Python 3.12** | The programming language | All the code is Python |
| **uv** | Installs Python packages and runs the project | One command (`uv sync`) sets everything up |
| **VS Code** | The code editor | Where you read and edit files |
| **Git** | Saves versions of your work | Lets you go back if something breaks |
| **Terminal** | A window where you type commands | Open it in VS Code with **Ctrl + `** |

### Python packages (installed automatically by `uv sync`)

| Package | What it does for us |
|---|---|
| `beautifulsoup4` | Reads the web page and pulls out the text |
| `pyarrow` | Saves tables of data to `.parquet` files (like a fast spreadsheet) |
| `sentence-transformers` | Runs the free AI search model |
| `pytest` | Runs the tests that check our code works |
| `ruff` | Checks the code for mistakes and messy style |

Others (`langchain`, `chromadb`, `streamlit`, etc.) are installed now but only used
in later phases.

---

## 4. Project folders

```
compliance-rag/
├── src/compliance_rag/     ← the actual code
│   ├── ingest.py           ← Phase 1: reads the GDPR and cuts it into pieces
│   ├── retrieval.py        ← Phase 2: the search
│   └── evaluate.py         ← Phase 2: scores the search
├── tests/                  ← small checks that the code works
├── eval/
│   └── gdpr_golden.jsonl   ← the 50 test questions with correct answers
├── data/                   ← downloaded and generated data (not saved in Git)
│   ├── raw/gdpr.html       ← the GDPR web page
│   ├── processed/          ← the cut-up text
│   └── eval/dense.json     ← detailed search results per question
├── docs/GUIDE.md           ← this file
├── pyproject.toml          ← list of packages and commands
├── .env.example            ← template for secret keys (needed from Phase 4)
└── README.md               ← short technical summary
```

---

## 5. Setting up (one time)

Open the terminal in VS Code (**Ctrl + `**) and run:

```powershell
uv sync
```

This creates a hidden folder `.venv` (a **virtual environment**: a private box of
Python packages just for this project) and installs everything into it.

**Do I need to "activate" the virtual environment?** No. Starting commands with
`uv run` uses it automatically. If you want to activate it anyway:

```powershell
.venv\Scripts\Activate.ps1
```

**What about the `.env` file?** It will hold your OpenAI API key. Nothing uses it
until Phase 4, so you can skip it for now. When the time comes, copy
`.env.example` to `.env` and paste your key in. `.env` is never saved to Git, so
your key stays private.

---

## 6. Phase 1: Getting the law into the computer

### Step 1: Download the GDPR

The official text is on EUR-Lex, the EU's legal website:
https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679

We saved the page as **"Webpage, HTML only"** to `data/raw/gdpr.html`.

> **What happened here:** the browser first saved it under the name
> `Webpage, HTML only.xml` in the wrong folder, along with a folder of website
> styling files we don't need. We moved the page to the right place and sent the
> extra folder to the Recycle Bin.

### Step 2: Understand how the GDPR is organised

The GDPR has two kinds of text:

- **Recitals (173 of them):** the background. They explain *why* the law exists
  and what it intends, in fairly everyday language. Example: Recital 65 explains
  the idea behind the "right to be forgotten".
- **Articles (99 of them):** the actual rules. Example: Article 17 says exactly
  when you can have your data erased.

Each one gets an ID: `gdpr-art17` is Article 17, and `gdpr-rct65` is Recital 65.

### Step 3: Pull out the text (`uv run ingest`)

The program `ingest.py` reads the web page and:

1. Finds every recital and article. The EUR-Lex page labels them with IDs like
   `art_17` and `rct_65`.
2. Removes the website clutter and keeps only the legal text.
3. Saves them as **272 documents** (99 articles + 173 recitals).

### Step 4: Cut the text into pieces (chunks)

Some articles are long. A search works better on small pieces, so we cut each
document into **chunks** of at most about 1,200 characters, roughly a long
paragraph or two. We always cut **between** paragraphs, never in the middle of a
sentence.

Result: **396 chunks**, each with an ID like `gdpr-art17-c0` (Article 17, chunk 0).

### Step 5: A bug we found and fixed

When we looked at the real output, some sentences were broken in two:

```
...publication in the
Official Journal of the European Union
.
```

**Why:** on the website, *Official Journal* is in italics, and our code treated
the italic part as a separate line. List items like **(a)** were also cut off from
their text. Since we split chunks at line breaks, that could have split sentences
apart.

**Fix:** we rebuilt the text paragraph by paragraph, kept italics inside their
sentence, and stuck labels like (a) back onto their text. We also added a test
so this bug can't quietly come back.

Run Phase 1 yourself:

```powershell
uv run ingest
```

Expected output: `272 documents (99 articles, 173 recitals) -> 396 chunks`

---

## 7. Phase 2: Building and grading a basic search

### Step 1: Write an answer key (the "golden set")

Before building a search, we need a way to **grade** it. So we wrote 50
questions a normal person might ask, each paired with the article that answers it.

Example lines from `eval/gdpr_golden.jsonl`:

```json
{"id": "q01", "question": "Can I ask a company to delete all the personal data it holds about me?", "relevant": ["gdpr-art17"]}
{"id": "q38", "question": "What is the maximum fine for breaking the GDPR?", "relevant": ["gdpr-art83"]}
```

Important points:

- **A person wrote these, not an AI.** The correct answers come from reading the
  GDPR. This is the answer key the search gets graded against.
- The questions are in **everyday words** on purpose, because that's how real
  users ask. They don't copy the law's wording.
- A few questions have two correct answers (e.g. an article and a recital).
- Answers are linked to the article (`gdpr-art17`), not to a chunk
  (`gdpr-art17-c0`). That way, changing the chunk size later doesn't break the key.

### Step 2: Build the search

The search is in `retrieval.py`. It uses a free AI model called
**`BAAI/bge-small-en-v1.5`**.

**What the model does:** it turns a piece of text into a list of 384 numbers,
called an **embedding**. Texts with similar *meaning* get similar numbers, even
when they use different words. "Delete my data" and "right to erasure" end up
close together.

How the search works:

1. **Once at the start:** turn all 396 chunks into numbers.
2. **For each question:** turn the question into numbers too.
3. **Compare:** score each chunk by how close its numbers are to the question's
   numbers.
4. **Return** the chunks with the highest scores.

This is called **dense retrieval**, or **semantic search** (search by meaning).

**Why this model?**

- **Free.** No API key, no cost.
- **Runs on a normal laptop.** No special graphics card needed.
- **Small** (about 130 MB), so it's quick to download and run.
- **Good for its size** on public search benchmarks.

That makes it a good **baseline**: a simple first version to measure everything
else against.

> **Note:** this model does **not** write answers. It only finds text. Writing
> answers comes in Phase 4.

### Step 3: Grade the search (`uv run evaluate`)

For each of the 50 questions, we search and check where the correct article
appears in the results. There are two kinds of score.

**Recall@k:** "Is the right answer in the top *k* results?"
For example, Recall@5 = 74% means that for 74% of questions, the correct article
was somewhere in the top 5.

**MRR (Mean Reciprocal Rank):** "How close to the top is the right answer?"

- Correct answer ranked 1st → 1 point
- 2nd → 1/2 point
- 3rd → 1/3 point
- Not found → 0

The average over all questions is the MRR. 1.0 would be perfect.

### Step 4: Results

| Is the right article in the... | Score |
|---|---|
| Top 1 result | 36% |
| Top 3 | 64% |
| Top 5 | 74% |
| Top 10 | 84% |
| **MRR** | **0.545** |

In plain words: the right article is usually somewhere near the top, but it's
ranked **first** only about a third of the time.

### Step 5: Why is it failing?

We looked at the questions it got wrong. There are three main reasons.

**1. It finds the explanation instead of the rule.**
The most common mistake. The GDPR explains each topic twice: in a recital
(background) and in an article (the rule). The search often ranks the recital
first.

| Question | Should be | Search put first |
|---|---|---|
| When does a company have to tell affected people their data was leaked? | Article 34 (the rule) | Recital 85 (the explanation) |
| Is a company allowed to process health data? | Article 9 | Recital 53 |
| What must a company tell me when it collects my data? | Article 13 | Recital 61 |

**Why:** recitals are written in more everyday language, so they sound more like
our questions. Articles use formal legal wording.

**2. Neighbouring articles look alike.**
Articles 37, 38 and 39 are all about the Data Protection Officer: who needs one,
their position, their tasks. The search finds all three but sometimes in the
wrong order.

**3. Everyday words vs legal words.**
*"Two companies decide together how data is used. Who is responsible?"* The legal
term is **"joint controllers"** (Article 26), but the question never uses those
words, and the small model didn't make the connection.

**Is this bad?** No. A first version is supposed to be simple. Now we have a
number to beat, and we know exactly what kinds of mistakes to fix.

Run Phase 2 yourself:

```powershell
uv run evaluate
```

Per-question details are saved to `data/eval/dense.json`.

---

## 8. Phase 3 (next): Making search smarter

Each idea targets one of the problems above:

1. **Keyword search (BM25):** an old-school search that matches exact words.
   It's great when the question contains specific terms like "data protection
   officer" or "standard contractual clauses".
2. **Combining both (RRF, Reciprocal Rank Fusion):** merges the ranking from
   meaning-search and the ranking from keyword-search into one list. A chunk that
   ranks well in both moves to the top.
3. **Reranking (cross-encoder):** a second, more careful AI model reads the
   question together with each of the top ~20 chunks and re-scores them. It's
   slower but more accurate, so we only use it on the shortlist.

After each step, we rerun the same 50 questions and compare scores.

---

## 9. Saving our work with Git

Git keeps **save points** (called **commits**) of the project.

- It works **entirely on your computer**. A hidden `.git` folder inside the
  project holds the history.
- **GitHub** is an optional online copy. We can connect it later, and all the
  save points go up with it.
- Some things are deliberately **not** saved (listed in `.gitignore`): the
  `.venv` folder, the `data/` folder (it can be rebuilt with `uv run ingest`), and
  `.env` (your secret keys).

Save points so far:

1. `Phase 1: GDPR ingestion and chunking`
2. `Phase 2: golden set and dense-retrieval baseline`

Useful commands:

```powershell
git log --oneline     # list save points
git status            # see what changed since the last save point
```

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **GDPR** | EU law on personal data and privacy (Regulation 2016/679) |
| **Article** | A rule in the law |
| **Recital** | Background explanation at the start of the law |
| **RAG** | Retrieval-Augmented Generation: find the text first, then let the AI answer from it |
| **Chunk** | A small piece of a document, used for searching |
| **Embedding** | A list of numbers that represents the meaning of a text |
| **Dense retrieval** | Searching by meaning using embeddings |
| **BM25** | Classic keyword search |
| **Reranker / cross-encoder** | A slower, more careful model that re-orders the top results |
| **Golden set** | Questions with known correct answers, used to grade the system |
| **Baseline** | The first simple version, which later versions must beat |
| **Recall@k** | Share of questions where the right answer is in the top k results |
| **MRR** | Average of 1/rank of the first correct result; 1.0 is perfect |
| **Virtual environment (.venv)** | A private set of Python packages for one project |
| **uv** | The tool that installs packages and runs commands |
| **Commit** | A save point in Git |
| **API key** | A secret password that lets the program use a paid AI service |
