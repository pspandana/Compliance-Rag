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
| 3 | Make the search smarter (keyword search, combining, reranking) | ✅ Done |
| 4 | Add the AI that writes answers and the AI that checks them | ✅ Done |
| 5 | Grader agent, abstain / human review, fuller confidence score, article filter, LLM judge, regression gate, GitHub Actions file | ✅ Done (the GitHub part is untested until the repo is pushed) |
| 6 | Second version of the law (UK GDPR): metadata, jurisdiction-aware search and answers | ✅ Done |
| 7 | Ingestion rebuilt in PySpark (clean, tag, dedupe, link, chunk) | ✅ Done (small data, local mode) |
| 8 | Chunk-size and prompt tuning, with a held-out half and a noise measurement | ✅ Done (no change adopted: nothing beat the noise) |
| 9 | Not started: Docker, Terraform, Azure AI Foundry, Confluence, GitHub run of CI | Not started |

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
│   ├── ingest.py           ← Phases 1, 6: reads the EU and UK GDPR, cuts them into pieces
│   ├── ingest_spark.py     ← Phase 7: the same job as Spark stages
│   ├── scope.py            ← Phase 6: decides EU, UK or both for a question
│   ├── retrieval.py        ← Phases 2-3, 6: the search methods and metadata filters
│   ├── evaluate.py         ← Phases 2-3: scores the search
│   ├── llm.py              ← Phase 4: connects to OpenAI
│   ├── answer.py           ← Phases 4-5: the full pipeline and its evaluation
│   ├── grader.py           ← Phase 5: the Grader agent
│   ├── confidence.py       ← Phases 4-5: Bayesian confidence and the routing rule
│   ├── calibrate.py        ← Phase 5: turns reranker scores into probabilities
│   ├── judge.py            ← Phase 5: LLM judge for answer correctness
│   ├── tune_chunks.py      ← Phase 8: compares chunk sizes
│   ├── tune_prompts.py     ← Phase 8: compares prompt variants, with a noise measurement
│   └── gate.py             ← Phase 5: fails if quality drops below the baseline
├── tests/                  ← small checks that the code works
├── .github/workflows/ci.yml ← Phase 5: GitHub runs lint, tests and the gate
├── eval/
│   ├── gdpr_golden.jsonl   ← the 50 test questions with correct answers
│   ├── jurisdiction_golden.jsonl ← 20 UK-only and EU-vs-UK questions
│   ├── gdpr_unanswerable.jsonl ← 12 questions the GDPR cannot answer
│   ├── calibration.json    ← the fitted score-to-probability curve
│   └── baseline.json       ← the numbers the gate compares against
├── data/
│   ├── raw/gdpr.html       ← the GDPR web page (not saved in Git)
│   ├── processed/          ← the cut-up text (saved in Git, so GitHub can run the checks)
│   └── eval/               ← detailed results per question (not saved in Git)
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

**What about the `.env` file?** It holds your OpenAI API key and is needed from
Phase 4. **Copy** `.env.example` to `.env` (keep the original; it's the template
for anyone else using the project) and paste your key after `OPENAI_API_KEY=`.
`.env` is never saved to Git, so your key stays private.

> **What happened here:** the template was renamed to `.env` rather than copied.
> We restored `.env.example` from Git. No harm done, and the key was never
> saved in Git.

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

## 8. Phase 3: Making search smarter

We tried three improvements, one on top of the other, and graded each one on the
same 50 questions. All the code is in `retrieval.py`.

### Step 1: Keyword search (BM25)

**What it is:** a classic search method, the same family as old search engines.
It doesn't understand meaning; it counts matching words. It scores a chunk higher
when:

- the chunk contains the question's words,
- those words are **rare** in the GDPR overall ("encryption" counts for more than
  "data", which is everywhere),
- the chunk is short, so the match isn't diluted.

**Preparing the words (tokenizing):** before searching, we lowercase everything,
split it into words, and drop very common words like "the", "is", "what" (called
**stopwords**). *"What is the Right to Erasure?"* becomes `["right", "erasure"]`.

**Why try it:** meaning-search can miss exact legal terms. If someone asks about
"standard contractual clauses", keyword search finds chunks with those exact words.

**Result on its own:** worse than meaning-search (correct article first only 26%
of the time). People ask in everyday words, and the law uses legal words. "Delete
my data" doesn't contain the word "erasure", so keyword search can't connect them.
Its job is to *help* the other search, not replace it.

### Step 2: Combining both (hybrid search with RRF)

**The problem:** we now have two ranked lists per question, one from each search.
Their scores are on completely different scales (meaning-search gives 0 to 1,
keyword search gives 0 to 30 or more), so we can't just add the scores.

**The solution: Reciprocal Rank Fusion (RRF).** Ignore the scores and use only
**positions**. Each list gives a chunk points based on where it ranks:

```
points = 1 / (60 + position)
```

- 1st place → 1/61 ≈ 0.0164 points
- 2nd place → 1/62 ≈ 0.0161 points
- 10th place → 1/70 ≈ 0.0143 points

Add up the points from both lists. A chunk that ranks **fairly well in both**
beats a chunk that ranks first in one and is missing from the other. The 60 is a
standard value that stops the first place from dominating everything.

**Example:** chunk B is 2nd in both lists (0.0161 + 0.0161 = 0.0323 points).
Chunk A is 1st in one list only (0.0164 points). B wins, because both searches
agree it's relevant.

**Result:** a small improvement (correct article first 37% of the time, in the
top 3 for 69%).

### Step 3: Reranking (cross-encoder)

**The idea:** the first two searches are fast but shallow. They look at the
question and each chunk **separately**, turn them into numbers, and compare the
numbers. A **cross-encoder** reads the question and the chunk **together**, word
by word, the way a person would, and judges how well the chunk answers the
question. That's much more accurate but much slower.

**The trick:** use the fast search to make a **shortlist** (the top 30 chunks),
then let the slow, careful model re-order only those 30.

- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Why this one:** free, small (about 90 MB), runs on a laptop, and was trained
  on a million real search questions from Bing. It's a well-known, reliable
  starting choice.

**Result:** the biggest jump. The correct article comes first 47% of the time
(was 36%), and it's in the top 3 for 80% (was 64%).

### Phase 3 results: everything side by side

| Search method | Right article 1st | In top 3 | In top 5 | In top 10 | MRR |
|---|---|---|---|---|---|
| Meaning-search only (Phase 2) | 36% | 64% | 74% | 84% | 0.545 |
| Keyword search only | 26% | 49% | 61% | 70% | 0.445 |
| Both combined (hybrid) | 37% | 69% | 77% | 81% | 0.577 |
| **Combined + reranker** | **47%** | **80%** | **84%** | **86%** | **0.670** |

**The winner is "combined + reranker"**, and we'll use it in Phase 4.

### What's still going wrong?

Looking at the 25 questions the best version still gets wrong, **most have the
correct article in 2nd place, right behind its matching recital**. For example,
for "How long does a company have to tell the regulator about a data breach?" it
returns Recital 85 first and Article 33 second.

That matters less than it sounds. In Phase 4 the AI will read the **top few**
chunks, not just the first, so it will see both the explanation and the rule.
The top-3 score (80%) is the number that matters most for the answers.

**Things we could try later:**

- **Give articles a small boost** over recitals. It's easy, but we'd have to be
  careful not to tune it just to make our 50 questions look good.
- **A bigger reranker model** (e.g. `BAAI/bge-reranker-base`). It's more
  accurate but about 10 times larger and slower.
- **More test questions**, so the scores are more reliable. With 50 questions,
  one question = 2%.

### Run Phase 3 yourself

```powershell
uv run evaluate                  # score all four methods
uv run evaluate --show-misses    # also list the questions each method gets wrong
uv run evaluate --retrievers dense rerank   # only some methods
```

The first run downloads the reranker model (about 90 MB).

---

## 9. Phase 4: Writing, checking and scoring answers

Now the system actually **answers** questions. Try it:

```powershell
uv run ask "Can I ask a company to delete my data?"
```

### Which AI model, and why

- **Model:** `gpt-5.4-mini` from OpenAI (paid, needs the API key in `.env`).
- **Why:** it's recent, fast and cheap, and writing short answers from given text
  doesn't need the biggest model. All 50 test questions cost well under $1.
- **Changing it:** put a different name in `.env`, e.g. `OPENAI_MODEL=gpt-5.5`.
  The code is in `llm.py`, which can also switch to Azure OpenAI with
  `LLM_PROVIDER=azure`.

### What happens when you ask a question (5 steps)

**Step 1: Search.** The best search from Phase 3 (combined + reranker) finds the
top **5** chunks of the GDPR.

**Step 2: Write the answer (the "generator").** The AI gets the question and the 5
chunks, each labelled with its ID (like `gdpr-art17-c0`). Its instructions:

- answer in plain English, using **only** these chunks (no outside knowledge),
- then list **every fact** in the answer as a separate **claim**,
- next to each claim, say which chunk it came from (a **citation**).

We ask for this in a fixed format (called **structured output**), so the program
can read the claims one by one instead of guessing from free text.

**Step 3: Fact-check (the "checker").** A second AI call reads the same chunks and
each claim, and decides for every claim: **supported** or **unsupported**, with a
one-sentence reason. Its instructions are strict: anything added, exaggerated or
not in the cited chunk counts as unsupported.

The program also applies one rule itself: a claim that cites **nothing**, or
cites a chunk the AI was never shown (a made-up citation), is automatically
unsupported.

**Step 4: Confidence score (the Bayesian part).** See the next section.

**Step 5: Self-correction.** If **any** claim was unsupported, the system tries
again **once**:

- with **more** text (top 10 chunks instead of 5),
- and with the checker's feedback ("these claims were rejected because...").

Then it keeps whichever attempt has the higher confidence.

### The Bayesian confidence score, explained

We want a number that says "how much should I trust this answer?"

**The simple way (and its problem):** count supported claims. "3 of 3 supported"
= 100%. "1 of 1 supported" = also 100%. But one check is much weaker evidence
than three. A simple percentage can't tell them apart.

**The Bayesian way:** think of each claim as a coin toss that lands "supported"
with some unknown probability. We start with **no opinion**: every probability
from 0% to 100% is equally likely (this starting belief is called the **prior**).
Each fact-check result is evidence that updates that belief. The updated belief
is called the **posterior**.

The math for this kind of yes/no evidence is a **Beta distribution**, and the
update is just counting:

```
start:  Beta(1, 1)                              ← no opinion
after:  Beta(1 + supported, 1 + unsupported)
confidence = (1 + supported) / (2 + supported + unsupported)
```

| Fact-check result | Confidence | 90% range | Meaning |
|---|---|---|---|
| 1 of 1 supported | 0.67 | 0.22 – 0.97 | Looks good, but very little evidence |
| 3 of 3 supported | 0.80 | 0.47 – 0.99 | Good |
| 8 of 8 supported | 0.90 | 0.72 – 0.99 | Strong, and we're sure about it |
| 1 of 4 supported | 0.33 | 0.08 – 0.66 | Don't trust this answer |

The **90% range** (called a **credible interval**) shows how sure we are about the
score itself. Few checks give a wide range; many agreeing checks give a narrow one.

We also give a simple label: **high** (0.8 or more), **medium** (0.6 to 0.8),
**low** (under 0.6), or **none** (no claims at all, e.g. the text didn't contain
the answer).

The math uses the `scipy` package, and the code is in `confidence.py`.

### Example output

```
Answer: Yes. Under the GDPR, you can ask the controller to erase your personal
data without undue delay in certain situations, such as when the data is no
longer needed, you withdraw consent, ...

Claims:
  [supported]   Under the GDPR, a data subject can ask the controller to erase
                personal data without undue delay in certain situations.  ['gdpr-art17-c0']
  [supported]   One situation is when the personal data is no longer necessary ...
  ...

Confidence: high - 0.89 (90% range 0.69-0.99), 7 of 7 claims supported
Sources: Article 17 - Right to erasure ('right to be forgotten'); Recital 66
Attempts: 1
```

> **Note:** this section describes the *first version* of the answer pipeline. Phase 5
> (next section) added a grader, abstaining, a fuller confidence score and a judge, so
> the numbers below are the old ones.

### Testing it on all 50 questions (first version)

This was run with the Phase 4 pipeline (it has since been replaced):

```powershell
uv run ask --golden 50
```

This answers every test question and checks whether the answer **cites the
correct article** from our answer key. Details are saved to `data/eval/answers.json`.

| Measure | Result |
|---|---|
| Answer cites the correct article | **86%** (43 of 50) |
| Claims supported by the cited text | **100%** (all 205 claims, after retries) |
| Average confidence | 0.82 |
| Questions that needed a retry | 4 |

### What this tells us (important!)

**1. The answers stick to the text.** After self-correction, every claim was
backed by the passage it cited. The AI isn't making things up.

**2. But "supported" doesn't mean "complete" or "best".** The checker only asks
"does the cited text say this?" It does **not** ask "was this the right text to
look at?" When the search misses the key article, the AI writes a faithful answer
from the wrong or weaker passages, and still gets a high score. Examples:

- **q44** ("Do they have to tell others about my correction?"): the search missed
  Article 19, so the answer used Articles 16 and 17 instead. Confidence 0.86.
- **q04** ("What legal grounds allow processing?"): the search missed Article 6, the
  main answer, so the answer was built from side passages.

So **confidence measures faithfulness, not correctness**. The search is still the
weakest link; when it fails, everything after it is limited.

**3. The low scores were the right ones to doubt.** The lowest confidence (0.60)
was **q43**, which cited the wrong article (it confused codes of conduct with
binding corporate rules). The two 0.67 scores (q49, q50) were correct answers
with only one claim each, so the system honestly said "little evidence".

**4. Some misses are softer than they look.** For **q29** (family photos), the
answer cited Recital 18, which explains exactly that exemption. It's a good
source, just not the article in our answer key.

**5. The same model checks its own work.** Using one AI to both write and check
is cheaper, but it can share the same blind spots. A different (or bigger) model
as the checker would be a stronger test.

### Ideas to improve later

- Better search (Phase 3 ideas): a bigger reranker, more chunks, boosting articles.
- A **completeness check**: ask "does this fully answer the question?" and count
  that in the confidence score too.
- Use a **different model** for the fact-checker.
- Add **more test questions** so the numbers are more reliable.

---

## 10. Phase 5: Smarter agents, safer decisions, and a quality gate

Phase 4 had a weakness: it always produced an answer, even when the GDPR couldn't
answer the question, and it fed the writer some off-topic text. Phase 5 fixes that
and adds a safety net. The pipeline is now **three AI agents** plus a decision step:

```
question → search (articles only) → GRADER → GENERATOR → VERIFIER → confidence → decision
                                       ↓ nothing relevant                           ├ auto-answer
                                    abstain                                         ├ human review
                                                                                    └ abstain
```

Try it (the last one should refuse):

```powershell
uv run ask "Can I ask a company to delete my data?"
uv run ask "Who won the 2018 FIFA World Cup?"
```

### Step 1: Metadata filter ("articles only")

Every chunk now carries a tag, `kind`: **article** or **recital**. The search can be
told to look only at chunks with a given tag (a **metadata filter**).

Articles are the binding rules; recitals are background. Searching articles only:

| Search | Right article 1st | In top 3 | MRR |
|---|---|---|---|
| Articles and recitals (Phase 3) | 47% | 80% | 0.670 |
| **Articles only** | **81%** | **88%** | **0.878** |

**Be careful with this result.** I wrote the answer key, and 48 of the 50 questions
are labelled with an *article*. So the filter is partly rewarded for matching my
labelling style. Recitals are still useful for explaining a rule. The pipeline uses
articles-only by default; `--kind any` turns the filter off.

### Step 2: The Grader agent

Search ranks chunks but never says "none of these help". The **Grader** is an AI
call that reads the question and each of the top 8 chunks and marks each one
relevant or not. Only relevant chunks (at most 5) go to the writer. If *no* chunk
is relevant, we try once with a bigger pool (15), then give up.

Measured on the 50 questions (a chunk counts as "irrelevant" if it isn't from a
document our answer key lists):

| | Result |
|---|---|
| Irrelevant chunks in the context, before the grader | 66% |
| Irrelevant chunks in the context, after the grader | **25%** |
| Questions where the grader kept the right article (when it was in the pool) | **100%** |

This measure is strict, since a neighbouring article can still be useful. We did not
run the pipeline without the grader to compare answer quality, so we know the grader
removes noise but not exactly how much it improves answers.

### Step 3: Abstain, or send to a human

The system now makes one of three decisions:

- **Abstain**: the grader found nothing relevant, or the writer found no claims to
  make. It says "I can't answer this from the GDPR text I have."
- **Auto**: confident enough to answer directly.
- **Review**: an answer exists but isn't trusted enough; a person should look.

To test this we wrote **10 questions the GDPR cannot answer** (HIPAA fines, the EU AI
Act, Irish tax, the World Cup, ...) in `eval/gdpr_unanswerable.jsonl`.

| Unanswerable questions (10) | Result |
|---|---|
| Abstained | **10 of 10** (one earlier run sent 1 to human review instead) |
| Answered automatically (the bad outcome) | **0** |

### Step 4: A fuller Bayesian confidence score

Phase 4's score used only the fact-checker. Now it combines **three kinds of
evidence** in one Beta distribution. Each is a yes/no observation about "can we trust
this answer?":

| Evidence | How it counts |
|---|---|
| **Verifier** | each claim: supported = 1 success, unsupported = 1 failure |
| **Grader agreement** | for the 3 best-ranked chunks: grader says relevant = success, not relevant = failure |
| **Retrieval relevance** | the search's best score, turned into a probability *p*; counts as 2 half-observations (*p* successes and 1−*p* failures) |

```
confidence = (1 + successes) / (2 + successes + failures)    ← the posterior mean
```

**Calibration (turning a score into a probability):** the reranker gives raw scores
like −5.9 or −10.7 that don't mean anything by themselves. `uv run calibrate` fits a
simple curve using the 60 questions (50 answerable, 10 not) so that a score becomes
an honest probability. For example, a score of −10 (off-topic) maps to about 0.29 and
−4 (a good match) to about 0.71.

**The decision rule** (`confidence.py`): abstain if no claims or no relevant chunks;
**auto** if confidence ≥ 0.80 *and* the lower end of the 90% range ≥ 0.55; otherwise
**review**.

### Step 5: Grading answers with a judge (LLM-as-judge)

Until now we checked "did the answer cite the right article?" and "are the claims
supported?". Neither says whether the answer is actually **correct**. So a bigger AI
(`gpt-5.4`, not the small writer model) acts as a **judge**: it reads the question,
the system's answer and the **text of the gold articles** from our answer key, and
says *correct*, *partial* or *incorrect*. Using a bigger, different model than the
writer reduces the chance of it sharing the writer's mistakes.

Words used in the project for the three quality measures:

- **Groundedness / faithfulness**: are the claims supported by the retrieved text?
  (the verifier; in this project they are the same measurement)
- **Correctness**: does the answer agree with the gold text? (the judge)
- **Citation hit**: does it cite a labelled document?

### Results on the 50 answerable questions (the saved baseline)

| Measure | Result |
|---|---|
| Cites a labelled document | 94% |
| Claims supported (groundedness) | 99.4% |
| Judged **correct** | **80%** (12% partial, 8% incorrect) |
| Routed to auto / review / abstain | 32% / 68% / 0% |

The AI is not fully repeatable: a second run gave 96% / 100% / 80% and 24% auto.
That is why the gate (below) allows a small margin for answer metrics.

### The honest finding: what the confidence score can and cannot do

- ✅ It **separates answerable from unanswerable questions almost perfectly**
  (AUC ≈ 1.0; every unanswerable question scored 0.46 or less, and 9 of 10 are
  abstained outright).
- ❌ It does **not** predict whether an answer is *correct*. Among answerable
  questions, auto-answered answers were correct 69% of the time vs 85% for those sent
  to review (AUC 0.41, no better than chance, on only 50 questions). Several
  **partial** answers got high confidence because the verifier only checks that what
  is *said* is supported, not that the answer is *complete*.
- So today the score is good at "should we refuse?" and weak at "is this right?". Also,
  routing 68% of answers to human review is cautious, not clever. We did **not** tune
  the thresholds to make the numbers look better.

**Ways to improve it next:** a "completeness" check in the verifier; more questions
(50 is small, and one question moves a number by 2%); a stronger or separate verifier
model.

### Step 6: The regression gate

The **gate** is a script that fails when quality drops below a saved baseline
(`eval/baseline.json`), so a change that quietly makes things worse gets caught.

```powershell
uv run gate               # retrieval metrics only; free, no API key (what CI runs)
uv run gate --answers     # also the full answer evaluation (costs a little API money)
uv run gate --update      # save today's numbers as the new baseline
```

| What is checked | Allowed drop |
|---|---|
| Recall@1, Recall@3, MRR | 0.02 (search is deterministic, so this is tight) |
| Cites, supported, correct, abstained | 0.06 (AI answers vary run to run) |
| Unanswerable questions answered automatically | none (must stay 0) |

Run it after changing chunking, search, prompts or models.

### Step 7: GitHub Actions (written, not yet run)

`.github/workflows/ci.yml` tells GitHub to run, on every push: the linter, the unit
tests, and the retrieval gate. A second job runs the full answer gate, only when you
start it by hand (it costs API money and needs an `OPENAI_API_KEY` secret in the
repo settings). **This file has never run**, because the project isn't on GitHub yet.
Expect a first-run fix or two.

To make this possible the cleaned GDPR text (`data/processed/`, about 300 KB of public
EU law) is now saved in Git, since GitHub doesn't have the web page we downloaded.

### Tests

17 unit tests pass: the parser, the chunker, the scoring maths, the confidence
maths, the calibration, routing and the gate comparison. The AI-calling parts (grader,
generator, verifier, judge) are checked by the evaluation runs, not by unit tests.

---

## 11. Phase 6: A second version of the law (the UK GDPR)

So far we had one document. Real compliance questions often involve **several
versions of the same rules**, and the versions disagree. After Brexit the UK copied
the GDPR into its own law and has been changing it since. We added that copy so the
system has to answer: *which version are you asking about?*

### Step 1: Get the UK text

Source: https://www.legislation.gov.uk/eur/2016/679/data.htm (the whole document in
one page; free to reuse under the Open Government Licence). Saved as
`data/raw/uk_gdpr.html`. This is the **latest revised text**, including changes made
by the Data (Use and Access) Act 2025, retrieved on 2026-10-26.

### Step 2: Read it (a second parser)

The UK page looks nothing like the EU one, so `ingest.py` has a second parser,
`parse_uk_gdpr_html`. The page is full of **amendment markup** that must be stripped
or it pollutes the text and the search:

- little footnote markers like `F57`,
- square brackets `[ ]` around changed words,
- a "U.K." label on every heading,
- repealed paragraphs, which appear as `...`.

Facts about the UK text, found by reading it rather than assuming:

| | EU GDPR | UK GDPR |
|---|---|---|
| Articles in our data | 99 | 82 (some UK-only, like 22A–22D, 45A–45C, 84A–84D) |
| Recitals | 173 | **none** (the UK version doesn't include them) |
| Missing UK articles | – | 38 article numbers are repealed there, e.g. 22, 44, 45, 53–56, 59–77, 81, 87–93, 97–99 |

The repealed ones matter. For example **Article 77, the right to complain to the
regulator, is not in the UK GDPR text**: the UK moved it to the Data Protection Act
2018, which we don't have. So the system can't answer UK complaint questions, and
it should say so.

### Step 3: Tag every piece of text

Each document and chunk now carries extra **metadata**:

| Field | Example |
|---|---|
| `jurisdiction` | `EU` or `UK` |
| `effective_date` | `2018-05-25` (EU: when it started to apply) or `2020-12-31` (UK: end of the Brexit transition) |
| `version` | `Regulation (EU) 2016/679, OJ L 119, 4.5.2016` or `UK GDPR, latest revised text ... (retrieved 2026-10-26)` |

Be clear about what "version" means here: **two jurisdictions' versions**, not a
history of one text changing over time. We do not have dated revisions of the same
article.

### Step 4: Search the right version (filters)

The search can now filter on any tag (`kind`, `jurisdiction`) **before** ranking, so
chunks from the wrong version can never take a result slot. One shared index serves
every filter (`retrieval.py`).

Which version to use is decided by `scope.py`, with plain rules and no AI:

| The question... | Searches | Example |
|---|---|---|
| mentions the UK only | UK GDPR | "What is the maximum fine in the UK?" |
| mentions the EU only | EU GDPR | "...outside the EU..." |
| mentions both, or asks to compare/differ | both (half the results from each) | "How does the consent age differ between the EU and UK?" |
| mentions neither | **EU, and the answer says it assumed so** | "Can I ask a company to delete my data?" |

The writer is told which version each passage comes from, must say which version
each statement applies to, must state differences when the two disagree and say
when they agree, and the fact-checker rejects claims credited to the wrong version.

### Step 5: New test questions

`eval/jurisdiction_golden.jsonl` has 20 questions I wrote after reading the UK text:

- **10 UK-only**, e.g. the UK age of consent (13, against 16 in the EU text), UK
  fines (£17.5 million, against €20 million), "recognised legitimate interests"
  (new in the UK), and the UK's split-up automated decision rules.
- **10 comparisons**, e.g. consent age and fines, and also **"are they the same?"**
  cases (breach deadline, data protection officer rules), which check the system
  doesn't invent differences. One (complaints) is deliberately a case where the UK
  text is silent.

We also added 2 more unanswerable questions (about the UK Data Protection Act 2018
and which EU countries lowered the consent age) because related text exists but
doesn't answer them.

### Results

Search (articles only, with the reranker):

| | R@1 | R@3 | MRR |
|---|---|---|---|
| 50 original EU questions (unchanged by adding the UK) | 0.81 | 0.88 | 0.878 |
| 20 jurisdiction questions | 0.525 | 0.925 | 0.804 |

R@1 is lower on the 20 only because 10 of them have **two** correct answers (one per
version), so at most half are in position 1. The right articles are in the top 3 for
92.5%.

Answers (the saved baseline; AI answers vary a few points run to run):

| | 50 original | 20 jurisdiction | 12 unanswerable |
|---|---|---|---|
| Cites the labelled article(s) | 94% | 95% (all of them) | – |
| Claims supported by the text | 97% | 98% | – |
| Judged correct | 76% | 90% | – |
| Abstained / sent to review / answered automatically | – | – | 10 / 2 / **0** |

### What went wrong (honest notes)

- **A "boilerplate" article crowds the results.** UK Article 91A ("Regulations made by
  the Secretary of State") scores high for almost any UK-flavoured question because
  it is full of UK and regulation vocabulary. It often takes the top spot, pushing
  the right article to second. We left it visible rather than patching around it.
- **The "are the legal bases the same?" question (c05) failed.** An abstract
  question like that matches no article well, the grader found nothing relevant, and
  the system abstained. That is safe but wrong, and the judge marks it incorrect.
- **Correctness on the 50 original questions dipped from 80% to 76%.** That is within
  the run-to-run variation we have seen, and the corpus and prompts also changed, so
  we can't say why. We did not tune to bring it back.
- **Unanswerable questions about nearby law went to review, not abstain.** The text
  looked related enough to produce a cautious answer. None were answered
  automatically.
- **The confidence score still doesn't predict correctness** (see Phase 5).

---

## 12. Phase 7: Rebuilding ingestion in PySpark

Apache Spark is a tool for processing large amounts of data in parallel; **PySpark**
is its Python interface. The pipeline in `ingest_spark.py` does the same job as
`ingest.py`, as Spark stages:

| Stage | What it does |
|---|---|
| 1. Parse | Read the raw web pages into records (BeautifulSoup, as before) |
| 2. Clean | Tidy whitespace in text and titles |
| 3. Tag | Check every record has valid metadata (jurisdiction, a real effective date, a version) and non-empty text; stop if not |
| 4. Deduplicate | Drop records whose normalised text repeats within the same jurisdiction |
| 5. Link | For each document, list the other-jurisdiction documents with **word-for-word identical** text (`same_text_as`) |
| 6. Chunk | Split every document into chunks, in parallel, using the same packing rules |

Run it with:

```powershell
uv run --group spark spark-ingest
```

### Results

| | |
|---|---|
| Documents in → out | 354 → 354 |
| Duplicates removed | **0** |
| Documents identical in the other jurisdiction | 10 (five articles: 7, 11, 16, 19, 20) |
| Chunks | 576 |
| Spark time | about 15 seconds |
| Output vs the plain-Python pipeline | **identical** (checked automatically on every run) |

### Be honest about what this shows

- **It's a working pipeline, not a demonstration of scale.** 354 documents is tiny;
  plain Python does the same in a second. Spark runs on one laptop ("local mode").
  The value is that the pipeline is structured to scale, not that we proved it.
- **Deduplication found nothing real**, because nothing repeats in these two
  documents. The duplicate-removal code is proven only by a unit test that plants a
  duplicate on purpose.
- **The "link" stage found something useful:** five articles didn't change between
  the EU and UK. For those, a comparison answer should say "the same".
- It adds two columns to the data (`text_hash` on documents and `same_text_as` on
  both); nothing downstream needed to change. Retrieval scores were identical before
  and after.

### Setting up Java (needed by Spark)

Spark runs on Java. The normal Windows installer needs an administrator pop-up that a
background session can't click, so we used the **portable** version instead: a zip
of Temurin JDK 17 from the official release, **checked against its published
checksum** (the first download was cut off, the check caught it, and we downloaded
again), unzipped into `%LOCALAPPDATA%\compliance-rag\jdk17`. Nothing was installed
system-wide. The pipeline finds Java there automatically; if you have Java elsewhere,
set `JAVA_HOME`.

PySpark is an **optional** package group, so people (and GitHub) who don't need Spark
don't have to install it. The Spark unit tests are skipped when it's missing.

---

## 13. Phase 8: Tuning chunk size and prompts

"Tuning" means trying settings and keeping the best. With only 70 questions it is easy
to fool yourself, so the rules were written **before** running anything.

### The safeguards

- **Tune half and holdout half.** The 70 answerable questions (50 EU + 20 jurisdiction)
  are split by alternating position: 35 to choose a setting on, 35 to confirm it on.
  A setting that only wins on the questions it was chosen on is probably luck.
- **A noise measurement.** The AI is not repeatable. We ran the *same* prompt twice;
  any difference between those runs is pure chance, and a new prompt has to beat that.
- **A decision rule, fixed in advance** (below), so we can't decide a result is good
  after seeing it.

### Chunk size (search only, free to run)

Rule: adopt a size other than 1200 only if tune MRR improves by at least 0.02 and
holdout MRR doesn't drop by more than 0.01.

| Max chunk size (chars) | Chunks | Tune MRR | Holdout MRR |
|---|---|---|---|
| 400 | 1223 | 0.780 | 0.879 |
| 600 | 930 | 0.824 | 0.885 |
| **800** | 748 | **0.852** | **0.894** |
| **1200 (current)** | 576 | 0.835 | 0.878 |
| 1800 | 472 | 0.811 | 0.869 |
| 2400 | 428 | 0.799 | 0.812 |
| 4000 | 382 | 0.810 | 0.783 |
| 8000 | 356 | 0.810 | 0.768 |

**Verdict: keep 1200.** 800 was best (+0.017 tune, +0.015 holdout) but missed the
+0.02 bar, and a gain that small is inside the noise of 35 questions. The clear
finding is on the other side: **very large chunks hurt**, especially on the holdout
half, because the embedding model only reads about 2,000 characters and cuts off the
rest. We only measured search; chunk size also changes how much text the writer sees,
which we did not test.

Run it: `uv run tune-chunks`.

### Prompts (the generator's instructions; calls the API)

We read why the judge marked answers partial or incorrect. Most partial answers
**left out items from a list** in the article (e.g. only some of what a processor
contract must include). Two variants try to fix that:

- **v1 "complete":** include every condition, requirement, exception and time limit
  in the passages, use numbers exactly as given, and don't cap the answer at a few
  sentences.
- **v2 "complete + restrained":** v1, plus "state only what the passages say".

Rule: a variant is adopted only if it beats the current prompt (v0) on **both** halves
by more than the run-to-run noise, and the share of supported claims doesn't fall by
more than 3 points.

| Run | Correct: tune | Correct: holdout | All 70 | Partial | Supported claims | Fixed / broke vs v0 |
|---|---|---|---|---|---|---|
| v0 (current) | 80.0% | 80.0% | 80.0% | 10.0% | 98.8% | – |
| v0 again (**noise**) | 71.4% | 85.7% | 78.6% | 14.3% | 98.9% | 3 / 4 |
| v1 complete | 82.9% | 74.3% | 78.6% | 12.9% | 97.7% | 4 / 5 |
| v2 complete + restrained | 82.9% | 80.0% | 81.4% | 10.0% | 98.9% | 5 / 4 |

"Fixed / broke" counts questions that became correct / stopped being correct.

**The noise is the main finding.** The *same* prompt run twice gave different verdicts
on 7 of 70 questions (10%), and a half of 35 questions moved by up to 9 points.
Against that, v1 and v2 changed things by no more than the noise did. The "be
complete" instructions did **not** measurably reduce partial answers.

**Verdict: keep the current prompt (v0).** The variants stay in the code
(`GENERATOR_VARIANTS`, `uv run ask --prompt v1`) so the experiment can be repeated.

Run it: `uv run python -m compliance_rag.tune_prompts v0 v0 v1 v2` (about 20 minutes
per variant; add `--resume` to reuse finished runs).

### What this means for the rest of the project

- **70 questions can't detect small improvements.** A real 3-5 point gain looks like
  chance. To tune further we would need several hundred questions, not more prompt
  tweaks.
- **The regression gate's tolerance for answer quality may be too tight for the
  20-question jurisdiction set**, where one question is worth 5 points and identical
  runs can differ by that much. A failed gate there should be rerun before trusting it.
- **The remaining partial answers are mostly not a prompt problem.** The judge's reasons
  point at search: the right article was missing or only part of it reached the writer
  (long articles are split), or a recital was needed but articles-only search
  excluded it.

### Two things that went wrong while running this

- A Windows security policy (Application Control) blocked the newly generated
  `tune-prompts.exe` launcher, and my first attempt's error was hidden, so it looked
  like it was running. Running the same code with `python -m` worked.
- A brief network failure killed the second run partway. We added automatic retries
  for the API calls and a `--resume` option that reuses finished runs.

---

## What this project does and doesn't show

It is a working, measured prototype of the pattern
*retrieve → grade → generate → verify → decide*, over **two jurisdictions' versions**
of one regulation, with a Spark ingestion pipeline. **Not built:**

- dozens or hundreds of documents, or dated revisions of the same text (we have two
  related documents, so "200 documents" would be untrue);
- Azure AI Foundry evaluations, Docker, Terraform, Confluence;
- scale: Spark was run on a laptop with a few hundred documents;
- a test set big enough to measure small improvements (70 questions; see Phase 8).

Only quote numbers this project measured: the tables above, not figures from
elsewhere.

---

## 14. Saving our work with Git

Git keeps **save points** (called **commits**) of the project.

- It works **entirely on your computer**. A hidden `.git` folder inside the
  project holds the history.
- **GitHub** is an optional online copy. We can connect it later, and all the
  save points go up with it.
- Some things are deliberately **not** saved (listed in `.gitignore`): the
  `.venv` folder, the downloaded web pages in `data/raw/`, the results in
  `data/eval/`, and `.env` (your secret keys). The cleaned text in `data/processed/`
  **is** saved (about 300 KB of public law), so GitHub can run the checks.

Save points so far:

1. `Phase 1: GDPR ingestion and chunking`
2. `Phase 2: golden set and dense-retrieval baseline`
3. `Add beginner-friendly project guide`
4. `Phase 3: BM25, hybrid RRF and cross-encoder reranking`
5. `Phase 4: generator, fact-checker, Beta confidence and self-correction`
6. `Stop tracking generated eval results`
7. `Phase 5: grader, abstain/review routing, calibrated confidence, judge, gate, CI`
8. `Phase 6-7: UK GDPR as a second version, jurisdiction scoping, PySpark ingestion`
9. `Phase 8: chunk-size and prompt tuning (no change adopted)`

Useful commands:

```powershell
git log --oneline     # list save points
git status            # see what changed since the last save point
```

---

## 15. Glossary

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
| **Tokenizing** | Splitting text into individual words for keyword search |
| **Stopwords** | Very common words ("the", "is") ignored by keyword search |
| **Hybrid search** | Using meaning-search and keyword search together |
| **RRF (Reciprocal Rank Fusion)** | Merging ranked lists using positions, not scores |
| **Shortlist** | The top results from fast search, handed to the slower reranker |
| **Beta distribution** | The math used for the Bayesian confidence score (Phase 4) |
| **Generator** | The AI step that writes the answer |
| **Fact-checker** | The AI step that checks each claim against the text |
| **Claim** | One single fact stated in an answer |
| **Citation** | The chunk ID a claim says it came from |
| **Structured output** | Asking the AI to reply in a fixed format the program can read |
| **Prior / posterior** | The belief before / after seeing the evidence |
| **Credible interval** | The range the true confidence probably lies in (we use 90%) |
| **Self-correction** | Retrying with more text and feedback when a claim fails the check |
| **Faithfulness** | Whether an answer sticks to its sources (not the same as being complete) |
| **Grader** | An AI step that throws away retrieved chunks that don't help answer the question |
| **Abstain** | Refusing to answer because there is no evidence |
| **Escalate / human review** | Passing a shaky answer to a person instead of sending it automatically |
| **Metadata filter** | Searching only chunks with a given tag (here: only articles) |
| **Calibration** | Converting a raw score into an honest probability using known examples |
| **LLM-as-judge** | Using a (bigger) AI to grade another AI's answers |
| **Regression gate** | A check that fails when quality drops below a saved baseline |
| **CI (continuous integration)** | A server that runs your checks automatically on every change |
| **AUC** | How well a score separates two groups: 1.0 perfectly, 0.5 no better than a coin flip |
| **Jurisdiction** | Which legal system a text belongs to (here EU or UK) |
| **UK GDPR** | The UK's own copy of the GDPR, kept in UK law after Brexit and amended since |
| **Recital vs repealed** | A recital is background text; a repealed article has been removed from the law |
| **Amendment markup** | The footnote markers and brackets that law websites add to show what changed |
| **Metadata** | Facts about a piece of text (its jurisdiction, date, version), not its content |
| **Spark / PySpark** | A tool for processing data in parallel, and its Python interface |
| **Local mode** | Running Spark on one computer instead of a cluster |
| **Deduplication** | Removing records whose content repeats |
| **Parity check** | Confirming a new pipeline produces exactly the same output as the old one |
| **JDK** | Java Development Kit, the Java software Spark runs on |
| **Tuning** | Trying different settings and keeping the one that scores best |
| **Holdout set** | Questions kept aside, not used to choose a setting, so they can confirm it |
| **Overfitting** | A setting that looks good only because it was chosen on the same questions it is scored on |
| **Noise / replicate** | Run-to-run variation; a replicate is the same experiment run again to measure it |
| **Decision rule** | A pass/fail test written down before an experiment so the result can't be argued afterwards |
| **Prompt variant** | One version of the instructions given to the AI |
| **Checksum (SHA-256)** | A fingerprint of a file, used to confirm a download is complete and untampered |
| **Golden set** | Questions with known correct answers, used to grade the system |
| **Baseline** | The first simple version, which later versions must beat |
| **Recall@k** | Share of questions where the right answer is in the top k results |
| **MRR** | Average of 1/rank of the first correct result; 1.0 is perfect |
| **Virtual environment (.venv)** | A private set of Python packages for one project |
| **uv** | The tool that installs packages and runs commands |
| **Commit** | A save point in Git |
| **API key** | A secret password that lets the program use a paid AI service |
