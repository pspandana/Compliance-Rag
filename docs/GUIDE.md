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
| 5 | Run tests automatically on GitHub whenever code changes | Next |
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
│   ├── retrieval.py        ← Phases 2-3: the search methods
│   ├── evaluate.py         ← Phases 2-3: scores the search
│   ├── llm.py              ← Phase 4: connects to OpenAI
│   ├── answer.py           ← Phase 4: writes, checks and retries answers
│   └── confidence.py       ← Phase 4: the Bayesian confidence score
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

### Testing it on all 50 questions

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

## 10. Phase 5 (next): Automatic checks on GitHub

Every time code is pushed to GitHub, GitHub Actions will automatically run the
tests and the search evaluation (no API key needed for those). This catches
mistakes before they pile up. It needs the project on GitHub first.

---

## 11. Saving our work with Git

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
3. `Add beginner-friendly project guide`
4. `Phase 3: BM25, hybrid RRF and cross-encoder reranking`
5. `Phase 4: generator, fact-checker, Beta confidence and self-correction`

Useful commands:

```powershell
git log --oneline     # list save points
git status            # see what changed since the last save point
```

---

## 12. Glossary

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
| **Golden set** | Questions with known correct answers, used to grade the system |
| **Baseline** | The first simple version, which later versions must beat |
| **Recall@k** | Share of questions where the right answer is in the top k results |
| **MRR** | Average of 1/rank of the first correct result; 1.0 is perfect |
| **Virtual environment (.venv)** | A private set of Python packages for one project |
| **uv** | The tool that installs packages and runs commands |
| **Commit** | A save point in Git |
| **API key** | A secret password that lets the program use a paid AI service |
