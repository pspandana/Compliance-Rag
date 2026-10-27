# Explaining this project

Ready-made answers for "what did you build, and why did you choose X?". Every answer says
whether the thing is **built here**, **written but not run**, or **not built**. Say only
what is true for you. All numbers are measured in this repo (see `eval/baseline.json`).

## The 30-second version

"I built a question-answering system over regulatory text, the EU GDPR and the UK GDPR.
Law is a bad place for confident wrong answers, so the system checks itself. One agent
filters the retrieved text, one writes an answer with citations, and one checks each claim
against the sources. A confidence score then decides whether to answer, send to a human,
or refuse. On 12 questions the documents can't answer, it answered none automatically.
I also measured where it fails, which is more useful than a single accuracy number."

## Why each piece exists

| Piece | Status | Why we used it (simple English) |
|---|---|---|
| Hybrid search + reranker | Built | Keyword search finds exact legal terms; meaning search finds paraphrases. Together they beat either alone, and the reranker re-orders the top results. First answer correct went from 0.47 to 0.81. |
| Grader / Generator / Verifier | Built | Splitting the job makes each step checkable. The grader drops irrelevant text, the generator cites sources, and the verifier catches unsupported claims. 97% of claims came back supported. |
| Beta (Bayesian) confidence | Built | A yes/no from the model is not enough. The score combines the verifier, the grader and search strength, and gives a range. Auto-answer only if the average is high and the low end of the range is still safe. |
| Abstain / human review | Built | In compliance a refusal beats a wrong answer. 10 of 12 unanswerable questions were refused and 2 went to review. |
| Metadata filters | Built | Each chunk carries jurisdiction, date and version, so a UK question searches only UK text. |
| PySpark ingestion | Built (local mode) | Cleaning, dedup and tagging as pipeline stages that scale to many documents. Here it's 2 documents, so it's about the design, not the size. Its output matches the plain-Python version. |
| Golden set + LLM judge | Built | 82 questions with known answers, so changes are measured and not guessed. |
| Regression gate | Built | A change that makes scores worse fails the check. CI runs the offline part. |
| GitHub Actions CI | Built, passing | Lint, tests and the retrieval gate run on every push. The manual answer-gate job hasn't been run. |
| Docker | Not built | Packs Python, the packages and Java for Spark into one image. Then CI, a teammate and a server all run the exact same setup. |
| Terraform | Not built | Describes the cloud setup as code: storage for documents, a place to run the app, the model endpoint. It can be previewed (`plan`), reviewed in git and rebuilt or deleted on demand. |
| Azure AI Foundry | Not built (Azure option in code, untested) | The enterprise route: models and data stay in the company's cloud, with access control, and evaluators can run there. It would replace the OpenAI judge. |
| Confluence ingestion | Not built | Real policies live in wikis. It would pull pages through the API into the same cleaning and chunking steps, keeping page ID, last-updated date and version as metadata. |

## Likely questions, with honest answers

**Why not just ask ChatGPT?**
It can't tell you which article it used, it can't refuse reliably, and it can quote the law
wrongly with confidence. Here every claim points to a source chunk and a second step checks
it.

**Why Bayesian confidence instead of the model saying "I'm 90% sure"?**
A model's stated confidence is not tied to evidence. The Beta score counts evidence (claims
supported or not, graders that agree, retrieval strength) and gives a range, so a few
data points give a wide, cautious range.

**Does the confidence predict correct answers?**
No, and I measured it. It separates answerable from unanswerable questions very well, but
it does not predict whether an answer is right (AUC 0.41 on 50 questions, which is no
better than chance). So it is useful for deciding when to refuse, not for ranking answers.

**What fails?**
Incomplete answers when a long article is split and only part reaches the writer; articles
the search misses; recitals excluded by the "articles only" filter; and one generic UK
article that crowds UK results. About 76% of EU answers were judged correct, 90% on the
jurisdiction set.

**Did tuning help?**
No, and I reported that. Re-running the same prompt flipped 7 of 70 verdicts, so 70
questions can't show gains of a few points. I set the decision rules before running, tested
on a held-out half, and kept the current settings.

**How big is it?**
Small on purpose: two related documents (99 + 82 articles, 173 recitals) and 82 test
questions. It is a proof of concept for the design and the evaluation, not a production
system.

**Why PySpark for two documents?**
To show the pipeline design (clean, validate, dedupe, tag, chunk) in a form that scales.
Locally it adds no speed. I checked that its output matches the plain-Python pipeline.

**How would you take it to production?**
1. Docker image, so it runs the same everywhere.
2. Terraform for storage, compute and the model endpoint, so environments can be rebuilt.
3. Move the judge and the models to Azure AI Foundry for data residency, with the same
   regression gate in CI.
4. Add Confluence as a source, with version and date metadata.
5. Grow the golden set to several hundred questions so small changes become measurable.

## Do not say

- "200 documents", "40% less irrelevant context" or "20% to 60%": not measured here.
- "Deployed with Docker / Terraform / Foundry": none of it is built in this repo.
- "Confidence predicts correctness": measured, it doesn't.
