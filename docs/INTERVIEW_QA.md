# Interview Q&A: challenges and decisions

Answers are in first person and in simple English. Every number is measured in this repo
(`eval/baseline.json`, `docs/GUIDE.md`). Use only what is true for you. Items marked
**(not built)** are not in this repo; answer them from your own experience or as a plan.

---

## Part 1: The project in general

**Q1. Tell me about this project.**
I built a question-answering system over the EU GDPR and the UK GDPR. In law, a confident
wrong answer is worse than no answer, so the system checks itself. One agent filters the
retrieved text, one writes an answer with citations, and one checks every claim against the
sources. A confidence score then decides: answer automatically, send to a human, or refuse.
I also built an evaluation setup so I could measure what works and what doesn't.

**Q2. Why not just use ChatGPT?**
Three reasons. It can't show which article an answer came from. It can state the law wrongly
with full confidence. And it doesn't reliably say "I don't know". In my system every claim
points to a source chunk, a second step checks it, and unanswerable questions get refused.

**Q3. What does "agentic" mean here?**
It means several steps, each with one job, that check each other. The Grader decides which
retrieved chunks are relevant. The Generator writes claims with citations. The Verifier labels
each claim as supported or not. If the grader rejects everything or a claim is unsupported,
the system retries once with a bigger pool of chunks.

**Q4. What are the headline results?**
- Search: the right article is first for 81% of 50 EU questions (56% with meaning search alone on the same corpus).
- Answers: 97% of claims were supported by the sources, and 94% cited the labelled article.
- Correctness, scored by an LLM judge: 76% on the EU questions, 90% on the 20 jurisdiction
  questions.
- Safety: on 12 questions the documents can't answer, 10 were refused, 2 went to a human, and
  0 were answered automatically.

---

## Part 2: Data challenges

**Q5. What was hard about getting the data?**
The EU GDPR was a saved web page. The UK GDPR page was messier. It had hidden tags for
amendments and commentary, repealed articles that show only as "...", and some articles with
no title because they used a different heading style. There was also a label like "12A.".
I wrote a separate UK parser and tests for these cases. 38 repealed articles are skipped on
purpose.

**Q6. Why did you add a second version of the law?**
The resume talks about document versions and dates. One document can't show that. The UK
GDPR is a related but different version of the same law, so the same questions and ideas
carry over. Each chunk now carries jurisdiction, effective date and version. A UK question
searches only UK text. (I'd considered a 200-page document from another source, but a second
version of the same law fits better.)

**Q7. How do you handle the same text appearing in both?**
5 articles are word-for-word identical in the EU and UK versions. I did not delete either,
because they belong to different laws. I link them with a `same_text_as` field. Dedup runs
within each jurisdiction and found 0 duplicates in this data.

**Q8. Why did you use PySpark for two documents?**
Honestly, for the design, not for the size. The stages (clean, validate metadata, dedupe,
tag, chunk) are written so they scale to many documents. Locally it's no faster. I checked
that its output is identical to the plain-Python pipeline, and that check runs on every
Spark ingestion.

---

## Part 3: Search challenges

**Q9. How did you improve search quality?**
In three steps. Meaning-based search alone put the right article first 56% of the time (36% on the
original corpus with recitals included).
Adding keyword search (BM25) and merging the two lists (Reciprocal Rank Fusion) helped,
because legal text has exact terms that meaning search can blur. A reranker then re-scored
the top results with the question and each chunk read together. That took it to 81%.

**Q10. What are search scores, and why calibrate them?**
The reranker outputs raw scores that aren't probabilities. I used Platt scaling to turn them
into probabilities, so they can be combined with other evidence in the confidence score.

**Q11. What still goes wrong in search?**
- UK Article 91A (a generic "Secretary of State regulations" article) crowds UK results. I
  left it visible instead of hiding it, and wrote it down as a known weakness.
- Long articles are split into chunks, so sometimes only part reaches the writer.
- The "articles only" filter excludes some recitals that would answer a question.
- Abstract questions like "are the legal bases the same?" match nothing well, so the system
  refuses, and the judge marks that wrong.

**Q12. What is metadata filtering and why does it matter?**
Each chunk has a type (article or recital) and a jurisdiction. The filter removes
non-matching chunks before ranking, using a mask on one shared index. This is how a
question about UK rules never gets EU text. When a question names neither, I assume EU and
the answer says so.

---

## Part 4: Confidence and safety

**Q13. How does the Bayesian confidence work?**
It counts evidence. Each supported claim is a success, each unsupported claim is a failure.
Grader agreement on the top 3 chunks and the calibrated retrieval strength add more counts.
Those counts form a Beta distribution, which gives an average and a range. The rule: answer
automatically only if the average is at least 0.80 and the low end of the 90% range is at
least 0.55. Otherwise send to review, or refuse.

**Q14. Why Beta and not just ask the model how sure it is?**
A model's own "90% sure" isn't tied to evidence. The Beta score is built from counted
evidence and gives a range, so little evidence means a wide, cautious range.

**Q15. Does the confidence predict correct answers?**
No, and I measured it. It separates answerable from unanswerable questions almost perfectly
(AUC close to 1.0), so it works for deciding when to refuse. But for ranking right against
wrong answers it scored AUC 0.41 on 50 questions, which is no better than chance. I report
that instead of hiding it. A next step would be adding features that track correctness, such
as answer completeness.

**Q16. Why is refusing important?**
In compliance, a wrong confident answer can cause real harm. Out of 12 questions the
documents can't answer, none was answered automatically. That is the most important safety
result in the project.

---

## Part 5: Evaluation challenges

**Q17. How do you evaluate it?**
There are 82 questions: 50 EU, 20 jurisdiction (10 UK-only, 10 comparisons) and 12
unanswerable. Search is scored by Recall@k and MRR against labelled articles. Answers are
scored on whether the claims are supported, whether they cite the labelled article, and
correctness judged by a stronger model (`gpt-5.4`) against the gold passages.

**Q18. Is an LLM judge reliable?**
Not perfectly. I measured the noise: running the same prompt twice flipped the verdict on 7
of 70 questions (about 10%). So I treat the judge as a rough tool, not exact truth, and I
don't claim gains smaller than that noise.

**Q19. What is the regression gate?**
A check that compares current scores to a saved baseline, with a tolerance for each section.
If a change makes something worse than the tolerance, it fails. The offline retrieval part
runs in CI on every push. The answer part costs API money, so it is a manual job.

**Q20. What are the weaknesses of your evaluation?**
It's small: 82 questions. The golden set favours articles, which flatters the article filter.
The gate's tolerance may be too tight for the 20-question jurisdiction set. And the judge
has its own noise. To detect small improvements I'd need several hundred questions.

---

## Part 6: Tuning

**Q21. What did you tune?**
Chunk size (from 400 to 8000 characters) and the generator prompt (three variants).

**Q22. How did you avoid fooling yourself?**
I split the 70 answerable questions into a tuning half and a held-out half. I wrote the
decision rules before running (for example, adopt a chunk size only with at least +0.02
MRR). And I measured noise by running the same prompt twice.

**Q23. What was the result?**
No change adopted. 800-character chunks scored slightly better (+0.017 on the tuning half,
+0.015 on the holdout), below my 0.02 bar. Chunks above about 2,000 characters clearly hurt.
The "complete" prompts didn't beat the noise. I count this as a good result: I proved the
current settings are fine and that more prompt tweaking would chase noise.

---

## Part 7: Environment and engineering challenges

**Q24. What went wrong technically, and how did you fix it?**
- **Java for Spark:** the normal installer needed an admin click and got cancelled. I used a
  portable JDK. The first download was truncated; a checksum caught it, and I resumed.
- **A Windows security policy blocked a new launcher file.** My command hid the error, so a
  job looked like it was running when it wasn't. Now I run modules directly and never hide
  errors on background jobs.
- **A network outage killed a 20-minute run.** I added 8 API retries and a `--resume` option.
- **A file lock** stopped a rebuild while another job was running. I waited for it.
- **Garbled quotes** came from editing a file through PowerShell. I switched to proper
  editing for files.
- **GitHub rejected password login.** The sign-in had to happen from my own terminal, then
  the push worked.

**Q25. What did you learn about time estimates?**
Mine were too low. One pass of 70 questions takes about 20 minutes because each question
makes several API calls. After that I used real timings and ran long jobs in the background.

**Q26. How did you make the project easy to continue after a break?**
My first session was lost, so I keep `docs/GUIDE.md` for beginners, a README with a roadmap
and a status note. Everything is committed, and CI runs on GitHub (lint, tests and the
retrieval gate pass).

---

## Part 8: Tools not built here (Docker, Terraform, Foundry, Confluence)

**Q27. How would you containerise it? (not built)**
A Dockerfile with Python 3.12, the locked dependencies, the model files and Java for Spark.
CI would build the image so every machine and server runs the same environment.

**Q28. Where does Terraform fit? (not built)**
It would define the cloud setup as code: storage for documents, a place to run the app, and
the model endpoint. I'd run `plan` to preview changes, review them in git, and `apply` to
create them, so environments can be rebuilt or deleted easily.

**Q29. Why Azure AI Foundry? (not built; the code has an untested Azure option)**
In an enterprise, models and data often must stay in the company's cloud, with access
control and logging. Foundry hosts the models and can run evaluations there. It would
replace my OpenAI judge, and the same regression gate would run against it.

**Q30. How would you add Confluence? (not built)**
Call the Confluence API for pages, strip the markup, and feed the text into the same
cleaning and chunking steps. Keep page ID, last-updated date and version as metadata, the
same way I use effective date and version for the GDPR. Then re-ingest when a page changes.

---

## Part 9: Judgment questions

**Q31. What would you do next?**
1. Grow the question set to several hundred, so small gains become measurable.
2. Fix the known search misses (split articles, recitals, the Article 91A crowding).
3. Add features so confidence also tracks correctness.
4. Add Docker, then infrastructure code and a managed model platform.

**Q32. What are you most proud of?**
Not the accuracy. It's that I could say which numbers I trust and which I don't. I measured
the judge's noise, kept test and tuning data separate, and reported a negative result.

**Q33. What is the biggest limitation?**
Scale and evidence. It's two related documents and 82 questions. It is a proof of concept
for the design and the evaluation method, not a production system.

---

## Part 10: Your resume, line by line

First, decide which story is true for you, because it changes every answer below:

- **Story A: the resume describes real work at a company.** Then the 200 documents, the
  40% and the 20%→60% come from that work. Answer from that experience. This repo is your
  personal rebuild of the same ideas, on public law, so you can show the design and the
  evaluation method. Say so openly: "The work project was confidential, so I rebuilt the
  approach on the GDPR."
- **Story B: the resume describes only this project.** Then use the measured numbers below.
  The 200 documents, 40% and 20%→60% are not backed by anything here and must go.

Never mix the two. Do not give this repo's numbers as the work project's, or the reverse.

### Bullet 1: "enterprise POC ... 200 regulated documents ... conflicting versions and unanswerable questions"
- **Backed by this repo:** the problem. Two versions of one law (EU and UK) that conflict in
  places, and 12 unanswerable questions.
- **Not backed:** the 200 documents. This repo has 2.
- **Likely question:** "What did the basic upload-and-chat approach get wrong?"
- **Answer:** "It answers from whatever chunks it gets, with no check on the source or the
  version. With two versions of one law it mixes them, and with a question the documents
  can't answer it still produces something. I fixed both: jurisdiction scoping, and a
  verifier plus refusal. On 12 unanswerable questions, none was answered automatically."

### Bullet 2: "Grader, Generator, Verifier ... abstains or escalates to a human"
- **Fully backed.** See Q3, Q4, Q16.
- **Likely question:** "What does 'escalate to a human' actually do?"
- **Answer:** "The answer is routed to a `review` queue with its evidence. Two of 12
  unanswerable questions went there. It isn't a full review tool. It's the routing decision
  and the evidence a reviewer would see."

### Bullet 3: "Bayesian confidence ... auto-answer versus human-review threshold"
- **Fully backed.** See Q13–Q15.
- **Likely question:** "How did you choose 0.80 and 0.55?"
- **Answer:** "I picked them to be cautious and checked them on the 82 questions. They kept
  unanswerable questions from being auto-answered. I didn't tune them heavily, and the score
  doesn't predict correctness (AUC 0.41), so I'd call it a refusal control, not a quality
  predictor."

### Bullet 4: "PySpark ... clean, deduplicate, chunk, tag ... reduced irrelevant context by 40%"
- **Backed:** the pipeline, the metadata (type, effective date, version, jurisdiction) and
  metadata-filtered retrieval.
- **Not backed:** the "40%". It was never measured here.
- **Honest way to talk about this repo:** "Filtering to articles and to the right
  jurisdiction raised first-answer-correct on the 50 EU questions from 0.61 (hybrid, no
  rerank) to 0.81 with the reranker. I didn't measure 'irrelevant context' as a percentage."
  Use the 40% only under Story A, with its own definition ready: "irrelevant context =
  chunks the grader rejects, per question."
- **Likely question:** "How did you measure 40%?" Have that answer ready, or drop it.

### Bullet 5: "50-question golden set ... recall@k and MRR ... groundedness, faithfulness, correctness ... 20% to approximately 60%"
- **Backed:** the 50-question set, Recall@k, MRR, supported-claims (groundedness/faithfulness)
  and judged correctness. Tuning of chunking, hybrid search and reranking.
- **This repo's own improvement, same corpus (articles only, EU + UK):** R@1 0.56 (dense
  only) → 0.81 (hybrid + rerank); MRR 0.731 → 0.878. On the original corpus with recitals
  included it was smaller: R@1 0.36 → 0.47, MRR 0.545 → 0.670. Never quote 0.36 → 0.81,
  because that compares two different corpora. Part of the gain comes from the
  articles-only filter, and the golden set favours articles.
- **Not backed:** "20% to 60%". If it's from real work, own it as that. If not, use the
  same-corpus numbers above.
- **Likely question:** "Define groundedness vs faithfulness vs correctness."
- **Answer:** "Groundedness: is each claim supported by the retrieved text? Faithfulness: does
  the answer say only what the sources say, with nothing added? Correctness: does it match
  the gold answer? In this repo, the verifier covers the first two (97% of claims
  supported) and the judge covers correctness (76% on EU, 90% on jurisdiction)."
- **Likely question:** "Your tuning improved things?" **Answer:** "Search improved a lot.
  The final chunk-size and prompt tuning did not beat the noise, so I changed nothing and
  I can show why."

### Bullet 6: "Azure AI Foundry evaluations ... LLM-as-judge ... human spot-checks ... regression gate"
- **Backed:** LLM-as-judge and the regression gate.
- **Not backed here:** Foundry (the judge is OpenAI `gpt-5.4`), and human spot-checks.
- **Likely question:** "How does Foundry fit, and who did the spot-checks?"
- **Answer under Story A:** your real Foundry setup. Under Story B: "I used an OpenAI judge
  here. Foundry would host the same judge inside the company's cloud, and the gate works
  the same way. I read the judge's written reasons for the failures myself, which is how I
  found the incomplete-list and search-miss patterns, but I haven't run a formal spot-check
  sample."
- Do not claim "automated Foundry evaluations" for this repo.

### Bullet 7: "GitHub Actions CI/CD ... containerized deployment, Terraform, Confluence"
- **Backed:** GitHub Actions with tests, lint and the retrieval gate (passing).
- **Not built:** containerized deployment, Terraform, Confluence.
- **Answer under Story A:** your real experience of each, one example each.
- **Answer under Story B:** drop "containerized deployment, Terraform, Confluence" from the
  bullet, or say "designed" and be ready to walk through Q27–Q30.

### Safer wording of the bullets for this repo (Story B)
1. Built a Grader/Generator/Verifier RAG pipeline over the EU and UK GDPR that abstains or
   routes to human review when evidence is weak; auto-answered 0 of 12 unanswerable questions.
2. Built Beta-posterior confidence from retrieval relevance, grader agreement and verifier
   support to set the auto-answer vs review threshold; showed it separates answerable from
   unanswerable questions but not correct from incorrect ones.
3. Built a PySpark ingestion pipeline (clean, dedupe, chunk, tag with type, jurisdiction,
   effective date and version) and metadata-filtered retrieval.
4. Created an 82-question golden set; raised first-answer-correct from 0.56 to 0.81 (same corpus) with
   hybrid search and reranking; 97% of claims supported, 76–90% judged correct.
5. Added an LLM-judge regression gate and GitHub Actions CI; ran holdout-based tuning with a
   noise measurement and reported a negative result.

### Five questions to practise out loud
1. "Walk me through one question from input to answer." (Scope → search → grader →
   generator → verifier → confidence → route. Use a real example from `uv run ask`.)
2. "Where does your system fail?" (Q11, plus incomplete lists.)
3. "Which numbers on your resume would you not stand behind?" (Have an honest answer.)
4. "What would you change if you had a week?" (Q31.)
5. "What did you learn?" (Q32.)

## Do not say

- "200 documents", "40% less irrelevant context", "20% to 60%": not measured here.
- "Deployed with Docker, Terraform or Foundry": not built in this repo.
- "Confidence predicts correctness": measured, it doesn't.
- "State of the art": this is a small, honest proof of concept.
