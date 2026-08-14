Vanna AI POC (~2-3 days)
------------------------

**What you're testing:** does retrieval-augmented SQL generation (schema + example Q→SQL pairs) beat the current bare chain on accuracy and consistency, cheaply enough to be worth adopting.

1.  Standalone script, not wired into Streamlit (poc/vanna\_poc.py or a notebook).
    
2.  Train it on KATHA's schema (DDL or the existing table\_desc\_katha.csv content) plus ~10-15 hand-written example question→SQL pairs covering the same question types as your test set.
    
3.  Run the shared test questions through it. Check two things separately:
    
    *   **Accuracy** — is the SQL valid and semantically correct?
        
    *   **Consistency** — ask the same question 3 times; does it generate the same SQL each time?
        
4.  Compare both numbers against the baseline chain.
    

**Go/no-go:** if it clearly beats baseline on accuracy and consistency without much added complexity → worth swapping into the real retrieval layer later. If it's marginal → not worth the new dependency (chromadb wiring, ongoing example-pair maintenance).

Wren AI POC (~2-3 days, heavier)
--------------------------------

**What you're testing:** not accuracy — determinism. This is aimed squarely at the "same question, same numbers" problem that neither the current chain nor Vanna actually solves.

1.  Stand up Wren AI locally via its docker-compose (fastest path, no need to touch your app).
    
2.  Model a small MDL semantic layer over 2-3 KATHA tables — define a couple of metrics/dimensions analogous to what MItra will need (a count grouped by category over time, a percentage breakdown) rather than raw table access.
    
3.  Run the same shared test questions, but specifically re-ask each one 2-3 times / rephrased, and check whether the returned numbers are literally identical each time — that's the real signal, not whether the SQL "looks right."
    
4.  Separately note how much MDL modeling effort 2-3 tables took, as a proxy for how much work the real 4 MItra sheets would need.
    

**Go/no-go:** if it measurably improves consistency, weigh that against the modeling + infra cost — that's likely a "phase 2, after the 15 days" decision rather than something to fold into the current plan. If the consistency gain doesn't justify running a second service, a cheaper fallback worth keeping in mind is just writing a small set of fixed/parameterized SQL templates for your most common question types instead of chasing a full semantic layer.

How to execute practically
--------------------------

*   Keep both under a throwaway poc/ folder (or entirely outside the repo) — not on v2-makeover, so they can't accidentally block or get tangled with the real build.
    
*   Time-box each to a hard 2-3 day stop.
    
*   Write a short decision-record (a few paragraphs: what you tested, the numbers, go/no-go) for each — that's the artifact worth keeping around afterward, not the throwaway POC code itself.