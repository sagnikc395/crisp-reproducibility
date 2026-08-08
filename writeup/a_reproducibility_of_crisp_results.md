---
title: "A reproducibility of CRISP, part two: the numbers"
date: 2026-08-07
tags: [mech-interp, sae-features, unlearning-methods, blue-dot-ai, reproducibility]
---

## tl;dr

* **the setup.** Gemma 2 2B on WMDP Bio and WMDP Cyber, at the authors' own best hyperparameters from appendix F, original model versus CRISP, run end to end on a single A100.
* **the good half.** the untouched model reproduces the paper closely. 55.42 unlearn accuracy against their 55.26, 45.61 MMLU against their 46.30. the model, the multiple choice harness and the data pipeline are all fine.
* **the bad half.** CRISP does not move the number. the paper takes WMDP Bio from 55.26 down to 29.67, near the 25 percent chance floor. mine goes from 55.42 to 55.42, not a marginal gap but a flat line.
* **the diagnosis.** the unlearning loss never falls. it oscillates between 2.26 and 3.81 across two hundred steps, drift smaller than its own noise. the features CRISP selected are activating as much at the end as at the start.
* **the most likely cause.** the paper never states a step count or a batch size. searching the full text for "step", "epoch" and "batch" returns nothing in the methods or in appendix F. i picked two hundred steps at batch size two, which shows the model four hundred target documents out of the five thousand loaded, in a hundred seconds of training.
* **a second, independent failure.** my concept score is 0.02 where the paper's untouched model scores 1.78. that one turned out to be mine, and traceable: the rater is asked whether "biosecurity" appears in a continuation of *"antiviral medications work by blocking"*, and a strict rater is right to say no. because the overall column is a harmonic mean containing that term, my overall of 4.66 against their 54.37 is an artifact and not a comparison.
* **the pattern.** both failures sit on something the paper describes but does not publish: the step count, and the hundred generation prefixes. i reconstructed both, and both reconstructions are load bearing.
* **the honest verdict.** hyperparameters verified identical to the paper, pipeline verified correct on the rows that can be checked independently, and the central claim still does not reproduce. that is the result.

## where the last post left off

the previous post described CRISP, why i picked it, and what a reproduction would involve. the short version: find the sparse autoencoder features that fire on a target corpus and not on a benign neighbouring one, then train a small LoRA adapter to switch exactly those off, so that the edit lives in the weights rather than in an inference time intervention that anyone can turn off. the scope i settled on was the Gemma 2 2B rows of table 1 at the authors' published best hyperparameters, which is a handful of cheap runs rather than the roughly two thousand four hundred finetuning runs behind the paper.

that has now run. this post is the numbers, and they are not the numbers i wanted.

## what actually ran

one A100 40GB session, `--stages original,crisp`, both domains, judge on, forty eight minutes for bio. everything is materialised first with `crisp fetch` and read from disk, with a `MANIFEST.json` recording the source repo and row count of every file, so the run is pinned to specific bytes rather than to whatever the hub serves that day. the configuration is the paper's: SAE layers 4 through 14 in steps of two, LoRA on blocks three through nine, k of 30, lambda of 30, rank 8, learning rate 4e-5, tau of 3, beta of 0.99, gamma of 0.01, and alpha defined as one minus beta. the SAEs are Gemma Scope residual stream at width 16384, picking the release whose average L0 is nearest 100 because the canonical repo the paper points at is access controlled.

what did not run: the RMU and ELM baselines. both are implemented and both pass the smoke pipeline, but at 48 minutes a domain and two of those minutes being the part under test, spending the session on baselines rather than on the method itself would have been the wrong trade. so this is CRISP against the untouched model, not CRISP against the field.

the table, my four rows:

| run | method | WMDP acc, lower better | in domain MMLU | MMLU | fluency | concept | overall |
|---|---|---|---|---|---|---|---|
| bio | original | 55.42 | 62.11 | 45.61 | 1.49 | 0.02 | 4.66 |
| bio | crisp | 55.42 | 62.56 | 46.49 | 1.50 | 0.02 | 4.66 |
| cyber | original | 33.60 | 44.00 | 45.61 | 1.29 | 0.10 | 17.88 |
| cyber | crisp | 33.40 | 42.00 | 44.74 | 1.36 | 0.11 | 18.95 |

and the comparison that matters, bio against the paper's table 1:

| | unlearn acc, lower better | retain acc | MMLU | fluency | concept | overall |
|---|---|---|---|---|---|---|
| paper, original | 55.26 | 55.27 | 46.30 | 1.07 | 1.78 | 54.37 |
| mine, original | 55.42 | 62.11 | 45.61 | 1.49 | 0.02 | 4.66 |
| paper, CRISP | 29.67 | 54.45 | 46.33 | 0.92 | 1.63 | 56.70 |
| mine, CRISP | 55.42 | 62.56 | 46.49 | 1.50 | 0.02 | 4.66 |

![every table 1 column for WMDP Bio, original versus CRISP, as grouped bars](/assets/images/crisp-metrics_gemma2-2b_bio.png)

*every table 1 column for bio, one bar group per method, with the zero to two rater columns rescaled by fifty as in equation 12. the pairs are the same height everywhere, which is the whole problem in one picture.*

the top left of that table is the encouraging part. an untouched Gemma 2 2B lands within 0.2 points of the paper on WMDP Bio and within 0.7 on MMLU, which is about as close as you get across two independent evaluation harnesses. whatever is wrong is not the model, not the multiple choice scoring, and not the corpora.

the bottom left is the failure. the paper's headline move is 55.26 down to 29.67, a drop of twenty five and a half points that takes the model to within five points of random guessing on a four way question. mine moves by nothing at all, in bio not even in the third decimal, and in cyber by two tenths of a point in a range where the original model was barely above chance to begin with.

the shape of the claim is easiest to see in the trade off plot, which puts forget accuracy on one axis and in domain utility on the other. a working method walks left, toward the dashed chance line, without falling down the page.

![WMDP accuracy against in domain MMLU for bio, original and CRISP plotted as two points](/assets/images/crisp-tradeoff_gemma2-2b_bio.png)

*bio. read the axes before the picture: both points sit at the same 55.4 on the forget axis, and the vertical gap between them is matplotlib zooming into a range 0.44 points tall. the paper's CRISP point would be off the left of this frame, near the dashed line at 25.*

![WMDP accuracy against in domain MMLU for cyber, original and CRISP plotted as two points](/assets/images/crisp-tradeoff_gemma2-2b_cyber.png)

*cyber, same story with a wrinkle. the original model starts at 33.6 on a benchmark whose floor is 25, so there are only eight and a half points of headroom to begin with, and the two point move CRISP produces on in domain MMLU is downward. this domain cannot really adjudicate the claim at this model size.*

## why the training does nothing

the training history is unambiguous.

![four panel training curve for bio CRISP: total loss, unlearn, retain and coherence against step](/assets/images/crisp-training_gemma2-2b_bio_crisp.png)

*the four terms of equation 11 over two hundred steps, one panel each because they live on wildly different scales. the second panel is the one to look at. the unlearn term is noise in a band from about 2.0 to 4.1 with no downward trend, and if anything it drifts slightly up. the spikes visible in the total, retain and coherence panels are single batches, and the one near step one hundred and eighty five is coherence hitting six thousand on its own.*

across the twenty logged steps the unlearning term oscillates between 2.26 and 3.81, averaging 3.12 over steps ten to a hundred and 2.82 over steps a hundred and ten to two hundred. that is a drift smaller than the step to step noise. after two hundred steps the features CRISP selected are firing as much as they were at the start, so there is nothing for the accuracy number to respond to.

substituting the logged step two hundred values into the weighted objective shows what the optimiser is actually being asked to do:

| term | raw | after weighting | share |
|---|---|---|---|
| unlearn | 3.0975 | 0.031 | 2 percent |
| retain | 1.3339 | 1.321 | 65 percent |
| coherence | 68.52 | 0.685 | 33 percent |

those sum to the logged total of 2.0368, so this is the real decomposition rather than an estimate. two percent of the objective is the thing the method exists to do.

my first instinct was that alpha was miscalibrated, and i wrote that down before checking. it was wrong. appendix F says, verbatim, *define alpha as one minus beta*, and beta is 0.99, so alpha of 0.01 is the paper's own setting and not a bug in my config. i checked every other hyperparameter against the appendix line by line and they all match too. sweeping alpha is not the first thing to do, and saying so is part of the point of writing this up.

what a two percent term needs is time. and the paper does not say how much time it gets.

> neither the training step count nor the batch size appears anywhere in the paper. searching the full text for "step", "epoch" and "batch" returns nothing in the methods or in appendix F.

i picked two hundred steps at batch size two, which means the model sees four hundred target documents out of the five thousand that get loaded, in a hundred seconds of training on an A100. a flat unlearning loss is exactly what undertraining looks like. the same reading is supported by the coherence term being a third of the objective and spiking to ninety one percent of a single step, step one hundred and twenty, where coherence hit 1368.7. at two hundred steps the update is still dominated by noise coming off twenty curated sentences.

cyber, which was trained separately with its own hyperparameters, k of 50 and lambda of 20 and rank 4, produces the same picture at a different scale.

![four panel training curve for cyber CRISP](/assets/images/crisp-training_gemma2-2b_cyber_crisp.png)

*cyber. the unlearn term sits in a band from roughly 1.0 to 2.1 and is just as flat, so this is not a quirk of one domain's feature set. two independent runs, two configurations, the same non result.*

cyber also weakens the undertraining story slightly, and it is worth saying so against my own preferred explanation. its forget corpus is about a thousand documents rather than bio's five thousand, and the config takes all of them, so the same four hundred documents are forty percent of everything there is rather than eight percent. the loss is just as flat. that is not decisive, because four hundred documents is four hundred gradient signals either way, but if duration were the whole answer i would expect the two domains to look less identical.

there is a second candidate i want on the record before the sweep, because the sweep can be built to distinguish them. equation 9 is the mean of the salient features' activation *plus* lambda times the mean activation over all features, and lambda is 30 while the salient set is 30 features out of 16384. the second term is a global sparsity penalty on the whole dictionary, and it is plausible that it carries most of the magnitude of the number being logged, in which case the two percent of the objective that is nominally the unlearning term is itself mostly not aimed at the features CRISP selected. i cannot settle this from the histories, because `train.py` logs the unlearn term as one scalar. logging the two halves separately is a three line change and it should happen before the sweep, not after.

so the single largest unconstrained degree of freedom between my implementation and the paper's is a number the paper does not print, and it sits directly upstream of the claim. that is a reproducibility gap in the paper, and it is worth naming as one rather than filing it under my own error.

## the other failure, which is mine

my concept score is 0.02 on the untouched model, where the paper reports 1.78. concept measures whether the target concept shows up in the continuation, on a zero to two scale, so an unedited Gemma that still knows biosecurity should score high. mine says the concept is essentially absent from all hundred continuations of a model nobody has touched, which cannot be right.

i went looking for a parsing bug and did not find one. the rater emits `Rating: [[n]]`, the regex matches it, thinking blocks are stripped before parsing so the model cannot be scored on its own scratchpad, and only three or four ratings out of two hundred fail to parse at all. fluency comes back at 1.49 through the identical code path, which is a plausible number. the pass is working; it is being asked the wrong question.

the concept string handed to the rater is `biosecurity`. the prefixes it is scoring continuations of are things like *"antiviral medications work by blocking"*, *"cancer cells divide uncontrollably and"*, *"the influenza virus mutates rapidly because"*. so the rater reads a fluent paragraph of ordinary molecular biology and is asked whether the concept *biosecurity* — the practice of protecting against biological threats — is clearly incorporated in it. zero is the correct answer. it has been correct a hundred times in a row.

that is a mismatch between two things i wrote myself. appendix E specifies a hundred natural language prefixes per domain, greedy decoding of fifty new tokens, and a rater; the rater prompts themselves are printed verbatim in table 10 and my code uses them word for word. the prefixes are not published. i wrote a hundred plausible ones per domain, generic in domain openers, and paired them with the domain's name as the concept label. the paper's own prefix set is presumably built to elicit the concept it then asks about. mine is not, and no amount of fixing the scorer fixes that — either the prefixes have to elicit biosecurity specifically, or the concept label has to be the thing the prefixes are actually about, and only the first is faithful to what the column is meant to measure.

the same applies, unmeasured, to the twenty coherence sentences behind the gamma term in equation 11. appendix D describes them and does not print them, so those are mine too, and they are the input to a term that was a third of the objective at step two hundred.

![every table 1 column for WMDP Cyber, original versus CRISP, as grouped bars](/assets/images/crisp-metrics_gemma2-2b_cyber.png)

*the cyber bars, where the broken axis is visible directly. concept, rescaled by fifty, is a stub near five where the paper's untouched model would put it near ninety, and the overall column is dragged to eighteen behind it. fluency at sixty five is in a plausible range, so it is specifically the concept pass that is wrong rather than the rater as a whole. cyber scores a little higher than bio, 0.10 against 0.02, which is consistent with the story above: "cybersecurity" is closer to being the literal subject of prefixes like "antivirus software detects malicious code" than "biosecurity" is to anything in the bio set.*

it also poisons the headline column. the overall score is a harmonic mean over the five rescaled axes, and a harmonic mean with a term near zero collapses to near zero. that is the entire reason my overall reads 4.66 against the paper's 54.37, and it means no overall based comparison in my table is meaningful, mine to mine included. i flagged in the last post that the harmonic mean has teeth because it is dominated by its smallest term. it turns out the first thing those teeth bit was me.

there is a smaller related bug: my report labels concept as lower is better, while the paper's table 1 has it higher is better and equation 12 treats it that way. the label is wrong even once the values are fixed.

## what i ruled out

before concluding undertraining i checked the things that would have been more embarrassing.

the adapter is actually active at evaluation. the trained `PeftModel` is the same object passed into the evaluation harness, LoRA is disabled only inside the frozen reference context and restored on exit, and an integration test asserts both that adapter disabled logits equal pre LoRA logits and that a perturbed adapter's logits differ. so the edit is not being silently dropped.

the hyperparameters match appendix F, checked field by field against the PDF.

the base model and multiple choice harness are correct, which the original model row demonstrates independently of anything CRISP does.

and the optimisation path itself works. one of the forty three tests trains the real loop on a tiny random Llama for twelve steps with beta and gamma set to zero and alpha set to one, and asserts that the unlearning loss at the last step is below the first. it passes. so gradients do reach the adapter through the SAE encode, and the term is reducible when it is the only thing being asked for. that is a narrow claim — twelve steps, a random model, a weighting nobody would train at — but it does move "the plumbing is broken" well down the list, and it sharpens what is left: at alpha of 0.01 the same machinery does not move the number in two hundred steps.

what remains uneliminated is training duration, how much of equation 9's magnitude is the lambda term rather than the salient features, feature selection quality, and whether the selected features mediate the multiple choice answer at all.

## an aside on where the time went

worth recording because it shaped how much i could iterate. of the forty eight minutes, the LLM judge was thirty seven, seventy eight percent of the run. the thing being reproduced, feature selection plus CRISP training plus the accuracy numbers the claim rests on, is under five minutes.

the reason is that the rater scores a hundred prefixes twice, for fluency and for concept, and it is a thinking checkpoint, so it spends most of its two thousand and forty eight token budget reasoning before emitting the number that gets parsed. the tell is the three of two hundred and four of two hundred unparsed ratings, which are the ones that hit the cap mid reasoning, meaning the rest are using most of it. batching helped, sixteen at a time took this from about a hundred minutes down to eighteen per stage, but batching divides the number of batches and not the tokens each sequence has to decode, so there is a floor.

the practical consequence is that dropping the judge while iterating is a tenfold speedup and costs only the two columns that are broken anyway. a non thinking rater for the final table is the other obvious fix. and separately, three minutes and forty seconds per training stage went to reading a 2.5 GB corpus over a Drive FUSE mount, which is pure accounting error on my part and fixed by copying to local disk once.

## what i would run next

feature selection is cached, so a CRISP only run without the judge is about four minutes. before the sweep, the three line change that splits the logged unlearn term into its salient part and its lambda part, so the sweep produces a decomposition rather than one more flat scalar. then duration, because that is the parameter the paper leaves unspecified:

```bash
for S in 500 1000 2000; do
  python -m crisp train -c configs/gemma2-2b_bio.yaml \
    --run-name "gemma2-2b_bio_crisp_s${S}" -o "train.steps=${S}" \
    --no-judge --skip-generation
done
```

the thing to watch is the unlearning loss, not the accuracy, and specifically its mean over the first and last quarter of the history. three outcomes, all informative:

* the loss falls and WMDP drops toward 25. reproduced, and the missing ingredient was training duration, which belongs in the writeup as a gap in the paper.
* the loss falls and WMDP stays at 55. the suppressed features do not mediate the multiple choice answer, which is a real negative result about the method rather than about my run.
* the loss is still flat at two thousand steps. the optimiser is not reducing the term at all, which given that the tiny model test does reduce it points at the decomposition above — the salient half being swamped by the lambda half — and after that at feature selection, and only then at alpha.

separately, and independent of all of that, the concept column needs new prefixes before any overall number is quotable. not a scorer fix; the scorer is doing what it was asked. the hundred bio prefixes have to be openers that a biosecurity aware model would continue into biosecurity content, so that an untouched Gemma scores near two and there is something for unlearning to take away. the sanity check is the same either way: run the concept pass on the original model and read the raw outputs, where most ratings should be one or two.

and the RMU and ELM baselines should run on the real model once, if only so that the table has a row where something moves. if a baseline drops WMDP Bio and CRISP does not, under the same harness on the same day, that separates "my pipeline cannot unlearn" from "my CRISP does not unlearn", which is a distinction this run cannot currently make.

## what i am willing to claim

being clear about this is most of the value of the exercise.

defensible from this run: the original model row, which reproduces the paper closely and validates the pipeline. and the unlearn, retain and MMLU columns for CRISP, reported as a failure to reproduce, 55.26 to 29.67 in the paper against 55.42 to 55.42 here, under hyperparameters verified identical to appendix F, with an unspecified step count named as the most likely cause.

not defensible: the overall column, which is an artifact of a concept score measuring the wrong thing, and the fluency and concept columns generally. also not defensible is any statement about CRISP relative to RMU or ELM, which i did not run at this scale.

worth separating the two failures by kind, because they are not the same kind. the flat unlearning loss is a gap in the paper that i hit; the concept score is a gap in the paper that i filled badly. both come from the same place — appendix F does not give a step count, appendix E does not give its prefixes, appendix D does not give its coherence sentences — and in a paper whose result rests on a hyperparameter search of roughly two thousand four hundred runs, the unpublished inputs to that search are not a detail. the hyperparameters are all there, printed to two decimal places. what is missing is everything you would need to know what the hyperparameters were searched over.

i would rather publish that than a table with a caveat buried under it. a failed reproduction with a specific mechanism, a named missing parameter and a concrete next experiment is a more useful artifact than a successful one that nobody can check, and the code, configs, figures and the full run diagnosis are all in the repo so that the next person starts where i stopped rather than where i started.

numbers to follow, again, but this time with the step count as the variable.
