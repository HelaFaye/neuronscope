# The self-improvement loop, and the trap in it

You asked whether models can improve themselves here. They can, and there is
one specific way to do it that is not self-deception.

## The trap

The obvious loop is: suppress H-Neurons, score the output with the classifier,
tune the suppression to minimise that score, repeat.

That loop optimises a number and teaches you nothing. The classifier *chose*
those 69 neurons; asking it whether suppressing them helped is asking a ruler
to measure itself. You will drive the score wherever you like while the model's
actual behaviour goes somewhere unrelated. Goodhart, in the tightest possible
loop.

The same trap catches the delegation gate: if you tune `max_score` until few
answers are flagged, you have not improved the minions, you have moved a
threshold.

## The version that works

Every loop below has the same shape: **the thing being optimised and the thing
measuring it must not be the same artifact.**

```
  collect  -> questions the classifier has never seen
  measure  -> merge_eval.py, rule judge, paired McNemar
  change   -> suppression scale, profile, minion routing
  re-measure on the SAME held-out set
  keep the change only if the paired test says it helped
```

Three rules that make it honest:

1. **Held-out means never-seen.** Not "a different split of the same
   collection run" -- the classifier saw that distribution. Collect a fresh
   batch, tag it `--task eval`, and never train on it.
2. **The judge is not the classifier.** `merge_eval.py` grades against gold
   answers with the rule judge. The classifier is what you are testing, so it
   does not get a vote.
3. **Paired, not aggregate.** Same questions before and after, McNemar on the
   discordant pairs. An aggregate accuracy that moved 2% on 40 questions moved
   by noise.

## What can actually improve, in order of what is available

**1. The suppression scale.** `tune_scale_server.py` sweeps alpha and picks the
value that reduces hallucination without breaking canary tasks. This is real
self-improvement and needs no training -- but its own help text is right that
below ~200 eval items the confidence interval is wider than the effect.

**2. Minion routing.** Log which worker handled which task type and what the
gate scored. After enough runs, `good_at` stops being your guess and becomes a
measurement. Cheap, honest, and it compounds.

**3. Abstention SFT.** `export_sft_dataset.py` turns your collected pairs into
training data: correct answers stay, hallucinated ones become declines. That is
a genuine improvement loop -- the model learns to say "I don't know" exactly
where it used to fabricate. It needs a training run, which your 12 GB carve-out
cannot host; a 9B QLoRA wants ~24 GB. The friend's 16 GB card could do 4-bit
QLoRA at a short sequence length. This is the one with real upside and real
hardware cost.

**4. Task-specific profiles.** `task_neurons.py` already tells you whether
coding and trivia recruit the same neurons. If they do not, a profile per
domain beats one general profile, and the delegation router can pick the
profile that matches the subtask.

## What cannot improve itself, and why

Nothing here should rewrite its own evaluation set, its own judge, or its own
`max_score`. A loop permitted to touch those will converge on a configuration
that scores well and does nothing, and it will look like success the whole way.
Keep those three files under version control and change them by hand.
