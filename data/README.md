# Data

Put the TriviaQA parquet files here:

    data/TriviaQA/rc.nocontext/train-00000-of-00001.parquet
    data/TriviaQA/rc.nocontext/validation-00000-of-00001.parquet

They ship with the upstream H-Neurons repo, or:

    hf download mandarjoshi/trivia_qa --repo-type dataset \
        --include "rc.nocontext/*" --local-dir data/TriviaQA

(`hf` comes with `huggingface_hub`; older versions call it `huggingface-cli`.)

Generated artifacts (consistency_samples.jsonl, answer_tokens.jsonl,
*_qids.json, activations/) land here too and are gitignored.
