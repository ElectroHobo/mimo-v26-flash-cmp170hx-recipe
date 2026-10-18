# Model and checkpoint licences

What this recipe downloads and runs, and under which terms. The recipe itself
is MIT ([LICENSE](../LICENSE)).

| Component | Source | Licence |
|---|---|---|
| Engine and image | [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx), a fork of [vLLM](https://github.com/vllm-project/vllm) | Apache-2.0; see its `NOTICE` for the attribution it asks for |
| Checkpoint | [ProCreations/MiMo-V2.6-Flash-RL-NVFP4](https://huggingface.co/ProCreations/MiMo-V2.6-Flash-RL-NVFP4), an NVFP4 conversion of the model below | MIT (the model card states that the original model and tokenizer remain under the upstream MIT licence) |
| Model | [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL), including the DFlash drafter packaged with the checkpoint | MIT |
| Benchmark prompts | category prompts from [MiaAI-Lab's DGX Spark MiMo recipe](https://github.com/MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks) | that repository is AGPL-3.0; the prompts are used here as data with attribution, and the harness in `bench/` is this recipe's own |

Nothing from kaka86mm's [vllm-mimo-170hx](https://github.com/kaka86mm/vllm-mimo-170hx)
(MIT) is copied here; its launch configuration informed ours and is credited in
the README.
