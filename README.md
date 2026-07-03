# Vedaz AI Astrologer — Fine-Tuning Assessment

Fine-tuning Qwen2.5-Instruct on Vedaz's Vedic-astrology conversation data,
plus a deployment write-up for serving the result on a VPS with vLLM.

**Note:** Due to the lack of GPU resources in the current environment, the repository contains a complete, reproducible QLoRA fine-tuning pipeline. The provided scripts can be executed on any CUDA-enabled GPU to generate the LoRA adapter weights.

## Submission Contents

| Requirement | Location |
|---|---|
| GitHub repository | this repo |
| Write-up (PDF/Markdown) | [`writeup/vLLM_VPS_Hosting_Writeup.pdf`](writeup/vLLM_VPS_Hosting_Writeup.pdf) ([Markdown source](writeup/vLLM_VPS_Hosting_Writeup.md)) |
| Model adapter / HF link | see [Model Adapter](#model-adapter) below |

## Repo Structure

```
.
├── finetuning/
│   ├── prepare_data.py          # cleans the raw chat-data export into train/val JSONL
│   ├── finetune_qwen_vedaz.py   # QLoRA fine-tune of Qwen2.5-Instruct (TRL SFTTrainer)
│   ├── push_to_hub.py           # uploads the resulting adapter/model to Hugging Face
│   ├── requirements.txt
│   └── data/
│       ├── train.jsonl          # 50 cleaned conversation examples
│       └── val.jsonl            # 5 cleaned conversation examples
└── writeup/
    ├── vLLM_VPS_Hosting_Writeup.md
    └── vLLM_VPS_Hosting_Writeup.pdf
```

## 1. Data

The source export (`Chat_Data_for_assessment_of_applicants.json`) wasn't
valid JSONL — some entries were pretty-printed across multiple lines with
trailing commas. `prepare_data.py` stream-parses it robustly and writes
clean, schema-validated `train.jsonl` / `val.jsonl` splits (55 conversations
total, 50/5 split). Note the honest caveat: 55 examples is small — this
demonstrates a correct end-to-end fine-tuning pipeline, not a
production-scale dataset. For a real deployment, more labeled conversations
(especially harder edge cases: crisis situations, repeated pushback for a
guaranteed answer, mixed Hindi/English code-switching) would meaningfully
improve robustness.

```bash
python finetuning/prepare_data.py \
  --input Chat_Data_for_assessment_of_applicants.json \
  --out_dir finetuning/data
```

## 2. Fine-Tuning

QLoRA (4-bit base + LoRA adapters) via Hugging Face `transformers` + `peft`
+ TRL's `SFTTrainer`, using Qwen's own chat template and masking loss to
assistant turns only. Runs on a single 16–24GB GPU.

```bash
pip install -r finetuning/requirements.txt

python finetuning/finetune_qwen_vedaz.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --train_file finetuning/data/train.jsonl \
  --val_file finetuning/data/val.jsonl \
  --output_dir out/qwen2.5-vedaz-lora \
  --epochs 3 \
  --merge_after
```

For a smaller GPU, swap `--model_name` to `Qwen/Qwen2.5-3B-Instruct` or
`Qwen/Qwen2.5-1.5B-Instruct`.

**This step needs an actual GPU** (this repo/README was prepared without one
available) — run it on your own machine, Colab, or a rented GPU box
(RunPod / Lambda / Vast.ai all work fine for a 7B QLoRA run on this
dataset size — expect well under 30 minutes).

## 3. Model Adapter

<!--
Once you've run step 2 and pushed the result with push_to_hub.py, replace
this line with your actual Hugging Face link, e.g.:
**Adapter:** https://huggingface.co/<your-username>/vedaz-qwen2.5-lora
-->
**Not yet published** — run the fine-tuning step above, then:

```bash
pip install huggingface_hub
huggingface-cli login

python finetuning/push_to_hub.py \
  --local_dir out/qwen2.5-vedaz-lora \
  --repo_id <your-hf-username>/vedaz-qwen2.5-lora
```

This uploads the LoRA adapter only (a few MB, loads on top of the public
`Qwen/Qwen2.5-7B-Instruct` base) — the standard way to share a fine-tune
without re-hosting the full base model weights. Paste the resulting
`https://huggingface.co/...` link here and in this README once done.

## 4. Serving

See [`writeup/vLLM_VPS_Hosting_Writeup.md`](writeup/vLLM_VPS_Hosting_Writeup.md)
for the full VPS + vLLM deployment guide (systemd service, Nginx + TLS
reverse proxy, OpenAI-compatible client usage, monitoring notes).

## License

MIT — see [LICENSE](LICENSE). Conversation data belongs to Vedaz and is
included here only for this assessment.
