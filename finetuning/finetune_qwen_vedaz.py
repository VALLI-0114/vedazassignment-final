"""
finetune_qwen_vedaz.py
-----------------------
QLoRA fine-tunes a Qwen2.5-Instruct (or Qwen3-Instruct) model on the
Vedaz astrology conversation dataset (data/train.jsonl, data/val.jsonl
produced by prepare_data.py).

Design choices (and why):
- QLoRA (4-bit base + LoRA adapters) so this runs on a single consumer/
  cloud GPU (16-24GB VRAM) instead of needing multi-GPU full fine-tuning.
- TRL's SFTTrainer applies the model's own chat template, so Qwen's
  <|im_start|>/<|im_end|> formatting is handled automatically and stays
  consistent with how the model will be served later.
- Loss is masked to assistant turns only (train_on_responses_only-style
  collator) so the model isn't penalized for "predicting" the user's
  questions or the fixed system prompt.

Usage:
    python finetune_qwen_vedaz.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --train_file data/train.jsonl \
        --val_file data/val.jsonl \
        --output_dir out/qwen2.5-vedaz-lora \
        --epochs 3

For a smaller GPU, swap in Qwen/Qwen2.5-3B-Instruct or Qwen/Qwen2.5-1.5B-Instruct.
"""
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train_file", default="data/train.jsonl")
    ap.add_argument("--val_file", default="data/val.jsonl")
    ap.add_argument("--output_dir", default="out/qwen2.5-vedaz-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--merge_after", action="store_true",
                     help="Merge LoRA into base weights after training and save full model.")
    return ap.parse_args()


def main():
    args = build_args()

    # ---- 1. Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 2. Dataset ----
    # SFTTrainer's chat-template formatting expects a "messages" column.
    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    val_ds = load_dataset("json", data_files=args.val_file, split="train")

    def to_text(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    train_ds = train_ds.map(to_text, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(to_text, remove_columns=val_ds.column_names)

    # ---- 3. 4-bit base model (QLoRA) ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # ---- 4. LoRA config ----
    # Qwen2.5 attention/MLP proj names; covers both attention and MLP blocks.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # ---- 5. Mask loss to assistant turns only ----
    # Qwen chat template wraps the assistant turn as:
    #   <|im_start|>assistant\n ... <|im_end|>
    # We only backprop on tokens after this marker.
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer
    )

    # ---- 6. Training config ----
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,  # keep conversations un-packed; dataset is small
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")

    # ---- 7. Optionally merge adapter into base weights ----
    if args.merge_after:
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        merged = PeftModel.from_pretrained(base, args.output_dir)
        merged = merged.merge_and_unload()
        merged_dir = args.output_dir + "-merged"
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged full-precision model saved to {merged_dir} (this is what you'll point vLLM at)")


if __name__ == "__main__":
    main()
