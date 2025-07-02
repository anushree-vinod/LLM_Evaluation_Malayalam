from transformers import AutoTokenizer, TrainingArguments, Gemma3ForCausalLM
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from itertools import product
import evaluate
import torch
import pandas as pd
import numpy as np
import os
import gc

from transformers.utils import logging as t_logging

t_logging.set_verbosity_error()

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def custom_prompt(example):
    text = example["input"]
    prompt = f"""
Generate a **clear and concise news headline in Malayalam only** based on the following text.
Text (Malayalam): {text}
Important:
- The output must be **only a headline in Malayalam**.
- Do **not** use any other language or script.
- Do **not** include any extra commentary or formatting.
- Do **not** copy the text word-for-word.
- Start your output with: Headline:"""
    return {"prompt": prompt, "completion": " " + example["target"]}

def generate_headlines_inference(model, dataset, tokenizer):
    model.eval()

    generated, references = [], []
    inputs = dataset["prompt"]

    for input_text, ref in zip(inputs, dataset["completion"]):
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids.to("cuda:0")

        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=100, do_sample=False)

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = decoded.split("Headline:")[-1].strip()
        generated.append(pred)
        references.append(ref.strip())

    return generated, references



if __name__ == '__main__':
    df = load_dataset("ai4bharat/IndicHeadlineGeneration", "ml")
    train_dataset = df["train"].shuffle(seed=42).select(range(1000))
    val_dataset = df["test"].shuffle(seed=42).select(range(100))

    train_dataset = train_dataset.map(custom_prompt, remove_columns=["input", "target"])
    val_dataset = val_dataset.map(custom_prompt, remove_columns=["input", "target"])

    model_name = "google/gemma-3-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    rouge = evaluate.load("rouge")

    learning_rates = np.logspace(-5, -2, num=8)
    lr_scheduler_types = ["linear", "cosine"]
    weight_decays = [0.01, 0.05, 0.1]

    results_log = []
    
    for lr, scheduler, wd in product(
        learning_rates,
        lr_scheduler_types,
        weight_decays
    ):
        trial_id = f"lr{lr}_sched{scheduler}_wd{wd}"

        model = Gemma3ForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16
        )
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        args = TrainingArguments(
            output_dir=f"./results/{trial_id}",
            num_train_epochs=1,
            lr_scheduler_type=scheduler,
            per_device_train_batch_size=1,
            gradient_accumulation_steps = 4,
            learning_rate=lr,
            weight_decay=wd,
            bf16=True,
            logging_steps=10,
            save_steps=1000,
            report_to="none",
            group_by_length=True
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=lora_config,
            processing_class=tokenizer,
            args=args,
        )

        trainer.train()

        preds, refs = generate_headlines_inference(model, val_dataset, tokenizer)

        rouge_result = rouge.compute(predictions=preds, references=refs)

        result_row = {
            "trial": trial_id,
            "learning_rate": lr,
            "lr_scheduler_type": scheduler,
            "weight_decay": wd,
            **rouge_result
        }
        results_log.append(result_row)

        # Delete model and trainer objects
        del model
        del trainer

        # Empty CUDA cache
        torch.cuda.empty_cache()

        # Collect garbage
        gc.collect()

        print('-' * 50)
        print(f"Completed: {trial_id}")

    df_log = pd.DataFrame(results_log)
    os.makedirs("logs", exist_ok=True)
    df_log.to_csv("logs/tuning_results.csv", index=False)

    # Save results with best config marked
    df_log = pd.DataFrame(results_log)

    # Sort by ROUGE-L (descending) and mark best trial
    df_log = df_log.sort_values(by="rougeL", ascending=False).reset_index(drop=True)
    # df_log["best"] = ["Best: " if i == 0 else "" for i in range(len(df_log))]

    # Save to file
    os.makedirs("logs", exist_ok=True)
    df_log.to_csv("logs/tuning_results.csv", index=False)
