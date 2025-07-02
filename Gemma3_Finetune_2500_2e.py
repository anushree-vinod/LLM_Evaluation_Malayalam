from transformers import AutoTokenizer, TrainingArguments, Gemma3ForCausalLM
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
import torch

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



if __name__ == '__main__':

    # Load a sample dataset (for example purposes, we use wikitext-2)
    df = load_dataset("ai4bharat/IndicHeadlineGeneration", "ml")

    # Shuffle and select 1000 samples from the training set
    train_dataset = df["train"].shuffle(seed=42).select(range(2000))

    # Shuffle and select 300 samples from the test set
    test_dataset = df["test"].select(range(300))

    # Apply the custom prompt formatting
    train_dataset = train_dataset.map(custom_prompt, remove_columns=["input", "target"])
    test_dataset = test_dataset.map(custom_prompt, remove_columns=["input", "target"])

    # Define the model name
    model_name = "google/gemma-3-4b-it"

    # Load the tokenizer and adjust padding settings
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Use eos token as pad token
    tokenizer.padding_side = "right"

    # Load the Gemma 3 model
    model = Gemma3ForCausalLM.from_pretrained(model_name, device_map="cuda:0", torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # Disable caching for training

    # Set up LoRA configuration for causal language modeling
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=25,
        report_to="tensorboard",
        group_by_length=True,
        bf16=True,
    )

    # Create the SFTTrainer with LoRA parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("finetuned_models/gemma3_finetuned_2500_2e-4")