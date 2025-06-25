import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset
from tqdm import tqdm
from datasets import load_dataset

import evaluate
import numpy as np
import re

# Load metrics once
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

def get_dataset():
    df = load_dataset("ai4bharat/IndicHeadlineGeneration", "ml")
    df = df["test"].select(range(500))
    texts = df['input']
    references = df['target']

    return texts, references

def compute_metrics(predictions, references, rouge_stemmer=True):
    """
    Computes ROUGE, BLEU, and BERTScore for given predictions and references.
    
    Args:
        predictions (list): List of predicted texts.
        references (list): List of reference texts.
        rouge_stemmer (bool): Whether to use stemming for ROUGE.
    
    Returns:
        dict: ROUGE, BLEU, and BERTScore metrics rounded to four decimal places.
    """
    
    # Compute ROUGE scores
    rouge_result = rouge.compute(predictions=predictions, references=references, use_stemmer=rouge_stemmer)
    
    # Compute BLEU scores
    bleu_result = bleu.compute(
        predictions=predictions,
        references=[[r] for r in references]
    )

    # Compute BERTScore (for Malayalam use lang="ml")
    bert_result = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="ml"  # Malayalam
    )

    return {
        "ROUGE-1": round(rouge_result['rouge1'], 4),
        "ROUGE-2": round(rouge_result['rouge2'], 4),
        "ROUGE-L": round(rouge_result['rougeL'], 4),
        "BLEU": round(bleu_result["bleu"], 4),
        "BERTScore-P": round(np.mean(bert_result["precision"]), 4),
        "BERTScore-R": round(np.mean(bert_result["recall"]), 4),
        "BERTScore-F1": round(np.mean(bert_result["f1"]), 4)
    }

texts, references = get_dataset()

def print_results(predictions, num_results = 5):
    for i in range(num_results):
        print(f"Input Text:\n{texts[i]}\n")
        print(f"Generated Summary:\n{predictions[i]}\n")
        print(f"Reference Summary:\n{references[i]}\n")
        print("="*80)

def create_prompt(text):
    return [
        {"role": "system", "content": "You are a helpful assistant that generates news headlines."},
        {"role": "user", "content": f"""
Generate a clear and concise news headline in Malayalam only based on the following text.

Text (Malayalam): {text}

Important:
- The output must be only a headline in Malayalam.
- Do not use any other language or script.
- Do not include any extra commentary or formatting.
- Do not copy the text word-for-word.
- Start your output with: Headline:

Example (do not include this in your output):
Headline: ഇന്ത്യയിൽ പുതിയ ശാസ്ത്രീയ കണ്ടെത്തൽ
"""}
    ]


if __name__ == "__main__":

    # Load DeepSeek Chat model
    model_name = "deepseek-ai/deepseek-llm-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:1"
    )
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id


    # Generate headlines
    deepseek_preds = []

    batch_size = 4
    max_total_tokens = 4096
    max_new_tokens = 512

    # Calculate max input length allowed
    max_input_tokens = max_total_tokens - max_new_tokens


    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Malayalam headlines"):
        batch_texts = texts[i:i+batch_size]
        prompts = [create_prompt(text) for text in batch_texts]

        try:
            input_tensor = tokenizer.apply_chat_template(
                prompts,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens
            ).to(model.device)

            attention_mask = input_tensor != tokenizer.pad_token_id

            with torch.no_grad():
                outputs = model.generate(input_tensor, attention_mask=attention_mask, max_new_tokens=512)
            
            
            for j in range(len(batch_texts)):
                output_text = tokenizer.decode(
                    outputs[j][input_tensor.shape[1]:],
                    skip_special_tokens=True
                )
                deepseek_preds.append(output_text)

            # result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
            # deepseek_preds.append(result)
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print("⚠️ CUDA OOM – Skipping this input.")
            deepseek_preds.append("[OOM ERROR – Skipped]")
            torch.cuda.empty_cache()


    deepseek_preds = [re.sub(r"(?i)^\s*Headline:\s*", "", p) for p in deepseek_preds]
    res = compute_metrics(deepseek_preds, references)
    print(res)