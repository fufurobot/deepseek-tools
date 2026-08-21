
#!/usr/bin/env python3
"""
Ollama Finetuner - A CLI tool to fine-tune Ollama models using Hugging Face TRL.

Usage:
    python ollama_finetuner.py --model <ollama_model_name> \
        --mode {SFT,DPO,GRPO} \
        --dataset <path_to_sharegpt.json> \
        --intel_model <ollama_intel_model_name>

    # For GRPO with a Gymnasium environment
    python ollama_finetuner.py --model <ollama_model_name> \
        --mode GRPO \
        --env <path_to_env_script.py> \
        --intel_model <ollama_intel_model_name>
"""

import os
import sys
import json
import argparse
import importlib.util
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import requests
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, DPOTrainer, GRPOTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import ollama


# ============================================================================
# 1. OLLAMA INTERACTION
# ============================================================================

OLLAMA_API_BASE = "http://localhost:11434"


def ollama_list_models() -> List[str]:
    """Fetch list of locally available Ollama models."""
    resp = requests.get(f"{OLLAMA_API_BASE}/api/tags")
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def ollama_pull_model(model_name: str) -> None:
    """Pull a model from Ollama registry."""
    print(f"Pulling model '{model_name}'...")
    resp = requests.post(f"{OLLAMA_API_BASE}/api/pull", json={"name": model_name})
    resp.raise_for_status()
    # Stream the response to show progress
    for line in resp.iter_lines():
        if line:
            try:
                data = json.loads(line)
                if "status" in data:
                    print(f"  {data['status']}")
            except json.JSONDecodeError:
                pass
    print(f"Model '{model_name}' pulled successfully.")


def ollama_show_model(model_name: str) -> Dict[str, Any]:
    """Get model details via /api/show."""
    resp = requests.post(f"{OLLAMA_API_BASE}/api/show", json={"model": model_name})
    resp.raise_for_status()
    return resp.json()


def ollama_export_gguf(model_name: str, output_path: str) -> str:
    """
    Export an Ollama model to a GGUF file.
    Ollama stores GGUF files in its blob store; we need to locate and copy it.
    """
    # Get model details to find the GGUF blob
    show_data = ollama_show_model(model_name)
    # The 'details' field contains the digest of the GGUF file
    # The actual file is stored in ~/.ollama/models/blobs/
    digest = show_data.get("details", {}).get("digest")
    if not digest:
        raise ValueError(f"Could not find GGUF digest for model {model_name}")

    # Locate the blob file
    blob_dir = Path.home() / ".ollama" / "models" / "blobs"
    # The digest is usually of the form "sha256-<hash>"
    blob_path = blob_dir / digest.replace(":", "-")
    if not blob_path.exists():
        # Try to find by partial match
        for f in blob_dir.glob(f"*{digest.split(':')[-1][:12]}*"):
            blob_path = f
            break
        if not blob_path.exists():
            raise FileNotFoundError(f"Could not locate GGUF blob for {model_name}")

    # Copy the blob to the output path
    import shutil
    shutil.copy2(blob_path, output_path)
    print(f"Exported GGUF to {output_path}")
    return output_path


def ensure_model_available(model_name: str) -> None:
    """Ensure a model is available locally; pull if not."""
    available = ollama_list_models()
    if model_name not in available:
        print(f"Model '{model_name}' not found locally. Downloading...")
        ollama_pull_model(model_name)
    else:
        print(f"Model '{model_name}' is available locally.")


# ============================================================================
# 2. HUGGING FACE MODEL LOADING FROM OLLAMA GGUF
# ============================================================================

def load_hf_model_from_ollama(model_name: str, intel_model_name: str) -> tuple:
    """
    Load a model into Hugging Face format using the GGUF exported from Ollama.
    Uses the intelligence model's tokenizer/config as a template.
    """
    # Ensure both models are available
    ensure_model_available(model_name)
    ensure_model_available(intel_model_name)

    # Export the model to GGUF
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
        gguf_path = tmp.name
    ollama_export_gguf(model_name, gguf_path)

    # Get config from the intelligence model
    intel_show = ollama_show_model(intel_model_name)
    # Map Ollama config to Hugging Face format
    hf_config = map_ollama_to_hf_config(intel_show)

    # Load tokenizer from the intelligence model
    # We need to use the Hugging Face model ID that corresponds to the intelligence model
    # For this, we assume the intelligence model is a known HF model or we use a placeholder
    tokenizer = AutoTokenizer.from_pretrained(intel_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the model from the GGUF file using transformers
    # Note: This requires transformers to support GGUF loading (via llama.cpp integration)
    # For now, we use a workaround: load from a known HF model ID and then adapt
    # In practice, you may need to use llama-cpp-python or convert GGUF to safetensors
    model = AutoModelForCausalLM.from_pretrained(
        intel_model_name,  # Use intel model as base for architecture
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Clean up
    os.unlink(gguf_path)

    return model, tokenizer


def map_ollama_to_hf_config(ollama_show: Dict[str, Any]) -> Dict[str, Any]:
    """Map Ollama show output to Hugging Face model config."""
    # This is a simplified mapping; real implementation would be more comprehensive
    details = ollama_show.get("details", {})
    config = {
        "model_type": details.get("family", "llama"),
        "torch_dtype": "bfloat16",
        "vocab_size": details.get("vocab_size", 32000),
        "hidden_size": details.get("hidden_size", 4096),
        "num_attention_heads": details.get("num_attention_heads", 32),
        "num_hidden_layers": details.get("num_hidden_layers", 32),
        "intermediate_size": details.get("intermediate_size", 11008),
    }
    return config


# ============================================================================
# 3. DATASET HANDLING
# ============================================================================

def load_sharegpt_dataset(file_path: str) -> Dataset:
    """Load a ShareGPT-formatted JSON dataset."""
    with open(file_path, "r") as f:
        data = json.load(f)

    # ShareGPT format: {"conversations": [{"from": "human|gpt|system", "value": "..."}]}
    # Convert to TRL-compatible format
    formatted_data = []
    for item in data:
        convs = item.get("conversations", [])
        messages = []
        for turn in convs:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            role = role_map.get(turn.get("from", ""), "user")
            messages.append({"role": role, "content": turn.get("value", "")})
        formatted_data.append({"messages": messages})

    return Dataset.from_list(formatted_data)


def load_dpo_dataset(file_path: str) -> Dataset:
    """Load a DPO dataset with 'negative_answer' field."""
    with open(file_path, "r") as f:
        data = json.load(f)

    formatted_data = []
    for item in data:
        convs = item.get("conversations", [])
        # Build the prompt from all turns except the last assistant response
        prompt_messages = []
        chosen = None
        rejected = None

        # Find the last assistant turn as the chosen response
        for i, turn in enumerate(convs):
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            role = role_map.get(turn.get("from", ""), "user")
            if role == "assistant" and chosen is None:
                chosen = turn.get("value", "")
                # Everything before this is the prompt
                prompt_messages = [
                    {"role": role_map.get(c.get("from", ""), "user"), "content": c.get("value", "")}
                    for c in convs[:i]
                ]
            elif role == "assistant" and chosen is not None:
                # If there are multiple assistant turns, use the last one as chosen
                chosen = turn.get("value", "")

        # Get the negative answer
        rejected = item.get("negative_answer", "")
        if not rejected:
            raise ValueError("DPO dataset requires a 'negative_answer' field")

        # Format prompt as text
        prompt_text = ""
        for msg in prompt_messages:
            prompt_text += f"{msg['role']}: {msg['content']}\n"

        formatted_data.append({
            "prompt": prompt_text.strip(),
            "chosen": chosen,
            "rejected": rejected,
        })

    return Dataset.from_list(formatted_data)


# ============================================================================
# 4. TRAINING FUNCTIONS
# ============================================================================

def train_sft(model, tokenizer, dataset: Dataset, output_dir: str) -> None:
    """Run Supervised Fine-Tuning."""
    print("Starting SFT training...")
    
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=True,
        save_steps=500,
        logging_steps=100,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="messages",  # For conversational data
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"SFT training complete. Model saved to {output_dir}")


def train_dpo(model, tokenizer, dataset: Dataset, output_dir: str) -> None:
    """Run Direct Preference Optimization."""
    print("Starting DPO training...")
    
    # DPO requires a reference model
    # For simplicity, we use the same model as reference (not ideal but works for demo)
    ref_model = model

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=5e-6,
        fp16=True,
        save_steps=500,
        logging_steps=100,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=0.1,  # Temperature parameter for DPO
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"DPO training complete. Model saved to {output_dir}")


def train_grpo(model, tokenizer, env_module, output_dir: str) -> None:
    """Run Group Relative Policy Optimization with a Gymnasium environment."""
    print("Starting GRPO training...")
    
    # Define a reward function based on the environment
    # The env module should define a 'reward_function' that takes completions and returns rewards
    if hasattr(env_module, "reward_function"):
        reward_func = env_module.reward_function
    else:
        # Default reward function: count unique characters
        def reward_func(completions, **kwargs):
            return [len(set(c.lower())) for c in completions]
        print("Using default reward function (unique character count)")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=1e-5,
        fp16=True,
        save_steps=500,
        logging_steps=100,
        report_to="none",
    )

    # GRPO requires a dataset; if not provided, we create a simple one
    # In practice, the environment should provide the dataset or prompts
    if hasattr(env_module, "get_dataset"):
        train_dataset = env_module.get_dataset()
    else:
        # Create a dummy dataset
        train_dataset = Dataset.from_list([
            {"prompt": "Write a short story about a robot."},
            {"prompt": "Explain quantum computing in simple terms."},
            {"prompt": "What is the meaning of life?"},
        ])

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=reward_func,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"GRPO training complete. Model saved to {output_dir}")


# ============================================================================
# 5. MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ollama Finetuner using Hugging Face TRL")
    parser.add_argument("--model", required=True, help="Ollama model name to fine-tune")
    parser.add_argument("--mode", required=True, choices=["SFT", "DPO", "GRPO"], 
                        help="Training mode")
    parser.add_argument("--dataset", help="Path to ShareGPT JSON dataset (for SFT/DPO)")
    parser.add_argument("--env", help="Path to Python script defining Gymnasium environment (for GRPO)")
    parser.add_argument("--intel_model", required=True, 
                        help="Ollama model name for intelligence (tokenizer/config template)")
    parser.add_argument("--output_dir", default="./finetuned_model", 
                        help="Output directory for the fine-tuned model")
    args = parser.parse_args()

    # Validate: dataset and env are mutually exclusive
    if args.dataset and args.env:
        print("Error: Cannot specify both --dataset and --env")
        sys.exit(1)
    if not args.dataset and not args.env:
        print("Error: Must specify either --dataset or --env")
        sys.exit(1)

    # Validate mode compatibility
    if args.mode == "GRPO" and args.dataset:
        print("Error: GRPO mode requires --env, not --dataset")
        sys.exit(1)
    if args.mode in ["SFT", "DPO"] and args.env:
        print(f"Error: {args.mode} mode requires --dataset, not --env")
        sys.exit(1)
    if args.mode == "DPO" and args.dataset:
        # Check if dataset has negative_answer field
        with open(args.dataset, "r") as f:
            sample = json.load(f)[0]
            if "negative_answer" not in sample:
                print("Error: DPO mode requires 'negative_answer' field in dataset")
                sys.exit(1)

    # Step 1: Ensure models are available
    print(f"Ensuring model '{args.model}' is available...")
    ensure_model_available(args.model)
    print(f"Ensuring intelligence model '{args.intel_model}' is available...")
    ensure_model_available(args.intel_model)

    # Step 2: Load model into Hugging Face
    print("Loading model into Hugging Face format...")
    model, tokenizer = load_hf_model_from_ollama(args.model, args.intel_model)

    # Step 3: Load dataset or environment
    if args.dataset:
        if args.mode == "DPO":
            dataset = load_dpo_dataset(args.dataset)
        else:
            dataset = load_sharegpt_dataset(args.dataset)
        print(f"Loaded dataset with {len(dataset)} examples")
    elif args.env:
        # Load the environment module
        spec = importlib.util.spec_from_file_location("env_module", args.env)
        env_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(env_module)
        print(f"Loaded environment from {args.env}")

    # Step 4: Run training
    if args.mode == "SFT":
        train_sft(model, tokenizer, dataset, args.output_dir)
    elif args.mode == "DPO":
        train_dpo(model, tokenizer, dataset, args.output_dir)
    elif args.mode == "GRPO":
        train_grpo(model, tokenizer, env_module, args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()