
import os
import json
from dataclasses import dataclass

# PyTorch Imports
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast

# Transformers Imports
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# PEFT Imports
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig

# Dataset Handling
from datasets import load_from_disk

# Progress Bar
from tqdm import tqdm

import random

@dataclass
class TrainConfig:
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    # model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    # model_id: str = "meta-llama/Llama-3.1-8B-Instruct"

    batch_size_training: int = 1
    val_batch_size: int=1

    chunk_size: int = 2048
    num_epochs: int = 2

    gradient_accumulation_steps: int = 1

    gradient_clipping: bool = True
    gradient_clipping_threshold: float = 1.0

    lr: float = 1e-3
    weight_decay: float = 0.01
    gamma: float = 0.85

    quantization: bool = True
    quantization_bit: int = 8
    mixed_precision: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path: str = "/home/mhanna/Master Degree/Finetuning and testing/full_data_set_HF/"

    lora_r: int = 8
    lora_alpha: int = 16

def load_llama_model_and_tokenizer(config: TrainConfig):

    print(f"Device selected: {config.device}")

    if config.quantization and config.device == "cuda":
        print(f"Loading model with {config.quantization_bit}-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=(config.quantization_bit == 8),
            load_in_4bit=(config.quantization_bit == 4)
        )

        model = LlamaForCausalLM.from_pretrained(
            config.model_id,
            device_map="auto",
            quantization_config=quantization_config
        )
    else:
        print("Loading model without quantization...")
        model = LlamaForCausalLM.from_pretrained(
            config.model_id,
            device_map="auto" if config.device == "cuda" else None
        )
        if config.device == "cpu":
            print("Moving model to CPU...")
            model = model.to("cpu")

    # Print total parameters before freezing
    total_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n - - -> Total trainable parameters BEFORE freezing: {total_params_before / 1e6:.2f} Million\n")

    # Freeze base model layers
    for param in model.base_model.parameters():
        param.requires_grad = False

    print("Base model layers are frozen.")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token  # Assign padding token

    # Resize embeddings if needed
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print(" #####  WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    print("Model and tokenizer loaded successfully.")

    return model, tokenizer

def configure_lora_model(model, config: TrainConfig):

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # target_modules=["q_proj" , "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model

model, tokenizer = load_llama_model_and_tokenizer(TrainConfig())
model = prepare_model_for_kbit_training(model)
lora_model = configure_lora_model(model, TrainConfig())

def load_dataset(config: TrainConfig):

    print("Loading dataset...")

    print(f"Loading dataset from {config.dataset_path}...")
    dataset = load_from_disk(config.dataset_path)
    print(f"Dataset loaded successfully. Total samples: {len(dataset)}")
    return dataset

def split_dataset(dataset, train_ratio=0.99, val_ratio=0.01):

    shuffled_dataset = dataset.shuffle(seed=42)  # Setting a seed ensures reproducibility

    # Calculate split sizes
    total_samples = len(shuffled_dataset)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)

    print(f"Dataset split: {train_size} training samples, {val_size} validation samples.")

    # Split the dataset
    train_dataset = shuffled_dataset.select(range(train_size))
    val_dataset = shuffled_dataset.select(range(train_size, train_size + val_size))

    return train_dataset, val_dataset

def process_and_tokenize(dataset, tokenizer):
    print("Processing, normalizing, and tokenizing dataset...")

    def normalize_sample(sample):
        total_width = sample["total_area"]["width"]
        total_depth = sample["total_area"]["depth"]

        # Normalize objects
        for obj in sample["objects"]:
            # Normalize Object Dimensions
            obj["width"] = round(obj["width"] / total_width, 3)
            obj["depth"] = round(obj["depth"] / total_depth, 3)

            # Normalize Object Coordinates
            for coord in obj["object_coordinate"]:
                coord["x"] = round(coord["x"] / total_width, 3)
                coord["z"] = round(coord["z"] / total_depth, 3)

        # Normalize Total Area Dimensions
        sample["total_area"]["width"] = round(sample["total_area"]["width"] / total_width, 3)
        sample["total_area"]["depth"] = round(sample["total_area"]["depth"] / total_depth, 3)

        return sample

    def tokenize_sample(sample):
        normalized_sample = normalize_sample(sample)
        normalized_sample = sample

        # Save the normalized sample as the response (label)
        response_json_str = str(normalized_sample).replace("'", '"')

        # Create a manipulated copy for the instruction (prompt_d)
        prompt_d = dict(normalized_sample)  # Create a deep copy
        manipulated_objects = []

        for obj in prompt_d["objects"]:
            del obj["object_coordinate"]

            # Randomly drop other keys
            keys_to_keep = list(obj.keys())
            for key in keys_to_keep:
                if random.random() < 0.5:
                    del obj[key]

            # Add non-empty objects to the new list
            if obj:  # Check if the object dictionary is not empty
                manipulated_objects.append(obj)

        # Update the objects list with non-empty objects
        prompt_d["objects"] = manipulated_objects

        # **Check if "objects" is empty and delete it**
        if len(prompt_d["objects"]) == 0:
            del prompt_d["objects"]

        # Convert the manipulated dictionary to a string
        prompt_d_json_str = str(prompt_d).replace("'",'"')

        # Prepare instruction and user string
        user_str = f" specifications:{prompt_d_json_str}"
        instruction_str = (
            'you are to design and generate a 2D industrial work area layout in JSON structure. the work area consists of two objects workbenches and shelves.'
            'Each object is defined by a list of "object_coordinates" not polygon vertices. Ensure objects do not overlap and follow '
            'the arrangement constraints: workbenches must follow a U-shaped arrangement to support workflow efficiency,'
            'with adjustable spacing, shelves must follow a linear arrangement either horizontal or vertical aligned on the boundary.'
            'You have to also match the specifications passed by the user in a JSON structure when they exist.'
        )

        # Construct the prompt
        prompt_str = (
            f"<|start_header_id|>system<|end_header_id|> {instruction_str}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|> {user_str}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>"
        )

        # Tokenize the prompt and response
        prompt = tokenizer(f"{tokenizer.bos_token}{prompt_str}", add_special_tokens=False)
        response = tokenizer(f"{response_json_str}{tokenizer.eos_token}", add_special_tokens=False)

        # Combine prompt and response into final input IDs and attention mask
        input_ids = prompt['input_ids'] + response['input_ids']
        attention_mask = [1] * len(input_ids)

        # Define labels for loss computation: mask prompt tokens with -100
        labels = [-100] * len(prompt['input_ids']) + response['input_ids']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    # Tokenize and process the entire dataset
    processed_dataset = dataset.map(
        lambda sample: tokenize_sample(sample),
        remove_columns=list(dataset.features)
    )

    print(f"Dataset processing, normalization, and tokenization completed. Total samples: {len(processed_dataset)}")
    return processed_dataset

class Chunk_Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.chunk_size = TrainConfig.chunk_size

        self.samples = []

        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }

        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            buffer = {k: v + sample[k] for k,v in buffer.items()}

            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(sample['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(sample['labels'], dtype=torch.long),
        }

    def __len__(self):
        return len(self.samples)

dataset = load_dataset(TrainConfig())

# Print the first 5 samples
print("\n First 5 samples in the dataset:")
for i in range(min(5, len(dataset))):  # Limit to 5 or fewer samples if the dataset is smaller
    print(f"Sample {i + 1}: {dataset[i]}")

train_raw_dataset, eval_raw_dataset = split_dataset(dataset)

# Tokenize each subset
train_tokenized_dataset = process_and_tokenize(train_raw_dataset, tokenizer)
eval_tokenized_dataset = process_and_tokenize(eval_raw_dataset, tokenizer)

print(f"\n Length of train_tokenized_dataset: {len(train_tokenized_dataset)}")
print(f"\n Length of eval_tokenized_dataset: {len(eval_tokenized_dataset)}")

# Chunk each subset
chunked_train_dataset = Chunk_Dataset(train_tokenized_dataset)
chunked_eval_dataset = Chunk_Dataset(eval_tokenized_dataset)

print(f"\n Total chunked_train_dataset: {len(chunked_train_dataset)}")
print(f"\n Total chunked_eval_dataset: {len(chunked_eval_dataset)}")

# Inspect a sample
print(train_tokenized_dataset[0])

# Decode the input_ids back to text
decoded_text = tokenizer.decode(train_tokenized_dataset[0]['input_ids'], skip_special_tokens=False)
print("\n ## Decoded Text:")
print(decoded_text)

# Calculate total tokens in the tokenized dataset
total_tokens = sum(len(sample["input_ids"]) for sample in train_tokenized_dataset)
print(f"Total tokens in the tokenized dataset: {total_tokens}")

# Create DataLoaders
train_loader = DataLoader(
    chunked_train_dataset,
    batch_size=TrainConfig.batch_size_training,
    shuffle=True,  # Shuffle batches for training
    num_workers=2,
    pin_memory=True if TrainConfig.device == "cuda" else False
)

eval_loader = DataLoader(
    chunked_eval_dataset,
    batch_size= TrainConfig.val_batch_size,
    shuffle=False,  # No need to shuffle evaluation data
    num_workers=2,
    pin_memory=True if TrainConfig.device == "cuda" else False
)

print(f"train_loader created with {len(train_loader)} batches.")
print(f"eval_loader created with {len(eval_loader)} batches.")

optimizer = AdamW(
    lora_model.parameters(),
    lr=TrainConfig.lr,
    weight_decay=TrainConfig.weight_decay,
)
scheduler = StepLR(optimizer, step_size=1, gamma=TrainConfig.gamma)

def evaluate(model, eval_dataloader, tokenizer, device, mixed_precision=True):

    print("Starting evaluation...")
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    eval_predictions = []

    # Progress bar for evaluation
    progress_bar = tqdm(eval_dataloader, desc="Evaluating", colour="green", dynamic_ncols=True)

    with torch.no_grad():
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Enable mixed precision if specified
            with autocast(enabled=mixed_precision):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()
            print(f"Step: {step}, Loss: {loss.item():.4f}")

            # Decode model predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            eval_predictions.extend(
                tokenizer.batch_decode(preds, skip_special_tokens=True)
            )

            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # Compute average loss and perplexity
    avg_loss = total_loss / len(eval_dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss))

    print(f"Evaluation completed. Average Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

    return avg_loss, perplexity.item(), eval_predictions

def train(
    model,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    optimizer,
    scheduler,
    gradient_accumulation_steps,
    train_config,
):

    device = train_config.device
    model.to(device)
    scaler = GradScaler(enabled=train_config.mixed_precision)

    # Initialize metrics
    train_loss = []  # Training loss per epoch
    train_prep = []  # Training perplexity per epoch
    val_loss = []    # Validation loss per epoch
    val_prep = []    # Validation perplexity per epoch
    eval_predictions_per_epoch = []  # Evaluation predictions per epoch

    total_steps = len(train_dataloader) * train_config.num_epochs
    progress_bar = tqdm(range(total_steps), colour="blue", desc="Training", dynamic_ncols=True)

    step = 0
    step_loss_accumulated = 0.0  # Accumulate loss across all steps for cumulative averaging

    for epoch in range(train_config.num_epochs):
        model.train()
        epoch_loss = 0.0  # Track loss for the epoch

        for batch in train_dataloader:
            step += 1

            # Move inputs to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Mixed precision training
            with autocast(enabled=train_config.mixed_precision):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / gradient_accumulation_steps
                epoch_loss += loss.item()

            # Accumulate loss for cumulative averaging
            step_loss_accumulated += loss.item()

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient accumulation step
            if step % gradient_accumulation_steps == 0:
                # Unscale gradients and apply clipping before optimizer step
                if train_config.gradient_clipping:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), train_config.gradient_clipping_threshold
                    )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            # Print cumulative average metrics every 20 steps
            if step % 20 == 0:
                cumulative_avg_loss = step_loss_accumulated / step  # Use cumulative step count
                cumulative_avg_perplexity = float(torch.exp(torch.tensor(cumulative_avg_loss)))
                print(f"Epoch: {epoch + 1}, Step: {step}, "
                      f"Cumulative Avg Loss: {cumulative_avg_loss:.4f}, "
                      f"Cumulative Avg Perplexity: {cumulative_avg_perplexity:.4f}")

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                "Epoch": epoch + 1,
                "Step": step,
                "Loss": f"{loss.item():.4f}"  # Optional: Track step loss
            })

        # Update the scheduler after each epoch
        scheduler.step()
        print(f"Epoch {epoch + 1}/{train_config.num_epochs} completed. "
              f"Learning rate adjusted to {scheduler.get_last_lr()}")

        # Log epoch-level training metrics
        epoch_loss /= len(train_dataloader)
        train_epoch_perplexity = float(torch.exp(torch.tensor(epoch_loss)))
        train_loss.append(epoch_loss)
        train_prep.append(train_epoch_perplexity)
        print(f"Epoch {epoch + 1} Training Loss: {epoch_loss:.4f}, Perplexity: {train_epoch_perplexity:.4f}")

        # Evaluate the model after each epoch
        eval_loss, eval_perplexity, eval_predictions = evaluate(
            model=model,
            eval_dataloader=eval_dataloader,
            tokenizer=tokenizer,
            device=train_config.device,
            mixed_precision=train_config.mixed_precision
        )
        val_loss.append(eval_loss)
        val_prep.append(eval_perplexity)
        eval_predictions_per_epoch.append(eval_predictions)
        print(f"Epoch {epoch + 1} Eval Loss: {eval_loss:.4f}, Eval Perplexity: {eval_perplexity:.4f}")

    # Compile results
    results = {
        "train_loss": train_loss,
        "train_prep": train_prep,
        "val_loss": val_loss,
        "val_prep": val_prep,
        "eval_predictions": eval_predictions_per_epoch,
    }
    print("Training completed.")
    return results

results = train(
    model=lora_model,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    tokenizer=tokenizer,
    optimizer=optimizer,
    scheduler=scheduler,
    gradient_accumulation_steps=TrainConfig.gradient_accumulation_steps,
    train_config=TrainConfig,
)

def save_lora_merged_model_to_directory(model, tokenizer, save_directory):

    # Step 1: Merge LoRA into the base model
    print("Merging LoRA weights into the base model...")
    merged_model = model.merge_and_unload()  # Merge LoRA into the base model
    print("LoRA weights successfully merged into the base model.")

    # Step 2: Save the merged model and tokenizer
    os.makedirs(save_directory, exist_ok=True)  # Create directory if it doesn't exist
    print(f"Saving LoRA-Merged model and tokenizer to {save_directory}...")
    merged_model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"LoRA-Merged model and tokenizer saved successfully in {save_directory}!")

save_directory = "/home/mhanna/Master Degree/Finetuning and testing/RUN_GENERATION/MK1/"
save_lora_merged_model_to_directory(lora_model, tokenizer, save_directory)

