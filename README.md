# Industrial Work Area Generation using Fine-Tuned LLMs

## 📌 Project Overview
This thesis addresses a critical gap in layout generation by developing a method to generate industrial work areas using Large Language Models (LLMs). The project explores how fine-tuned LLMs can create functional industrial layouts when real-world data is unavailable.

![Leveraging Large Language Models for Generating 2D Layout in Industrial Work Areas (2)-21_page-0001](https://github.com/user-attachments/assets/da89e675-f34e-46d0-9d31-5577b49c2d78)

## Project Phases  

### Phase 1: Data Design and Augmentation  
- Created a synthetic dataset since no public datasets exist for industrial work area layouts  
- Designed data schema to represent industrial work areas and their components  
- Implemented data augmentation techniques to expand the training set  

### Phase 2: Model Fine-Tuning  
- Dataset preprocessing to be suitable for `LlamaForCausalLM` class in PyTorch  
- Hyperparameter optimization  
- Parameter-Efficient Fine-Tuning (LoRA)  

### Phase 3: Model Evaluation  
- Prompt consistency  
- Spatial accuracy  
