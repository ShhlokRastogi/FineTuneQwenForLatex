# Fine-tuning Qwen2-VL-7B for LaTeX OCR

A Colab notebook that fine-tunes **Qwen2-VL-7B-Instruct** to convert images of mathematical expressions into LaTeX code. It uses [Unsloth](https://github.com/unslothai/unsloth) with 4-bit quantization and LoRA, so it fits on a single free-tier GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/blob/main/finetuning_qwen_vl.ipynb)

## Overview

| | |
|---|---|
| **Base model** | `unsloth/Qwen2-VL-7B-Instruct` (loaded in 4-bit) |
| **Method** | LoRA (r=16, alpha=16) on vision and language layers |
| **Dataset** | [`unsloth/Latex_OCR`](https://huggingface.co/datasets/unsloth/Latex_OCR) (image → LaTeX pairs) |
| **Task prompt** | "Write the LaTeX representation for this image." |
| **Trainer** | TRL `SFTTrainer` with `UnslothVisionDataCollator` |

## Requirements

- Google Colab with a GPU runtime (T4 or better)
- Python packages, installed in the first cell: `unsloth`, `unsloth_zoo`, `bitsandbytes`, `accelerate`, `xformers`, `peft`, `trl`, `triton`, `cut_cross_entropy`, `datasets`, `huggingface_hub`, `hf_transfer`, `sentencepiece`, `protobuf`

## Usage

1. Open the notebook in Colab (badge above) and set **Runtime → Change runtime type → GPU**.
2. Run all cells in order (**Runtime → Run all**).

## What the notebook does

1. **Install dependencies** for Unsloth and its supporting libraries.
2. **Load the model** with `FastVisionModel.from_pretrained` in 4-bit with Unsloth gradient checkpointing.
3. **Attach LoRA adapters** with `get_peft_model`, fine-tuning vision layers, language layers, attention modules, and MLP modules.
4. **Load the dataset** (`unsloth/Latex_OCR`, train split) and inspect a few samples.
5. **Convert samples to a chat format**: each example becomes a user turn (instruction + image) and an assistant turn (the LaTeX string).
6. **Run baseline inference** to see how the model performs before training.
7. **Train** with `SFTTrainer`.
8. **Run inference again** on a different sample to compare results.

## Training configuration

| Setting | Value |
|---|---|
| Per-device batch size | 2 |
| Gradient accumulation steps | 4 (effective batch size 8) |
| Max steps | 30 |
| Learning rate | 2e-4 |
| Optimizer | `adamw_8bit` |
| LR scheduler | linear, 5 warmup steps |
| Weight decay | 0.01 |
| Precision | bf16 if supported, otherwise fp16 |
| Max sequence length | 2048 |
| Seed | 3407 |

Training runs for only 30 steps, which is meant as a quick demonstration. For better results, remove `max_steps` and set `num_train_epochs` (for example, 1 to 3), or increase `max_steps`.

## Inference

After training, switch the model to inference mode and generate LaTeX for an image:

```python
FastVisionModel.for_inference(model)

messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Write the LaTeX representation for this image."},
]}]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

_ = model.generate(**inputs, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)
```

## Saving the model

The notebook does not save the fine-tuned weights, so they are lost when the Colab session ends. To keep them, add a cell after training:

```python
# Save LoRA adapters only (small)
model.save_pretrained("qwen2vl_latex_lora")
tokenizer.save_pretrained("qwen2vl_latex_lora")

# Or push to the Hugging Face Hub
model.push_to_hub("your-username/qwen2vl-latex-ocr-lora", token="YOUR_HF_TOKEN")
tokenizer.push_to_hub("your-username/qwen2vl-latex-ocr-lora", token="YOUR_HF_TOKEN")
```

## Adapting to your own data

Replace the dataset and instruction to fine-tune for a different image-to-text task. Your dataset needs an image column and a text column, which you then map into the conversation format in `convert_to_conversation`.

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) for the fast, memory-efficient fine-tuning framework
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) by Alibaba Cloud
- [`unsloth/Latex_OCR`](https://huggingface.co/datasets/unsloth/Latex_OCR) dataset
