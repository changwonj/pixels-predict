# Parameter-Efficient Fine-Tuning for Multimodal Science QA (SmolVLM-500M)

This project is for the Spring 2026 Deep Learning Final. It investigates parameter-efficient fine-tuning strategies for multimodal scientific multiple-choice question answering.
The model receives an image, a science question, and multiple answer choices, then predicts the correct option index.

We fine-tune `HuggingFaceTB/SmolVLM-500M-Instruct` using quantized PEFT methods (LoRA and DoRA) under a trainable-parameter budget of 5M parameters.
We compare prompt context settings (`q_choices`, `q_hint`, `q_hint_lecture`) and adapter placement strategies (attention-only vs attention+MLP projections).

Key result trends:
- Richer textual context (question + hint + lecture) gives the strongest validation performance.
- LoRA and DoRA are broadly similar.
- DoRA gives the best final validation result in these experiments.
- Wider adapter placement does not always improve accuracy, but can improve NLL.

## Repository Structure

- `train.ipynb` - LoRA training notebook (quantized PEFT fine-tuning + evaluation + artifact export).
- `train_dora.ipynb` - DoRA training notebook (same training pipeline, DoRA-specific adapter config).
- `generate_submission.ipynb` - Inference notebook to load a trained adapter and produce `submission.csv`.
- `artifacts/` - Metric CSVs and plots.

## Data Expectations

The notebooks expect extracted CSV/image data, which was provided by the course.

Expected CSVs:
- `train.csv`
- `val.csv`
- `test.csv`
- `sample_submission.csv` (for format checking)

Important fields used:
- `id`
- `image_path`
- `question`
- `choices` (JSON list in CSV, parsed with `json.loads`)
- `num_choices`
- `answer` (train/val only)
- optional: `hint`, `lecture`

## Training Notebook Walkthrough (`train.ipynb` and `train_dora.ipynb`)

### 1) Environment and data bootstrap
- Installs core dependencies (`transformers`, `peft`, `bitsandbytes`, `accelerate`, etc.).
- Mounts Google Drive.
- Copies and unzips dataset archive into `/content`.

### 2) Imports and global config
Defines:
- model: `HuggingFaceTB/SmolVLM-500M-Instruct`
- output directory (`OUT_DIR`) for checkpoints/artifacts
- training hyperparameters (epochs, batch sizes, LR, grad accumulation, warmup ratio)
- context modes:
  - `q_choices` (question + choices)
  - `q_hint` (question + hint + choices)
  - `q_hint_lecture` (question + hint + lecture + choices)
- optional sweep configs

### 3) Reproducibility and device setup
- Seeds Python/NumPy/Torch RNGs.
- Prints GPU/CUDA info when available.

### 4) Data loading and basic inspection
- Reads `train.csv` and `val.csv`.
- Parses `choices` JSON strings into Python lists.
- Supports optional subset mode for fast iteration.
- Prints dataset size and class/choice distributions.

### 5) Prompt/message construction
Core helper functions:
- `build_user_text(...)` builds the text prompt with optional hint/lecture context.
- `build_messages(...)` formats a chat-style user message containing image + text.

The prompt requests exactly one capital-letter answer, enabling next-token letter scoring.

### 6) Dataset classes
- `ScienceQATrainDataset`
  - loads image + row metadata
  - applies optional context dropout augmentation (`hint`/`lecture` blanking)
  - optionally shuffles choices and remaps the correct answer index
- `ScienceQAValDataset`
  - deterministic validation formatting (no augmentation/shuffling)

### 7) Quantized base model loading
`load_processor_and_model(...)`:
- loads processor/tokenizer
- ensures a pad token exists
- loads model in 4-bit NF4 (`BitsAndBytesConfig`)
- prepares model for k-bit PEFT training (`prepare_model_for_kbit_training`)

### 8) Adapter attachment under 5M cap
Shared logic:
- detect target modules
- try candidate ranks
- count trainable parameters
- keep the first config under cap

LoRA notebook:
- Uses `LoraConfig` without DoRA.

DoRA notebook:
- Uses `LoraConfig(..., use_dora=True)`.
- Includes `rank_pattern` and `alpha_pattern`.
- Config includes attention and MLP projections.

### 9) Collation + letter-token mapping
- Builds batched multimodal tensors from chat template + images.
- Maps each answer letter (`A..G`) to valid single-token IDs (raw and spaced variants).
- Enables robust letter-choice scoring.

### 10) Evaluation (`evaluate_mc`)
For each batch:
- Runs forward pass.
- Extracts next-token logits.
- Converts token logits to choice scores via log-sum-exp over letter token IDs.
- Masks invalid choices beyond each sample's `num_choices`.
- Computes:
  - `val_mc_accuracy`
  - `val_mc_nll`
  - `val_loss_letter`

### 11) Training loop (`train_one_run`)
- Creates dataloaders, optimizer (`AdamW`), cosine schedule with warmup.
- Uses gradient accumulation and gradient clipping.
- Optimizes cross-entropy over masked choice scores.
- Runs validation each epoch.
- Saves per-epoch checkpoint:
  - adapter weights
  - processor
  - `run_config.json` with run config and metrics

### 12) Experiment aggregation + evidence
- Runs either one config or a small context sweep.
- Saves:
  - `metrics_history.csv`
  - `experiment_summary.csv`
- Selects best run by accuracy (tie-break by NLL).
- Generates evidence plots (loss/accuracy/NLL/param comparisons).
- Writes `best_checkpoint.txt` for downstream inference.

## Submission Notebook Walkthrough (`generate_submission.ipynb`)

### 1) Setup and checkpoint selection
- Installs deps, mounts Drive, and unzips data.
- Resolves adapter path from:
  1. `best_checkpoint.txt` if present, otherwise
  2. fallback run path defined in notebook config

### 2) Optional experiment metadata check
- Reads selected `run_config.json` if present.
- Optionally filters `experiment_summary.csv` to show the selected run/checkpoint row.

### 3) Model reconstruction for inference
- Loads processor (from adapter dir if available).
- Loads quantized 4-bit base model.
- Attaches adapter via `PeftModel.from_pretrained(base, ADAPTER_DIR)`.
- Sets eval mode.

### 4) Prediction logic
- Rebuilds prompts/messages in the same format as training.
- Infers `context_mode` from `run_config.json` when available.
- Batches test rows, scores answer letters, masks invalid options.
- Chooses argmax choice index per sample.

### 5) Submission writing and checks
- Writes `submission.csv` with columns `[id, answer]`.
- Asserts column format and row count.
- Asserts each prediction is in `[0, num_choices)`.

## Typical Workflow

1. Run `train.ipynb` for LoRA experiments.
2. Run `train_dora.ipynb` for DoRA experiments.
3. Compare `experiment_summary.csv` and plots.
4. Use the selected best adapter with `generate_submission.ipynb`.
5. Upload generated `submission.csv`.

## Submission Artifacts

- [Project Report](https://drive.google.com/file/d/1NCBQNOWV-o0V9Tqs1a1YCzzF2gdmoX-O/view)
- [Model checkpoints](https://drive.google.com/drive/u/0/folders/1gSQVTLVwq4Vom6KEDfbcLXCQGO3kXuFl)