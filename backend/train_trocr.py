"""
Fine-tune TrOCR for Devanagari handwriting with LoRA.

This script is Mac/Apple Silicon friendly by default:
- uses MPS when available
- avoids CUDA-only fp16 training on MPS
- can train directly from HuggingFace or from extracted ./data folders
"""

from __future__ import annotations

import argparse
import io
import json
import os
import unicodedata
from pathlib import Path
from typing import Any, Optional

import editdistance
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
    set_seed,
)

MODEL_NAME = "paudelanil/trocr-devanagari-2"
DATASET_NAME = "c3rl/IIIT-INDIC-HW-WORDS-Hindi"
DEFAULT_IMAGE_PROCESSOR = "google/vit-base-patch16-224-in21k"

# LoRA target-module presets. PEFT treats a string as a regex over full module
# paths and a list as suffix matches. "legacy" reproduces the shipped adapter
# (note: bare "dense" also matches ViT FFN layers via suffix matching).
TARGET_MODULE_PRESETS: dict[str, Any] = {
    "legacy": ["query", "key", "value", "dense"],
    "attn": r".*(query|key|value|q_proj|k_proj|v_proj|out_proj|attention\.output\.dense)$",
    "attn-ffn": r".*(query|key|value|q_proj|k_proj|v_proj|out_proj|dense|fc1|fc2)$",
}


def get_best_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_trocr_processor(model_name: str) -> TrOCRProcessor:
    try:
        return TrOCRProcessor.from_pretrained(model_name)
    except Exception as exc:
        print(f"Falling back to ViT image processor + checkpoint tokenizer: {exc}")
        image_processor = ViTImageProcessor.from_pretrained(DEFAULT_IMAGE_PROCESSOR)
        image_processor.image_mean = [0.5, 0.5, 0.5]
        image_processor.image_std = [0.5, 0.5, 0.5]
        image_processor.rescale_factor = 1.0 / 255.0
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


class ExtractedDevanagariDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        processor: TrOCRProcessor,
        max_target_length: int,
        limit: Optional[int] = None,
        preprocess: str = "none",
    ):
        self.split_dir = split_dir
        self.processor = processor
        self.max_target_length = max_target_length
        self.preprocess = preprocess
        self.df = pd.read_csv(split_dir / "labels.csv")
        if limit:
            self.df = self.df.head(limit)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = self.split_dir / "images" / row["filename"]
        image = Image.open(image_path).convert("RGB")
        return encode_sample(image, str(row["text"]), self.processor, self.max_target_length, self.preprocess)


class HuggingFaceDevanagariDataset(Dataset):
    def __init__(
        self,
        split: str,
        processor: TrOCRProcessor,
        max_target_length: int,
        limit: Optional[int] = None,
        preprocess: str = "none",
    ):
        split_spec = f"{split}[:{limit}]" if limit else split
        self.dataset = load_dataset(DATASET_NAME, split=split_spec)
        self.processor = processor
        self.max_target_length = max_target_length
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        image = coerce_to_pil(sample["image"])
        return encode_sample(image, str(sample["text"]), self.processor, self.max_target_length, self.preprocess)


class ParquetDevanagariDataset(Dataset):
    def __init__(
        self,
        parquet_dir: Path,
        split: str,
        processor: TrOCRProcessor,
        max_target_length: int,
        limit: Optional[int] = None,
        preprocess: str = "none",
    ):
        pattern = parquet_dir / f"{split}-*.parquet"
        files = sorted(str(path) for path in parquet_dir.glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found for {split!r}: {pattern}")
        split_spec = f"train[:{limit}]" if limit else "train"
        self.dataset = load_dataset("parquet", data_files=files, split=split_spec)
        self.processor = processor
        self.max_target_length = max_target_length
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        image = coerce_to_pil(sample["image"])
        return encode_sample(image, str(sample["text"]), self.processor, self.max_target_length, self.preprocess)


class TrOCRSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with PEFT saving adjusted for VisionEncoderDecoder configs.

    Also overrides compute_loss: transformers 4.57's VisionEncoderDecoder
    routes `labels` through ForCausalLMLoss, which shifts targets a second
    time after decoder_input_ids were already shifted from labels. The
    double shift trains the model to predict the PREVIOUS token (base-model
    teacher-forced loss reads 15.75 where correct alignment reads 0.45) and
    collapses generation. We build decoder_input_ids ourselves and compute
    cross-entropy against unshifted labels.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        cfg_model = model
        while hasattr(cfg_model, "module"):  # unwrap DataParallel/DDP
            cfg_model = cfg_model.module
        config = cfg_model.config  # PeftModel forwards .config to the base model
        pad_id = config.pad_token_id
        start_id = config.decoder_start_token_id

        decoder_input_ids = labels.new_full(labels.shape, pad_id)
        decoder_input_ids[:, 1:] = labels[:, :-1].clone()
        decoder_input_ids[:, 0] = start_id
        decoder_input_ids[decoder_input_ids == -100] = pad_id

        outputs = model(
            pixel_values=inputs["pixel_values"],
            decoder_input_ids=decoder_input_ids,
        )
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[dict] = None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()

        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                save_embedding_layers=False,
            )
        elif hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        else:
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def coerce_to_pil(image_data: Any) -> Image.Image:
    if isinstance(image_data, Image.Image):
        return image_data.convert("RGB")
    if isinstance(image_data, dict) and "bytes" in image_data:
        return Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
    if isinstance(image_data, bytes):
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_data)!r}")


def encode_sample(
    image: Image.Image,
    text: str,
    processor: TrOCRProcessor,
    max_target_length: int,
    preprocess: str = "none",
) -> dict[str, torch.Tensor]:
    if preprocess == "app":
        # Crop/pad-to-square parity with the base checkpoint's training regime
        # (see backend/preprocessing.py). Raw wide strips mismatch the base
        # model's input distribution and destroy matra recognition.
        try:
            from backend.preprocessing import preprocess_pil_for_ocr
        except ImportError:
            from preprocessing import preprocess_pil_for_ocr
        image = preprocess_pil_for_ocr(image)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
    labels = processor.tokenizer(
        text,
        padding="max_length",
        max_length=max_target_length,
        truncation=True,
    ).input_ids
    labels = [
        label if label != processor.tokenizer.pad_token_id else -100
        for label in labels
    ]
    return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


def build_datasets(args: argparse.Namespace, processor: TrOCRProcessor):
    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
        train_dataset = ExtractedDevanagariDataset(
            data_dir / "train",
            processor,
            args.max_target_length,
            limit=args.train_limit,
            preprocess=args.preprocess,
        )
        eval_dataset = ExtractedDevanagariDataset(
            data_dir / "validation",
            processor,
            args.max_target_length,
            limit=args.eval_limit,
            preprocess=args.preprocess,
        )
    elif args.parquet_dir:
        parquet_dir = Path(args.parquet_dir).expanduser().resolve()
        train_dataset = ParquetDevanagariDataset(
            parquet_dir,
            "train",
            processor,
            args.max_target_length,
            limit=args.train_limit,
            preprocess=args.preprocess,
        )
        eval_dataset = ParquetDevanagariDataset(
            parquet_dir,
            "validation",
            processor,
            args.max_target_length,
            limit=args.eval_limit,
            preprocess=args.preprocess,
        )
    else:
        train_dataset = HuggingFaceDevanagariDataset(
            "train",
            processor,
            args.max_target_length,
            limit=args.train_limit,
            preprocess=args.preprocess,
        )
        eval_dataset = HuggingFaceDevanagariDataset(
            "validation",
            processor,
            args.max_target_length,
            limit=args.eval_limit,
            preprocess=args.preprocess,
        )

    if args.synthetic_dir:
        synthetic_dataset = ExtractedDevanagariDataset(
            Path(args.synthetic_dir).expanduser().resolve(),
            processor,
            args.max_target_length,
            limit=args.synthetic_limit,
            preprocess=args.preprocess,
        )
        print(f"Mixing {len(synthetic_dataset)} synthetic samples into {len(train_dataset)} real ones")
        train_dataset = ConcatDataset([train_dataset, synthetic_dataset])

    return train_dataset, eval_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR LoRA for Devanagari OCR.")
    parser.add_argument("--base-model", default=os.getenv("TROCR_BASE_MODEL", MODEL_NAME))
    parser.add_argument("--output-dir", default=os.getenv("TROCR_ADAPTER_ROOT", "./trocr-devanagari-lora"))
    parser.add_argument("--data-dir", default=None, help="Optional extracted dataset directory containing train/validation.")
    parser.add_argument("--parquet-dir", default=None, help="Optional local HF parquet directory containing train/validation/test parquet files.")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--generation-max-length", type=int, default=None,
                        help="Max generated tokens during validation/test; defaults to model generation config.")
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--eval-strategy", choices=["steps", "epoch", "no"], default="steps")
    parser.add_argument("--save-strategy", choices=["steps", "epoch", "no"], default="steps")
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--train-limit", type=int, default=None, help="Use a small subset for smoke tests.")
    parser.add_argument("--eval-limit", type=int, default=None, help="Use a small validation subset for smoke tests.")
    parser.add_argument("--preprocess", choices=["none", "app"], default="app",
                        help="app = crop/pad-to-square parity with the base checkpoint (recommended); "
                             "none = raw dataset images.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers (preprocessing is CPU-bound).")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    # LoRA ablation knobs (Paper 1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=None, help="Defaults to 2 * lora_r.")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        choices=sorted(TARGET_MODULE_PRESETS),
        default="legacy",
        help="legacy = shipped-adapter config; attn = attention projections only; attn-ffn = attention + FFN.",
    )
    parser.add_argument("--full-finetune", action="store_true", help="Train all weights (no LoRA baseline).")
    # Synthetic data augmentation (Paper 1)
    parser.add_argument("--synthetic-dir", default=None,
                        help="Directory with images/ + labels.csv from generate_synthetic.py, mixed into training.")
    parser.add_argument("--synthetic-limit", type=int, default=None,
                        help="Cap synthetic samples (for 10%%/25%%/50%% mixing ratios).")
    parser.add_argument("--eval-test", action="store_true",
                        help="After training, evaluate on the official test split and write test_metrics.json.")
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="Resume Trainer state from a checkpoint directory, e.g. output/checkpoint-4366.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_best_torch_device() if args.device == "auto" else args.device
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    print(f"Initializing TrOCR {'full fine-tune' if args.full_finetune else 'LoRA'} training on {device}")
    processor = load_trocr_processor(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    generation_max_length = args.generation_max_length or model.generation_config.max_length or args.max_target_length
    model.generation_config.max_length = generation_max_length
    model.generation_config.num_beams = 4

    if not args.full_finetune:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r,
            lora_dropout=args.lora_dropout,
            target_modules=TARGET_MODULE_PRESETS[args.target_modules],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")

    train_dataset, eval_dataset = build_datasets(args, processor)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    def compute_metrics(pred):
        labels_ids = pred.label_ids.copy()
        pred_ids = pred.predictions
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer_sum = 0.0
        word_errors = 0
        valid_samples = 0
        for prediction, label in zip(pred_str, labels_str):
            prediction = unicodedata.normalize("NFC", prediction.strip())
            label = unicodedata.normalize("NFC", label.strip())
            if label:
                cer_sum += editdistance.eval(prediction, label) / len(label)
                word_errors += int(prediction != label)
                valid_samples += 1

        if not valid_samples:
            return {"cer": 0.0, "wer": 0.0}
        return {"cer": cer_sum / valid_samples, "wer": word_errors / valid_samples}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        predict_with_generate=True,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        remove_unused_columns=False,
        fp16=device == "cuda",
        dataloader_pin_memory=device == "cuda",
        dataloader_num_workers=args.num_workers,
        save_total_limit=3,
        load_best_model_at_end=args.eval_strategy != "no" and args.save_strategy != "no",
        metric_for_best_model="cer",
        greater_is_better=False,
        generation_max_length=generation_max_length,
        report_to="none",
    )

    trainer = TrOCRSeq2SeqTrainer(
        model=model,
        processing_class=processor.image_processor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    run_config = {
        "seed": args.seed,
        "base_model": args.base_model,
        "preprocess": args.preprocess,
        "full_finetune": args.full_finetune,
        "lora_r": None if args.full_finetune else args.lora_r,
        "lora_alpha": None if args.full_finetune else (args.lora_alpha or 2 * args.lora_r),
        "lora_dropout": None if args.full_finetune else args.lora_dropout,
        "target_modules": None if args.full_finetune else args.target_modules,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_target_length": args.max_target_length,
        "generation_max_length": generation_max_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "eval_strategy": args.eval_strategy,
        "save_strategy": args.save_strategy,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "synthetic_dir": args.synthetic_dir,
        "synthetic_limit": args.synthetic_limit,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, ensure_ascii=False, indent=2)
    print(f"Training complete. Saved {'model' if args.full_finetune else 'LoRA adapter'} to {args.output_dir}")

    if args.eval_test:
        print("Evaluating on the official test split...")
        if args.parquet_dir:
            test_dataset = ParquetDevanagariDataset(
                Path(args.parquet_dir).expanduser().resolve(),
                "test",
                processor,
                args.max_target_length,
                limit=args.eval_limit,
                preprocess=args.preprocess,
            )
        else:
            test_dataset = HuggingFaceDevanagariDataset(
                "test", processor, args.max_target_length,
                limit=args.eval_limit, preprocess=args.preprocess,
            )
        test_results = trainer.predict(test_dataset)
        test_metrics = {**test_results.metrics, "num_samples": len(test_dataset)}
        with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(test_metrics, fh, ensure_ascii=False, indent=2)
        print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
