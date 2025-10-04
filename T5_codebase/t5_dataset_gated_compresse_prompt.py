import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

#------------------------------
# Clean version:
#   - Robust tokenize_function
#   - Always converts labels to string
#   - HuggingFace dataset returns torch tensors directly
#   - Adds `task_name` inside collate_fn
#------------------------------

class T5Dataset:
    def __init__(self, tokenizer, task_name):
        self.tokenizer = tokenizer
        self.task_name = task_name

    def tokenize_function(self, examples, max_length, target_len):
        # ---- Source text ----
        if "sentence1" in examples:
            source_texts = examples["sentence1"]
        elif "text" in examples:
            source_texts = examples["text"]
        elif "content" in examples:  # e.g., dbpedia_14
            source_texts = examples["content"]
        else:
            non_label_cols = [c for c in examples.keys() if c != "label"]
            if not non_label_cols:
                raise KeyError("No valid source text field found in dataset!")
            source_texts = examples[non_label_cols[0]]

        if "sentence2" in examples:
            source_pairs = [a + " " + b for a, b in zip(source_texts, examples["sentence2"])]
        else:
            source_pairs = source_texts

        source_encodings = self.tokenizer(
            source_pairs,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

        # ---- Target text (labels) ----
        if "label" in examples:
            labels = [str(l) for l in examples["label"]]
        else:
            labels = examples.get("target", ["0"] * len(source_pairs))

        target_encodings = self.tokenizer(
            labels,
            truncation=True,
            padding="max_length",
            max_length=target_len
        )

        return {
            "source_ids": source_encodings["input_ids"],
            "source_mask": source_encodings["attention_mask"],
            "target_ids": target_encodings["input_ids"],
            "target_mask": target_encodings["attention_mask"],
        }

    def collate_fn(self, batch):
        """Custom collation: stacks tensors and adds task_name field"""
        collated = {key: torch.stack([example[key] for example in batch]) for key in batch[0]}
        collated["task_name"] = self.task_name
        return collated

    def get_final_ds(self, task, batch_size=8, max_length=128, target_len=2,
                     k=-1, split="train", return_test=False, prefix_list=None):
        print(f"Loading dataset for task = {task}, split = {split}")

        try:
            dataset = load_dataset("glue", task)
        except Exception:
            print(f"Fallback: loading task '{task}' with default dataset loader")
            dataset = load_dataset(task)

        if split not in dataset:
            raise ValueError(f"Split {split} not found for task {task}")

        dataset_split = dataset[split]

        if k != -1 and k < len(dataset_split):
            dataset_split = dataset_split.shuffle(seed=42).select(range(k))

        tokenized_ds = dataset_split.map(
            lambda ex: self.tokenize_function(ex, max_length, target_len),
            batched=True,
            remove_columns=dataset_split.column_names
        )

        tokenized_ds.set_format(type="torch")

        # Wrap directly in DataLoader with collate_fn
        dl = DataLoader(tokenized_ds, batch_size=batch_size, shuffle=True,
                        collate_fn=self.collate_fn)

        if return_test:
            if "validation" in dataset and "test" in dataset:
                val_ds = dataset["validation"].map(
                    lambda ex: self.tokenize_function(ex, max_length, target_len),
                    batched=True,
                    remove_columns=dataset["validation"].column_names
                )
                test_ds = dataset["test"].map(
                    lambda ex: self.tokenize_function(ex, max_length, target_len),
                    batched=True,
                    remove_columns=dataset["test"].column_names
                )
                val_ds.set_format(type="torch")
                test_ds.set_format(type="torch")

                val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    collate_fn=self.collate_fn)
                test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                     collate_fn=self.collate_fn)
                return val_dl, test_dl
            else:
                return dl, DataLoader(tokenized_ds, batch_size=batch_size, shuffle=False,
                                      collate_fn=self.collate_fn)

        return dl