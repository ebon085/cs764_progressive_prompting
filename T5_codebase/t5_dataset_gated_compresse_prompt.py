import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

#------------------------------
# Clean version + dbpedia verbalization:
#   - Robust tokenize_function with task-specific template for dbpedia_14
#   - Verbalized labels for dbpedia_14 (strings, not numbers)
#   - HuggingFace dataset returns torch tensors directly
#   - Adds `task_name` via collate_fn
#------------------------------

class T5Dataset:
    def __init__(self, tokenizer, task_name):
        """
        Wrapper for GLUE/SuperGLUE and custom datasets for T5 continual learning.
        Each batch includes a `task_name` field for gating & analysis.
        """
        self.tokenizer = tokenizer
        self.task_name = task_name

        # DBPedia label mapping (14 classes)
        self.dbpedia_labels = [
            "company", "educational institution", "artist", "athlete", "office holder",
            "mean of transportation", "building", "natural place", "village", "animal",
            "plant", "album", "film", "written work"
        ]

    def _map_label_to_text(self, label):
        """
        For DBPedia: convert numeric label → descriptive text.
        For other tasks: return as-is (string).
        """
        if self.task_name == "dbpedia_14":
            return self.dbpedia_labels[int(label)]
        return str(label)

    def tokenize_function(self, examples, max_length, target_len):
        """
        Tokenizes source and target for T5.
        """
        # -------------------
        # SOURCE TEXT
        # -------------------
        if self.task_name == "dbpedia_14":
            # DBPedia uses "content" field
            source_texts = examples["content"]
        elif "sentence1" in examples:
            source_texts = examples["sentence1"]
        else:
            source_texts = examples["text"]

        source_encodings = self.tokenizer(
            source_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

        # Handle tasks with a second sentence (e.g., MRPC, QQP)
        if "sentence2" in examples:
            second_encodings = self.tokenizer(
                examples["sentence2"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            for k in source_encodings:
                source_encodings[k] = [
                    (s1 + s2)[:max_length]
                    for s1, s2 in zip(source_encodings[k], second_encodings[k])
                ]

        # -------------------
        # TARGET TEXT
        # -------------------
        if "label" in examples:
            labels_as_text = [self._map_label_to_text(l) for l in examples["label"]]
        else:
            labels_as_text = examples["text"]

        target_encodings = self.tokenizer(
            labels_as_text,
            truncation=True,
            padding="max_length",
            max_length=target_len
        )

        model_inputs = {
            "source_ids": source_encodings["input_ids"],
            "source_mask": source_encodings["attention_mask"],
            "target_ids": target_encodings["input_ids"],
            "target_mask": target_encodings["attention_mask"],
        }
        return model_inputs

    def get_final_ds(self, task, batch_size=8, max_length=128, target_len=2,
                     k=-1, split="train", return_test=False, prefix_list=None):
        """
        Load dataset, tokenize, and return DataLoader.
        Adds `task_name` field to each batch for use in gating & analysis.
        """
        print(f"Loading dataset for task = {task}, split = {split}")

        try:
            dataset = load_dataset("glue", task)
        except Exception:
            print(f"Fallback: loading task '{task}' with default dataset loader")
            dataset = load_dataset(task)

        if split not in dataset:
            raise ValueError(f"Split {split} not found for task {task}")

        dataset_split = dataset[split]

        # sample k examples if requested
        if k != -1 and k < len(dataset_split):
            dataset_split = dataset_split.shuffle(seed=42).select(range(k))

        # tokenize
        tokenized_ds = dataset_split.map(
            lambda ex: self.tokenize_function(ex, max_length, target_len),
            batched=True,
            remove_columns=dataset_split.column_names
        )

        # Torch Dataset wrapper
        class TorchDataset(Dataset):
            def __init__(self, hf_dataset, task_name):
                self.dataset = hf_dataset
                self.task_name = task_name

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                item = {key: torch.tensor(val) for key, val in self.dataset[idx].items()}
                item["task_name"] = self.task_name
                return item

        torch_ds = TorchDataset(tokenized_ds, task)
        dl = DataLoader(torch_ds, batch_size=batch_size, shuffle=True)

        if return_test and split == "validation":
            return dl, DataLoader(torch_ds, batch_size=batch_size, shuffle=False)

        return dl