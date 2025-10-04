import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm
import time
import os
from itertools import cycle
from copy import deepcopy
from transformers import AdamW, T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import matthews_corrcoef, f1_score

# Use your updated dataset wrapper
import t5_dataset_gated_compresse_prompt as t5_dataset

#---------------------------
# Changes made to the original code:
# - Added Gated + Compressed Prompt Manager to select and compress previous prompts
# - Updated train_step_lester and validate methods to use the new prompt manager
# ---------------------------

# ---------------------------
# Device utility
# ---------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ---------------------------
# Gated + Compressed Prompt Manager
# ---------------------------
class GatedPromptManager(nn.Module):
    """
    Scores previous prompts, keeps Top-K, and compresses the rest into a single surrogate token [1, D].
    """
    def __init__(self, prompt_dim: int, top_k: int = 3):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(prompt_dim, 1)                # learned scorer
        self.compressor = nn.Linear(prompt_dim, prompt_dim) # compress remainder

    def forward(self, prompts_list, current_prompt):
        if prompts_list is None or len(prompts_list) == 0:
            return [], None, []

        scores = []
        for p in prompts_list:
            pooled = p.mean(dim=0)        # [D]
            score = self.gate(pooled)     # [1]
            scores.append(score)
        scores = torch.cat(scores, dim=0).squeeze(-1)  # [N] or scalar

        k = min(self.top_k, scores.numel())
        top_idx = torch.topk(scores, k=k).indices
        if top_idx.ndim == 0:  # handle single index
            top_idx = [top_idx.item()]
        else:
            top_idx = top_idx.tolist()

        selected_prompts = [prompts_list[i] for i in top_idx]

        remainder = [prompts_list[i] for i in range(len(prompts_list)) if i not in top_idx]
        if len(remainder) > 0:
            pooled = torch.stack([r.mean(dim=0) for r in remainder], dim=0).mean(dim=0)  # [D]
            compressed_token = self.compressor(pooled).unsqueeze(0)  # [1, D]
        else:
            compressed_token = None

        return selected_prompts, compressed_token, scores.detach().cpu().numpy().tolist()


class ResMLP(torch.nn.Module):
    def __init__(self, bottleneck_size, module_type='MLP1', emb_dimension=512, residual=True):
        super().__init__()
        device = get_device()

        if module_type == 'MLP1':
            self.module = nn.Sequential(
                nn.Linear(emb_dimension, bottleneck_size),
                nn.Tanh(),
                nn.Linear(bottleneck_size, emb_dimension),
            )
        elif module_type == 'MLP2':
            self.module = nn.Sequential(
                nn.Linear(emb_dimension, bottleneck_size),
                nn.ReLU(),
                nn.Linear(bottleneck_size, bottleneck_size // 2),
                nn.Tanh(),
                nn.Linear(bottleneck_size // 2, emb_dimension),
            )
        elif module_type == 'transformer':
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=emb_dimension, nhead=2, dropout=0.05
            ).to(device)
            self.module = nn.TransformerEncoder(self.encoder_layer, num_layers=2).to(device)
        else:
            raise ValueError(f"Unsupported module_type: {module_type}")

        self.residual = residual
        if self.residual:
            print('Using skip connection in MLP')

    def forward(self, inputs):
        return self.module(inputs) + inputs if self.residual else self.module(inputs)


class T5ContinualLearner:
    def __init__(self,
                 model_name,
                 task_list,
                 batch_size=8,
                 select_k_per_class=-1,
                 prefix_len=0,
                 prefix_path=None,
                 freeze_weights=True,
                 freeze_except='shared',
                 lr=0.3,
                 weight_decay=1e-5,
                 seq_len=512,
                 early_stopping=True,
                 prefix_MLP='None',
                 bottleneck_size=800,
                 mlp_lr=None,
                 mlp_layer_norm=False,
                 weight_decay_mlp=None,
                 get_test_subset=True,
                 memory_perc=0.0,
                 use_gating=True,
                 top_k=3):

        self.device = get_device()
        self.task_list = task_list
        self.batch_size = batch_size
        self.select_k_per_class = select_k_per_class
        self.seq_len = seq_len
        self.freeze_weights = freeze_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.use_gating = use_gating
        self.top_k = top_k
        self.task_to_target_len = {
            'rte': 5, 'mrpc': 5, 'sst2': 2, 'qqp': 5, 'cola': 5, 'qnli': 5, 'mnli': 5, 'stsb': 3,
            'wic': 2, 'boolq': 2, 'copa': 2, 'wsc': 3, 'wsc_bool': 2, 'cb': 5, 'multirc': 5, 'record': 10,
            'rte_superglue': 5,
            'imdb': 2,
            'ag_news': 2, 'yahoo_answers_topics': 5, 'dbpedia_14': 5, 'amazon': 2, 'yelp_review_full': 2,
        }

        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        if freeze_weights:
            print('Freezing weights')
            self.do_freeze_weights(except_condition=freeze_except)

        self.prefix_len = prefix_len
        if prefix_len > 0:
            self.model.prompt = nn.Parameter(
                torch.tensor(self.init_new_prompt(prefix_len), requires_grad=True)
            )
            self.previous_prompts = [] if prefix_path is None else [torch.tensor(np.load(prefix_path), requires_grad=False).to(self.device)]
            hidden_dim = self.model.encoder.embed_tokens.weight.shape[1]
            self.prompt_manager = GatedPromptManager(prompt_dim=hidden_dim, top_k=self.top_k).to(self.device)

        self.model.to(self.device)
        self.prefix_MLPs = None
        self.optimizer = self.get_optimizer(lr, weight_decay)

        if self.early_stopping:
            self.best_prompt = self.model.prompt.detach().cpu().numpy() if self.prefix_len > 0 else deepcopy(self.model.state_dict())
            self.best_acc = 0.0

        self.get_test_subset = get_test_subset
        self.tasks_data_dict = self.get_tasks_data_dict(memory_perc=memory_perc)

        self.log_dict = {"train_latency_ms": [], "val_latency_ms": [], "prefix_lengths": [], "gating_scores": []}

    def get_optimizer(self, lr, weight_decay):
        optimizer_grouped_parameters = [{"params": [p for _n, p in self.model.named_parameters()],
                                         "weight_decay": weight_decay, "lr": lr}]
        return AdamW(optimizer_grouped_parameters, eps=1e-8)

    def init_new_prompt(self, prompt_len):
        N = self.model.encoder.embed_tokens.weight.shape[0]
        prompt_weigths = []
        for _ in range(prompt_len):
            with torch.no_grad():
                j = np.random.randint(N)
                w = deepcopy(self.model.encoder.embed_tokens.weight[j].detach().cpu().numpy())
                prompt_weigths.append(w)
        return np.array(prompt_weigths)

    def progress_previous_prompts(self, task=None):
        new_prompt = torch.tensor(self.best_prompt, requires_grad=False).to(self.device) if self.early_stopping else self.model.prompt.detach()
        self.previous_prompts.insert(0, new_prompt)
        print('Updated gated progressive prompt pool, total =', len(self.previous_prompts))

    # ---------------------------
    # Early stopping helpers
    # ---------------------------
    def update_best_model(self, acc, task=None):
        """
        Update best prompt/model if validation accuracy improves.
        """
        if acc > self.best_acc:
            if self.prefix_len > 0:
                best_prompt = self.model.prompt
                self.best_prompt = best_prompt.detach().cpu().numpy()
            else:
                self.best_model = deepcopy(self.model.state_dict())
            self.best_acc = acc

    def restore_best_model(self):
        """
        Restore the best prompt or model after early stopping.
        """
        if self.prefix_len > 0:
            self.model.prompt = nn.Parameter(torch.tensor(self.best_prompt, requires_grad=True))
            self.model.to(self.device)
            print("Restored best prompt")
        else:
            self.model.load_state_dict(deepcopy(self.best_model))
            print("Restored best model")

    def get_tasks_data_dict(self, memory_perc=0):
        tasks_data_dict = {}
        for task in self.task_list:
            print(task)
            ds2 = t5_dataset.T5Dataset(self.tokenizer, task)
            dataloader_train = ds2.get_final_ds(task, batch_size=self.batch_size, max_length=self.seq_len,
                                                target_len=5, k=self.select_k_per_class, split="train")
            val_split = "validation" if task in ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli',
                                                 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte',
                                                 'wnli', 'ax', 'copa', 'boolq', 'wic', 'wsc',
                                                 'wsc_bool', 'cb', 'record', 'multirc',
                                                 'rte_superglue'] else "test"
            dataloaders = ds2.get_final_ds(task, batch_size=self.batch_size, max_length=self.seq_len,
                                           target_len=5, k=500, split=val_split, return_test=self.get_test_subset)
            tasks_data_dict[task] = {"train": dataloader_train}
            if self.get_test_subset:
                dataloader_val, dataloader_test = dataloaders
                tasks_data_dict[task]['val'] = dataloader_val
                tasks_data_dict[task]['test'] = dataloader_test
            else:
                tasks_data_dict[task]['val'] = dataloaders
        return tasks_data_dict

    # Perform one train step for prompt tuning (Lester-style) with Gated+Compressed PP
    def train_step_lester(self, batch, task=None, progressive=True):
        # --- safe batch to device ---
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        model = self.model
        tokenizer = self.tokenizer

        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

        inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])
        k = inputs_embeds.shape[0]
        prompt = self.model.prompt

        if progressive:
            if self.use_gating:
                selected, compressed, scores = self.prompt_manager(self.previous_prompts, prompt)
                # log gating scores for analysis
                self.log_dict["gating_scores"].append(scores)
                concat_prompts = [prompt.repeat(k, 1, 1)] + [p.repeat(k, 1, 1) for p in selected]
                if compressed is not None:
                    compressed_b = compressed.unsqueeze(0).repeat(k, 1, 1)  # [B, 1, D]
                    concat_prompts.append(compressed_b)
                all_prompts = torch.cat(concat_prompts, dim=1)
            else:
                # ORIGINAL PP baseline: concat just current prompt (or all prev if desired)
                if len(self.previous_prompts) > 0:
                    prev = torch.cat([p for p in self.previous_prompts], dim=0)
                    all_prompts = torch.cat([prompt.repeat(k, 1, 1), prev.repeat(k, 1, 1)], dim=1)
                else:
                    all_prompts = prompt.repeat(k, 1, 1)

            inputs_embeds = torch.cat([all_prompts, inputs_embeds], dim=1)[:, :self.seq_len]
            # log effective prefix length
            self.log_dict["prefix_lengths"].append(all_prompts.shape[1])
        else:
            inputs_embeds = torch.concat([prompt.repeat(k, 1, 1),
                                          inputs_embeds], axis=1)[:, :self.seq_len]

        source_mask_updated = torch.concat(
            (batch["source_mask"][0][0].repeat(k, inputs_embeds.shape[1]),
             batch["source_mask"]), axis=1
        )[:, :self.seq_len]

        t0 = time.perf_counter()
        encoder_outputs = model.encoder(
            attention_mask=source_mask_updated,
            inputs_embeds=inputs_embeds,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        outputs = model(
            input_ids=batch["source_ids"],
            attention_mask=source_mask_updated,
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
            encoder_outputs=encoder_outputs,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.log_dict["train_latency_ms"].append(latency_ms)
        loss = outputs[0]
        return loss

    # Perform one train step for full model training
    def train_step(self, batch):
        # --- safe batch to device ---
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        model = self.model
        tokenizer = self.tokenizer

        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

        inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])
        encoder_outputs = model.encoder(
            attention_mask=batch["source_mask"],
            inputs_embeds=inputs_embeds,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        outputs = model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
            encoder_outputs=encoder_outputs,
        )
        loss = outputs[0]
        return loss

    # Process string for validation (remove pad and end tokens)
    def normalize_text(self, s):
        import string, re
        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the|)\b", re.UNICODE)
            return re.sub(regex, " ", text)
        def white_space_fix(text):
            return " ".join(text.split())
        def remove_punc(text):
            text2 = text.replace('<pad>', '').replace('</s>', '')
            exclude = set(string.punctuation)
            return "".join(ch for ch in text2 if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_exact_match(self, prediction, truth):
        return int(self.normalize_text(prediction) == self.normalize_text(truth))

    def compute_f1(self, prediction, truth):
        pred_tokens = self.normalize_text(prediction).split()
        truth_tokens = self.normalize_text(truth).split()
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)
        common_tokens = set(pred_tokens) & set(truth_tokens)
        if len(common_tokens) == 0:
            return 0
        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)
        return 2 * (prec * rec) / (prec + rec)

    # Compute task metrics on a validation (test) set
    def validate(self, dataloader_val, task, prompt=None, target_len=2, print_outputs=False):
        """
        If a prompt is available, apply Gated+Compressed selection over self.previous_prompts.
        """
        model = self.model
        tokenizer = self.tokenizer
        model.eval()

        corr, total, f1 = 0, 0, 0
        y_true, y_pred = [], []
        latencies = []

        for _, batch in enumerate(tqdm(dataloader_val)):
            # --- safe batch to device ---
            batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            inputs_embeds = model.encoder.embed_tokens(batch["source_ids"]).to(self.device)
            k = inputs_embeds.shape[0]

            if self.prefix_len > 0:
                # decide "current" prompt for gating
                if prompt is not None:
                    curr_prompt = prompt
                elif len(self.previous_prompts) > 0:
                    curr_prompt = self.previous_prompts[0]  # most recent prompt
                else:
                    curr_prompt = None

                if curr_prompt is not None:
                    if self.use_gating:
                        selected, compressed, scores = self.prompt_manager(self.previous_prompts, curr_prompt)
                        self.log_dict["gating_scores"].append(scores)
                        concat_prompts = [curr_prompt.repeat(k, 1, 1)] + [p.repeat(k, 1, 1) for p in selected]
                        if compressed is not None:
                            compressed_b = compressed.unsqueeze(0).repeat(k, 1, 1)
                            concat_prompts.append(compressed_b)
                        all_prompts = torch.cat(concat_prompts, dim=1)
                    else:
                        if len(self.previous_prompts) > 0:
                            prev = torch.cat([p for p in self.previous_prompts], dim=0)
                            all_prompts = torch.cat([curr_prompt.repeat(k, 1, 1), prev.repeat(k, 1, 1)], dim=1)
                        else:
                            all_prompts = curr_prompt.repeat(k, 1, 1)

                    inputs_embeds = torch.cat([all_prompts, inputs_embeds], dim=1)[:, :self.seq_len]
                    full_prefix_len = all_prompts.shape[1]
                    source_mask_updated = torch.concat(
                        (batch["source_mask"][0][0].repeat(k, full_prefix_len),
                         batch["source_mask"]), axis=1
                    )[:, :self.seq_len]
                else:
                    source_mask_updated = batch["source_mask"]
            else:
                source_mask_updated = batch["source_mask"]

            t0 = time.perf_counter()
            encoder_outputs = model.encoder(
                attention_mask=source_mask_updated,
                inputs_embeds=inputs_embeds,
                head_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
            )

            outs = model.generate(
                input_ids=batch["source_ids"],
                attention_mask=source_mask_updated,
                encoder_outputs=encoder_outputs,
                max_length=target_len,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            dec = [tokenizer.decode(ids) for ids in outs]
            targets = [tokenizer.decode(ids) for ids in batch['target_ids']]

            if task in ['stsb', 'cola', 'cb', 'multirc']:
                row_true = [self.normalize_text(x) for x in targets]
                row_pred = [self.normalize_text(x) for x in dec]
                if task == 'stsb':
                    row_true = [float(x) if any(c.isalpha() for c in x) is False else 0.0 for x in row_true]
                    row_pred = [float(x) if any(c.isalpha() for c in x) is False else 0.0 for x in row_pred]
                y_true += row_true
                y_pred += row_pred

            elif task == 'record':
                for x, y in zip(dec, targets):
                    corr += max([self.compute_exact_match(x, yi) for yi in y.split(';')])
                    f1 += max([self.compute_f1(x, yi) for yi in y.split(';')])
                total += batch['source_ids'].shape[0]

            else:
                corr += np.sum([self.normalize_text(x) == self.normalize_text(y) for x, y in zip(dec, targets)])
                total += batch['source_ids'].shape[0]

        self.log_dict["val_latency_ms"].append(np.mean(latencies))

        if task == 'cola':
            return matthews_corrcoef(y_true, y_pred)
        elif task == 'stsb':
            return np.corrcoef(y_true, y_pred)[0, 1]
        elif task == 'cb':
            return np.mean(np.array(y_true) == np.array(y_pred)), f1_score(y_true, y_pred, average='macro')
        elif task == 'multirc':
            # simplified path for your runs (no multirc_idx handling)
            return f1_score(y_true, y_pred, average='micro')
        elif task == 'record':
            return corr / total, f1 / total
        return corr / total

    # Freeze model weights
    def do_freeze_weights(self, except_condition='shared'):
        model = self.model
        for name, param in model.named_parameters():
            if param.requires_grad and except_condition not in name:
                param.requires_grad = False

    # Create replay buffers for data replay in CL
    def create_memory_replay_generators(self, task, split='train_mem'):  # creating previous tasks memory buffers
        print('Creating generators for previous tasks ...')
        tasks_to_generators = {}
        curr_task_num = self.task_list.index(task)
        for idx in np.arange(curr_task_num):
            prev_task = self.task_list[idx]
            print(prev_task)
            tasks_to_generators[prev_task] = iter(self.tasks_data_dict[prev_task][split])
        return tasks_to_generators

    # Perform memory replay from past tasks
    def memory_replay(self, tasks_to_generators, progressive):
        print("Rehearsal on " + str((', ').join(list(tasks_to_generators))))
        for prev_task in tasks_to_generators:
            generator_mem1 = tasks_to_generators[prev_task]
            try:
                b = next(generator_mem1)
            except StopIteration:
                generator_mem1 = iter(self.tasks_data_dict[prev_task]['train_mem'])
                tasks_to_generators[prev_task] = generator_mem1
                b = next(generator_mem1)

            # --- safe batch to device ---
            b = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in b.items()}

            if self.prefix_len > 0:  # prompt tuning
                loss = self.train_step_lester(b,
                                              task=prev_task if self.prefix_MLPs is not None else None,
                                              progressive=progressive)
            else:
                loss = self.train_step(b)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    # Perform training on a single task
    def train_one_task(self,
                       task,
                       epochs=40,
                       progressive=True,
                       eval_every_N=1,
                       eval_on_all_tasks=False,
                       data_replay_freq=-1):

        print('task = ', task)
        if progressive:
            assert self.prefix_len > 0  # can only do progressive prompts when prompt tuning
            print('progressive prompts')
        if self.early_stopping:
            self.best_acc = 0.0  # re-setting best acc

        model = self.model

        with torch.no_grad():
            model.prompt = nn.Parameter(torch.tensor(self.init_new_prompt(self.prefix_len),
                                                     requires_grad=True))
            self.optimizer = self.get_optimizer(self.lr, self.weight_decay)
        model.to(self.device)
        target_len = self.task_to_target_len.get(task, 5)
        dataloader_train = self.tasks_data_dict[task]['train']
        dataloader_val = self.tasks_data_dict[task]['val']

        val_acc = []

        for epoch in range(epochs):
            print(epoch)
            model.train()

            if data_replay_freq != -1 and 'train_mem' in self.tasks_data_dict[task]:
                tasks_to_generators = self.create_memory_replay_generators(task, split='train_mem')

            for i, batch in enumerate(tqdm(dataloader_train)):
                # --- safe batch to device ---
                batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

                if self.prefix_len > 0:  # prompt tuning
                    loss = self.train_step_lester(batch,
                                                  task=task if self.prefix_MLPs is not None else None,
                                                  progressive=progressive)
                else:
                    loss = self.train_step(batch)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # performing data replay on all previous tasks
                if data_replay_freq != -1 and i % data_replay_freq == 0 and 'train_mem' in self.tasks_data_dict[task]:
                    self.memory_replay(tasks_to_generators, progressive)

            # Build prompt for evaluation
            prompt = model.prompt if self.prefix_len > 0 else None

            # evaluate accuracy after each epoch
            if epoch % eval_every_N == 0:
                overall_acc = []
                if eval_on_all_tasks:
                    for eval_task in self.task_list:
                        acc = self.validate(self.tasks_data_dict[eval_task]['val'],
                                            eval_task,
                                            prompt=prompt if prompt is not None else None,
                                            target_len=self.task_to_target_len.get(eval_task, 5),
                                            print_outputs=False)
                        overall_acc.append(np.mean(acc))
                        if eval_task == task:  # record val accuracy for the current task
                            val_acc.append(np.mean(acc))
                    acc = np.mean(overall_acc)
                else:
                    acc = self.validate(dataloader_val, task,
                                        prompt=prompt if prompt is not None else None,
                                        target_len=target_len, print_outputs=True)
                    if task in ['record', 'cb']:
                        acc = np.mean(acc)  # averaging 2 scores
                    val_acc.append(acc)

                if self.early_stopping:
                    self.update_best_model(acc, task=task)
                print(epoch, task, '->', val_acc[-1])

        if progressive:
            self.progress_previous_prompts(task=task)
        else:
            if self.early_stopping:
                self.restore_best_model()
        return val_acc

    # Train model continually
    def train_continual(self,
                        task_list,
                        epochs=40,
                        save_path=None,
                        progressive=True,
                        eval_every_N=1,
                        test_eval_after_every_task=False,  # only needed for methods with catastrophic forgetting
                        data_replay_freq=-1):
        results_dict = {}
        if self.get_test_subset:
            results_dict['test'] = {}

        for num, task in enumerate(task_list):
            eval_on_all_tasks = False if progressive or len(task_list) == 1 else True
            eval_frq = eval_every_N if not eval_on_all_tasks else int(epochs // 3)
            val_acc = self.train_one_task(task, epochs,
                                          progressive=progressive,
                                          eval_every_N=eval_frq,
                                          data_replay_freq=data_replay_freq,
                                          eval_on_all_tasks=eval_on_all_tasks)
            print(task, val_acc)
            results_dict[task] = val_acc

            print('Calculating test acc ...')
            if self.get_test_subset:
                if progressive and len(self.previous_prompts) > 0:
                    # use the most recent prompt as the "current" one for gating at test time
                    curr_prompt = self.previous_prompts[0].detach()
                else:
                    curr_prompt = self.model.prompt if self.prefix_len > 0 else None

                if test_eval_after_every_task:
                    # eval test accuracy for all tasks
                    results_dict['test'][num] = {}
                    for test_task in task_list:
                        acc = self.validate(self.tasks_data_dict[test_task]['test'],
                                            test_task,
                                            curr_prompt,
                                            self.task_to_target_len.get(test_task, 5),
                                            print_outputs=True)
                        results_dict['test'][num][test_task] = acc

                else:
                    acc = self.validate(self.tasks_data_dict[task]['test'],
                                        task,
                                        curr_prompt,
                                        self.task_to_target_len.get(task, 5),
                                        print_outputs=True)
                    results_dict['test'][task] = acc

            # saving results dict and logs after each task
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                np.save(os.path.join(save_path, 'results_dict.npy'), results_dict)
                np.save(os.path.join(save_path, 'logs.npy'), self.log_dict, allow_pickle=True)

        return results_dict

    # Perform multi-task training
    def multi_task_training(self, num_epochs=5, progressive=False, save_path=''):
        tasks_data_dict = self.tasks_data_dict
        # getting index of the largest dataset (other datasets will be cycled)
        task_lengths = [len(tasks_data_dict[t]['train']) * self.batch_size for t in list(tasks_data_dict)]
        idx_biggest_task = np.argmax(task_lengths)
        n_tasks = len(list(tasks_data_dict))

        results_dict = {'test': {}}

        for epoch in range(num_epochs):
            print(epoch)

            dataloaders_list = [tasks_data_dict[t]['train'] if j == idx_biggest_task else cycle(tasks_data_dict[t]['train'])
                                for j, t in enumerate(tasks_data_dict)]
            mlt_dataloader = zip(*dataloaders_list)

            max_task = np.max([len(tasks_data_dict[t]['train']) for t in list(tasks_data_dict)])
            pbar = tqdm(total=max_task)

            for i, batch_combined in enumerate(mlt_dataloader):
                loss_combined = 0

                for task_num in range(n_tasks):
                    # --- safe batch to device ---
                    batch = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                             for k, v in batch_combined[task_num].items()}

                    if self.prefix_len > 0:  # prompt tuning
                        loss = self.train_step_lester(batch,
                                                      task=list(tasks_data_dict)[task_num] if self.prefix_MLPs is not None else None,
                                                      progressive=progressive)
                    else:
                        loss = self.train_step(batch)

                    loss_combined += loss

                loss_combined.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                pbar.update(1)

            results_dict['test'][epoch] = {}
            curr_prompt = self.model.prompt if self.prefix_len > 0 else None
            for test_task in self.task_list:
                acc = self.validate(self.tasks_data_dict[test_task]['test'],
                                    test_task,
                                    curr_prompt,
                                    self.task_to_target_len.get(test_task, 5),
                                    print_outputs=True)
                results_dict['test'][epoch][test_task] = acc

            if save_path != '':
                os.makedirs(save_path, exist_ok=True)
                np.save(os.path.join(save_path, 'results_dict.npy'), results_dict)
                np.save(os.path.join(save_path, 'logs.npy'), self.log_dict, allow_pickle=True)
            pbar.close()

        return results_dict