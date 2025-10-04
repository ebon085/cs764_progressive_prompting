import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse

import t5_dataset_old
from itertools import cycle
from copy import deepcopy
from transformers import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import matthews_corrcoef, f1_score


# ---------------------------
# Device utility
# ---------------------------
def get_device():
    """
    Choose the best available device:
    - Apple Silicon GPU via MPS if available
    - CUDA if available
    - CPU otherwise
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class ResMLP(torch.nn.Module):
    def __init__(self,
                 bottleneck_size,
                 module_type='MLP1',
                 emb_dimension=512,
                 residual=True):
        """MLP class for soft prompt re-parameterization. Can have a residual connection.

        Args:
            bottleneck_size (int): Dimension of the MLP bottleneck.
            module_type (str): 'MLP1', 'MLP2', or 'transformer'.
            emb_dimension (int): Embedding dimension (e.g., 512 for T5-small).
            residual (bool): Whether to use residual (skip) connection.
        """
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
        if self.residual:
            return self.module(inputs) + inputs
        else:
            return self.module(inputs)


class T5ContinualLearner:
    def __init__(self,
                 model_name,
                 task_list,
                 batch_size=8,
                 select_k_per_class=-1,
                 prefix_len=0,
                 prefix_path=None,  # path to the pre-trained progressive prompt
                 freeze_weights=True,
                 freeze_except='shared',
                 lr=0.3,
                 weight_decay=1e-5,
                 seq_len=512,
                 early_stopping=True,
                 prefix_MLP='None',
                 bottleneck_size=800,  # bottleneck size in case of using MLP reparametrization
                 mlp_lr=None,
                 mlp_layer_norm=False,  # (kept for interface compatibility; not used)
                 weight_decay_mlp=None,
                 get_test_subset=True,
                 memory_perc=0.0,
                 ):

        self.device = get_device()

        self.glue_datasets = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli',
                              'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']
        self.superglue_datasets = ['copa', 'boolq', 'wic', 'wsc', 'wsc_bool', 'cb',
                                   'record', 'multirc', 'rte_superglue']
        self.task_to_target_len = {
            'rte': 5,
            'mrpc': 5,
            'sst2': 2,
            'qqp': 5,
            'cola': 5,
            'qnli': 5,
            'mnli': 5,
            'stsb': 3,

            'wic': 2,
            'boolq': 2,
            'copa': 2,
            'wsc': 3,
            'wsc_bool': 2,
            'cb': 5,
            'multirc': 5,
            'record': 10,
            'rte_superglue': 5,

            'imdb': 2,

            'ag_news': 2,
            'yahoo_answers_topics': 5,
            'dbpedia_14': 5,
            'amazon': 2,
            'yelp_review_full': 2,
        }
        self.task_list = task_list

        self.freeze_weights = freeze_weights
        self.lr = lr
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.select_k_per_class = select_k_per_class
        self.early_stopping = early_stopping

        self.model_name = model_name  # e.g., "t5-large"
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Freezing model weights for prompt tuning
        if freeze_weights:
            print('Freezing weights')
            self.do_freeze_weights(except_condition=freeze_except)

        self.prefix_len = prefix_len

        # Creating a trainable soft prompt
        if prefix_len > 0:
            self.model.prompt = nn.Parameter(
                torch.tensor(self.init_new_prompt(prefix_len), requires_grad=True)
            )
            if prefix_path is None:
                self.previous_prompts = torch.zeros(
                    [0, self.model.prompt.shape[1]],
                    requires_grad=False
                ).to(self.device)
            else:  # initializing previous prompts from the path
                print('Using pre-trained progressive prompt - ' + prefix_path)
                self.previous_prompts = torch.tensor(
                    np.load(prefix_path), requires_grad=False
                ).to(self.device)

        # Model to device
        self.model.to(self.device)

        # Create MLP (if prompt re-parameterization is requested)
        self.get_MLP(prefix_MLP, bottleneck_size)  # adds prompt MLP reparametrization (and puts to device)

        self.lr = lr
        self.weight_decay = weight_decay
        self.mlp_lr = mlp_lr
        self.weight_decay_mlp = weight_decay_mlp
        self.optimizer = self.get_optimizer(
            lr, weight_decay,
            task=self.task_list[0],
            mlp_lr=mlp_lr,
            weight_decay_mlp=weight_decay_mlp
        )

        # Create best prompt/model copy for early stopping
        if self.early_stopping:
            if self.prefix_len > 0:
                # prompt tuning
                self.best_prompt = self.model.prompt.detach().cpu().numpy()
            else:
                # model tuning
                self.best_model = deepcopy(self.model.state_dict())  # saving best model
            self.best_acc = 0.0  # best avg accuracy on seen tasks

        # Get task -> data dictionary for CL training
        self.get_test_subset = get_test_subset
        self.tasks_data_dict = self.get_tasks_data_dict(memory_perc=memory_perc)

    # Create optimizer
    def get_optimizer(self, lr, weight_decay,
                      task=None, mlp_lr=None, weight_decay_mlp=None):  # task is used for MLP

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },

            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
        ]

        if task is not None and self.prefix_MLPs is not None:
            if weight_decay_mlp is None:
                weight_decay_mlp = weight_decay
            if mlp_lr is None:
                mlp_lr = lr

            optimizer_grouped_parameters.append({
                "params": [p for _n, p in self.prefix_MLPs[task].named_parameters()],
                "weight_decay": weight_decay_mlp,
                "lr": mlp_lr,
            })
        optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
        return optimizer

    # Create MLP for prompt tuning
    def get_MLP(self, prefix_MLP, bottleneck_size):
        if prefix_MLP == 'None':
            self.prefix_MLPs = None
        else:
            print('Using MLP reparametrization with bottleneck = ', bottleneck_size)
            N = self.model.encoder.embed_tokens.weight.shape[1]
            self.prefix_MLPs = {t: ResMLP(bottleneck_size=bottleneck_size,
                                          module_type=prefix_MLP,
                                          emb_dimension=N)
                                for t in self.task_list}
        if self.prefix_MLPs is not None:
            for t in self.task_list:
                self.prefix_MLPs[t].to(self.device)

    # Initialize new task prompt from random vocab tokens
    def init_new_prompt(self, prompt_len):
        model = self.model
        N = model.encoder.embed_tokens.weight.shape[0]
        prompt_weigths = []

        for _ in range(prompt_len):
            with torch.no_grad():
                j = np.random.randint(N)  # random token
                w = deepcopy(model.encoder.embed_tokens.weight[j].detach().cpu().numpy())
                prompt_weigths.append(w)
        prompt_weigths = np.array(prompt_weigths)
        return prompt_weigths

    # Concatenate newly learned prompt to the joint "Progressive Prompts"
    def progress_previous_prompts(self, task=None):
        if self.early_stopping:  # use best val acc prompt & MLP
            new_prompt = self.best_prompt  # prompt has already passed MLP
        else:  # use last prompt
            if task is not None and self.prefix_MLPs is not None:
                new_prompt = self.prefix_MLPs[task](self.model.prompt)
            else:
                new_prompt = self.model.prompt
            new_prompt = new_prompt.detach().cpu().numpy()

        new_prompt = torch.tensor(new_prompt, requires_grad=False).to(self.device)
        self.previous_prompts = torch.concat([new_prompt, self.previous_prompts], axis=0)
        print('Updated progressive prompts ', self.previous_prompts.shape)

    # Update best prompt/model based on val score
    def update_best_model(self, acc, task=None):
        if acc > self.best_acc:
            # getting best prompt
            if self.prefix_len > 0:
                best_prompt = self.model.prompt
                if self.prefix_MLPs is not None:
                    self.prefix_MLPs[task].eval()
                    best_prompt = self.prefix_MLPs[task](best_prompt)
                self.best_prompt = best_prompt.detach().cpu().numpy()

            # getting best model
            else:
                self.best_model = deepcopy(self.model.state_dict())  # saving best model
            self.best_acc = acc  # best avg accuracy on seen tasks

    # Retrieve best-performing model (for early stopping)
    def restore_best_model(self):
        if self.prefix_len > 0:
            self.model.prompt = nn.Parameter(torch.tensor(self.best_prompt, requires_grad=True))
            self.model.to(self.device)
            print("restored best prompt")
        else:
            self.model.load_state_dict(deepcopy(self.best_model))
            print("restored best model")

    # Create Dictionary of task_name -> dataloader (for CL experiments)
    def get_tasks_data_dict(self, memory_perc=0):
        tasks_data_dict = {}

        for task in self.task_list:
            tasks_data_dict[task] = {}
            print(task)
            data_params = {'task': task,
                           'batch_size': self.batch_size,
                           'max_length': self.seq_len,
                           'target_len': self.task_to_target_len[task],
                           'prefix_list': [],  # we are using vector prefix (instead of tokenization)
                           }
            ds2 = t5_dataset.T5Dataset(self.tokenizer, task)
            if task not in ['mrpc', 'cola', 'copa', 'rte', 'rte_superglue', 'cb', 'wsc', 'wsc_bool']:
                k = self.select_k_per_class
                k_val = max(500, int(0.2 * k)) if task != 'sst2' else 400
            else:
                k = self.select_k_per_class if (self.select_k_per_class <= 500 and task not in ['cb', 'copa', 'wsc', 'wsc_bool']) else -1
                k_val = -1
            if self.get_test_subset is False:
                k_val = -1  # use all val set
            dataloader_train = ds2.get_final_ds(**data_params, k=k, split='train')
            print('k = ', k, '  k-val = ', k_val)
            val_split = 'validation' if (task in self.glue_datasets) or (task in self.superglue_datasets) else 'test'
            dataloaders = ds2.get_final_ds(**data_params, k=k_val,
                                           split=val_split, return_test=self.get_test_subset)

            tasks_data_dict[task]['train'] = dataloader_train

            if memory_perc > 0:
                k_mem = max(1, int(len(dataloader_train) * self.batch_size * memory_perc))
                dataloader_mem = ds2.get_final_ds(**data_params, k=k_mem, split='train')
                tasks_data_dict[task]['train_mem'] = dataloader_mem

            if self.get_test_subset:
                dataloader_val, dataloader_test = dataloaders[0], dataloaders[1]
                tasks_data_dict[task]['val'] = dataloader_val
                tasks_data_dict[task]['test'] = dataloader_test
            else:
                tasks_data_dict[task]['val'] = dataloaders

            if task == 'multirc' and k_val == -1:
                self.multirc_idx = ds2.multirc_idx  # saving multirc idx for later computation
            else:
                self.multirc_idx = None
        return tasks_data_dict

    # Perform one train step for prompt tuning (following Lester et al.)
    def train_step_lester(self,
                          batch,
                          task=None,
                          progressive=True):
        prefix_len = self.prefix_len
        model = self.model
        embed_prompt = self.prefix_MLPs is not None
        if embed_prompt:
            assert task is not None
            mlp = self.prefix_MLPs[task]
        tokenizer = self.tokenizer

        batch = {k: batch[k].to(self.device) for k in batch}
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

        inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])

        k = inputs_embeds.shape[0]
        if embed_prompt:
            prompt = mlp(model.prompt)
        else:
            prompt = model.prompt

        if progressive:
            inputs_embeds = torch.concat([prompt.repeat(k, 1, 1),
                                          self.previous_prompts.repeat(k, 1, 1),
                                          inputs_embeds], axis=1)[:, :self.seq_len]
            full_prefix_len = self.previous_prompts.shape[0] + prompt.shape[0]  # prefix including all previous tasks
        else:
            inputs_embeds = torch.concat([prompt.repeat(k, 1, 1),
                                          inputs_embeds], axis=1)[:, :self.seq_len]
            full_prefix_len = prompt.shape[0]

        source_mask_updated = torch.concat(
            (batch["source_mask"][0][0].repeat(k, full_prefix_len),
             batch["source_mask"]), axis=1
        )[:, :self.seq_len]

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
        loss = outputs[0]

        return loss

    # Perform one train step for full model training
    def train_step(self, batch):
        model = self.model
        tokenizer = self.tokenizer

        batch = {k: batch[k].to(self.device) for k in batch}
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
        """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
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

    # Compute EM score used for some SuperGLUE tasks
    def compute_exact_match(self, prediction, truth):
        return int(self.normalize_text(prediction) == self.normalize_text(truth))

    # Compute F1 score used for some GLUE & SuperGLUE tasks
    def compute_f1(self, prediction, truth):
        pred_tokens = self.normalize_text(prediction).split()
        truth_tokens = self.normalize_text(truth).split()

        # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)

        common_tokens = set(pred_tokens) & set(truth_tokens)

        # if there are no common tokens then f1 = 0
        if len(common_tokens) == 0:
            return 0

        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)

        return 2 * (prec * rec) / (prec + rec)

    # Compute task metrics on a validation (test) set
    def validate(self,
                 dataloader_val,
                 task,
                 prompt=None,
                 target_len=2,
                 print_outputs=False):
        model = self.model
        prefix_len = self.prefix_len
        max_length = target_len
        tokenizer = self.tokenizer
        model.eval()

        corr, total, f1 = 0, 0, 0
        y_true, y_pred = [], []

        for _, batch in enumerate(tqdm(dataloader_val)):
            batch = {k: batch[k].to(self.device) for k in batch}
            inputs_embeds = model.encoder.embed_tokens(batch["source_ids"]).to(self.device)

            if prompt is not None:
                k = inputs_embeds.shape[0]
                inputs_embeds = torch.concat([prompt.repeat(k, 1, 1),
                                              inputs_embeds], axis=1)[:, :self.seq_len]

                full_prefix_len = prompt.shape[0]  # prompt is inputted by user
                source_mask_updated = torch.concat(
                    (batch["source_mask"][0][0].repeat(k, full_prefix_len),
                     batch["source_mask"]), axis=1
                )[:, :self.seq_len]
            else:  # full model fine tuning, no prompt added
                source_mask_updated = batch["source_mask"]

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
                max_length=max_length,
            )
            dec = [tokenizer.decode(ids) for ids in outs]
            texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
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
                # multiple answers
                for x, y in zip(dec, targets):
                    corr += max([self.compute_exact_match(x, yi) for yi in y.split(';')])
                    f1 += max([self.compute_f1(x, yi) for yi in y.split(';')])
                total += batch['source_ids'].shape[0]

            else:
                corr += np.sum([self.normalize_text(x) == self.normalize_text(y) for x, y in zip(dec, targets)])
                total += batch['source_ids'].shape[0]

        if task == 'cola':
            return matthews_corrcoef(y_true, y_pred)

        elif task == 'stsb':
            return np.corrcoef(y_true, y_pred)[0, 1]

        elif task == 'cb':
            return np.mean(np.array(y_true) == np.array(y_pred)), f1_score(y_true, y_pred, average='macro')

        elif task == 'multirc':
            if self.multirc_idx is not None:
                em = []
                for idx in set(self.multirc_idx):
                    k = np.where(self.multirc_idx == idx)[0]
                    score = (np.array(y_true)[k] == np.array(y_pred)[k]).all()
                    em.append(score)
                return np.mean(em), f1_score(y_true, y_pred, average='micro')
            else:
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

    # Freeze / unfreeze MLPs for given tasks (when requires_grad==False then freezing)
    def freeze_unfreeze_mlps(self, tasks, requires_grad=False):
        assert self.prefix_MLPs is not None
        for t in tasks:
            for name, param in self.prefix_MLPs[t].named_parameters():
                if param.requires_grad != requires_grad:
                    param.requires_grad = requires_grad
                    param.grad = None  # remove old gradient

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
        # for each memory buffer in tasks_to_generators perform memory replay
        print("Rehearsal on " + str((', ').join(list(tasks_to_generators))))
        for prev_task in tasks_to_generators:
            generator_mem1 = tasks_to_generators[prev_task]
            try:
                # Samples the batch
                b = next(generator_mem1)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                generator_mem1 = iter(self.tasks_data_dict[prev_task]['train_mem'])
                tasks_to_generators[prev_task] = generator_mem1
                b = next(generator_mem1)

            b = {k: v.to(self.device) for k, v in b.items()}
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

        if self.prefix_MLPs is not None:
            print('Freezing all MLPs except for ', task)
            mlp = self.prefix_MLPs[task]
            self.freeze_unfreeze_mlps([x for x in self.task_list if x != task], requires_grad=False)
            self.freeze_unfreeze_mlps([task], requires_grad=True)  # unfreezing current task

        model = self.model

        with torch.no_grad():
            model.prompt = nn.Parameter(torch.tensor(self.init_new_prompt(self.prefix_len),
                                                     requires_grad=True))
            self.optimizer = self.get_optimizer(self.lr, self.weight_decay,
                                                task=task)
        model.to(self.device)
        target_len = self.task_to_target_len[task]
        dataloader_train = self.tasks_data_dict[task]['train']
        dataloader_val = self.tasks_data_dict[task]['val']

        val_acc = []

        for epoch in range(epochs):
            print(epoch)
            model.train()
            if self.prefix_MLPs is not None:
                mlp.train()

            if data_replay_freq != -1:
                tasks_to_generators = self.create_memory_replay_generators(task, split='train_mem')

            for i, batch in enumerate(tqdm(dataloader_train)):
                batch = {k: batch[k].to(self.device) for k in batch}

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
                if data_replay_freq != -1 and i % data_replay_freq == 0:
                    self.memory_replay(tasks_to_generators, progressive)

            # evaluate accuracy after each epoch
            if self.prefix_MLPs is not None:
                mlp.eval()
                prompt = mlp(model.prompt)
            else:
                if self.prefix_len > 0:
                    prompt = model.prompt
                    print(prompt.shape)
                else:
                    prompt = None
            if progressive and prompt is not None:
                prompt = torch.concat([prompt, self.previous_prompts], axis=0)

            if epoch % eval_every_N == 0:
                overall_acc = []
                if eval_on_all_tasks:
                    # eval current model/prompt on all tasks (for approaches that suffer from catastrophic forgetting)
                    for eval_task in self.task_list:
                        acc = self.validate(self.tasks_data_dict[eval_task]['val'],
                                            eval_task,
                                            prompt=prompt, target_len=self.task_to_target_len[eval_task],
                                            print_outputs=False)
                        overall_acc.append(np.mean(acc))
                        if eval_task == task:  # record val accuracy for the current task
                            val_acc.append(np.mean(acc))
                    acc = np.mean(overall_acc)
                else:
                    acc = self.validate(dataloader_val, task,
                                        prompt=prompt, target_len=target_len, print_outputs=True)
                    if task in ['record', 'cb'] or (task == 'multirc' and self.multirc_idx is not None):
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
                if progressive:
                    curr_prompt = torch.tensor(self.previous_prompts, requires_grad=False).to(self.device)
                else:
                    if self.prefix_len > 0:
                        curr_prompt = self.model.prompt
                    else:
                        curr_prompt = None

                if test_eval_after_every_task:
                    # eval test accuracy for all tasks
                    results_dict['test'][num] = {}
                    for test_task in task_list:
                        acc = self.validate(self.tasks_data_dict[test_task]['test'],
                                            test_task,
                                            curr_prompt,
                                            self.task_to_target_len[test_task],
                                            print_outputs=True)
                        results_dict['test'][num][test_task] = acc

                else:
                    acc = self.validate(self.tasks_data_dict[task]['test'],
                                        task,
                                        curr_prompt,
                                        self.task_to_target_len[task],
                                        print_outputs=True)
                    results_dict['test'][task] = acc
            # saving results dict after each task
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                np.save(os.path.join(save_path, 'results_dict.npy'), results_dict)

        return results_dict

    # Perform multi-task training
    def multi_task_training(self, num_epochs=5, progressive=False, save_path=''):
        tasks_data_dict = self.tasks_data_dict
        val_scores = {x: [] for x in list(tasks_data_dict)}
        # getting index of the largest dataset (other datasets will be cycled)
        task_lengths = [len(tasks_data_dict[t]['train']) * self.batch_size for t in list(tasks_data_dict)]
        idx_biggest_task = np.argmax(task_lengths)
        n_tasks = len(list(tasks_data_dict))

        results_dict = {'test': {}}
        device = self.device

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
                    batch = {k: v.to(device) for k, v in batch_combined[task_num].items()}
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
            curr_prompt = None
            for test_task in self.task_list:
                acc = self.validate(self.tasks_data_dict[test_task]['test'],
                                    test_task,
                                    curr_prompt,
                                    self.task_to_target_len[test_task],
                                    print_outputs=True)
                results_dict['test'][epoch][test_task] = acc

            if save_path != '':
                os.makedirs(save_path, exist_ok=True)
                np.save(os.path.join(save_path, 'results_dict.npy'), results_dict)
            pbar.close()

        return results_dict
