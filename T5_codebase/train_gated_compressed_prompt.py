import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse

from t5_continual_gated_compressed_prompt import T5ContinualLearner

#from t5_continual import T5ContinualLearner

#------------------------
# Changes made:
#   previous_prompts handled as list of tensors → safe saving at end.
#   Log messages now explicitly say “Gated Progressive Prompts”.
# 	Added safety checks (if hasattr(model, "prompt")).
# 	Updated saving logic → saves results_dict placeholder + prompt list.
#------------------------


def train_on_task(model, dataloader, optimizer, device, task_id):
    model.train()
    # fetch current task-specific prompt
    current_prompt = model.prompt if hasattr(model, "prompt") else None

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # forward now applies gating + compression internally
        outputs = model(input_embeds=input_ids,
                        current_prompt=current_prompt,
                        labels=labels)

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def continual_prompt_training(tasks, model, optimizer, device):
    for task_id, dataloader in enumerate(tasks):
        print(f"Training task {task_id} with Gated Progressive Prompts...")
        train_on_task(model, dataloader, optimizer, device, task_id)

        # freeze and store prompt for this task
        # prompts are still one-per-task, GatedPromptManager decides usage
        if hasattr(model, "prompt"):
            model.previous_prompts.append(model.prompt.detach())


def main(args):
    save_path = os.path.join(args.save_dir, args.save_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    task_list = args.task_list

    model_name = args.model_name
    continual_learner = T5ContinualLearner(model_name,
                                           task_list,
                                           batch_size=args.batch_size,
                                           select_k_per_class=args.select_k_per_class,
                                           prefix_len=args.prefix_len,
                                           freeze_weights=args.freeze_weights == 1,
                                           freeze_except=args.freeze_except,
                                           lr=args.lr,
                                           seq_len=args.seq_len,
                                           early_stopping=args.early_stopping == 1,
                                           prefix_MLP=args.prefix_MLP,
                                           prefix_path=args.prefix_path if args.prefix_path != '' else None,
                                           mlp_layer_norm=args.mlp_layer_norm == 1,
                                           bottleneck_size=args.bottleneck_size,
                                           get_test_subset=args.get_test_subset == 1,
                                           memory_perc=args.memory_perc
                                           )
    if args.get_test_subset == 0:
        print("Not creating test subset")

    # run continual prompt training
    optimizer = continual_learner.optimizer
    device = continual_learner.device
    tasks = [continual_learner.tasks_data_dict[t]['train'] for t in continual_learner.task_list]

    continual_prompt_training(tasks, continual_learner.model, optimizer, device)

    # ----------------------------
    # Save results and prompt pool
    # ----------------------------
    np.save(os.path.join(save_path, 'results_dict.npy'), {"dummy": "training complete"})

    if len(continual_learner.previous_prompts) > 0:
        prompt_array = [p.detach().cpu().numpy() for p in continual_learner.previous_prompts]
        np.save(os.path.join(save_path, 'prompts.npy'), prompt_array)
    else:
        print("No prompts to save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='NLP prompt training script in PyTorch'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        help='base directory of all models / features',
        default='/data/home/arazdai/T5_prompts/T5_continual/'
    )

    parser.add_argument(
        '--save_name',
        type=str,
        help='folder name to save',
        required=True
    )

    parser.add_argument(
        '--task_list',
        nargs='+',
        help='List of tasks for training',
        required=True
    )

    parser.add_argument(
        '--model_name',
        type=str,
        help='Name of the model used for training',
        default="t5-base"
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size',
        default=8
    )

    parser.add_argument(
        '--seq_len',
        type=int,
        help='Length of a single repeat (in #tokens)',
        default=512
    )

    parser.add_argument(
        '--prefix_len',
        type=int,
        help='Length of prompt (in #tokens)',
        default=10
    )

    parser.add_argument(
        '--prefix_path',
        type=str,
        help='path to a pre-trained progressive prefix',
        default=''
    )

    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
        default=0.3
    )

    parser.add_argument(
        '--memory_perc',
        type=float,
        help='Memory perc',
        default=0.01
    )

    parser.add_argument(
        '--select_k_per_class',
        type=int,
        help='Select k examples from each class (default -1)',
        default=-1
    )

    parser.add_argument(
        '--freeze_weights',
        type=int,
        help='Whether to freeze model weights',
        default=0
    )

    parser.add_argument(
        '--freeze_except',
        type=str,
        help='If freeze_weights==1, freeze all weights except those that contain this keyword',
        default='xxxxxxx'
    )

    parser.add_argument(
        '--get_test_subset',
        type=int,
        help='Whether to create a separate test split',
        default=1
    )

    parser.add_argument(
        '--early_stopping',
        type=int,
        help='If early_stopping==1, do early stopping based on val accuracy',
        default=1
    )

    parser.add_argument(
        '--prefix_MLP',
        type=str,
        help='Type of MLP reparametrization (if None - Lester original)',
        default='None'
    )

    parser.add_argument(
        '--mlp_layer_norm',
        type=int,
        help='Do layer norm in MLP',
        default=1
    )

    parser.add_argument(
        '--bottleneck_size',
        type=int,
        help='MLP bottleneck size',
        default=800
    )

    main(parser.parse_args())