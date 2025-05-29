# torchrun --nproc_per_node=4 process_bench.py --model <model_path> -b 24 -w 4 -s "\n"
import argparse
import json
import os
import random
from copy import deepcopy

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm


def collate_fn(batch, tokenizer, separator = '\n'):
    input_ids = []
    score_ids = []
    labels = []
    separator_ids = tokenizer.encode(separator, add_special_tokens=False, return_tensors='pt')
    for i in batch:
        prompt_ids = tokenizer(i['problem'], add_special_tokens=False, return_tensors='pt')['input_ids']
        score_ids.append([])
        for completion in i['steps']:
            completion_ids = tokenizer(completion, add_special_tokens=False, return_tensors='pt')['input_ids']
            prompt_ids = torch.cat([prompt_ids, completion_ids, separator_ids], dim=-1)
            score_ids[-1].append(prompt_ids.size(-1) - 1)
        labels.append(i['label'])
        input_ids.append(prompt_ids)
    
    # right pad input_ids
    pad_token_id = tokenizer.pad_token_id
    max_len = max([i.size(-1) for i in input_ids])
    for i, input_idx in enumerate(input_ids):
        input_ids[i] = torch.cat([
            input_idx.squeeze(), 
            torch.LongTensor(
                [pad_token_id] * (max_len - input_idx.size(-1))
            )
        ])
    input_ids = torch.stack(input_ids)

    return dict(
        input_ids=input_ids,
        labels=labels,
        score_ids=score_ids
    )
    
def find_first_zero(tensor):
    zeros = (tensor == 0).nonzero()
    return zeros[0].item() if zeros.numel() > 0 else -1

def gather_objects(data, accelerator):
    world_size = accelerator.num_processes
    if world_size == 1:
        return data
        
    all_data = [None] * world_size
    torch.distributed.all_gather_object(all_data, data)
    
    if accelerator.is_main_process:
        result = []
        for process_data in all_data:
            result.extend(process_data)
        return result
    return None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    bs = args.batch_size
    num_of_workers = args.num_of_workers
    separator = args.separator
    model_path = args.model
    temperature = args.temperature

    model_name = model_path.split('/')[-1]

    configs = {
        'gsm8k': [207, 193], # error / correct num
        'math': [594, 406], 
        'olympiadbench': [661, 339], 
        'omnimath': [759, 241],
    }
    all_f1_scores = []
    save_dir = f'outputs/{model_name}'
    os.makedirs(save_dir, exist_ok=True)

    accelerator = Accelerator()
    print(f'Loading model from {model_path}')
    model = transformers.AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = accelerator.prepare(model)
    model.eval()

    for config, num in configs.items():
        dataset = load_dataset("Qwen/ProcessBench", split=config)
        sampler = None
        if accelerator.distributed_type == "MULTI_GPU":
            sampler = DistributedSampler(
                dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=False,
            )
        dataloader = DataLoader(
            dataset, 
            batch_size=bs, 
            collate_fn=lambda x: x, 
            num_workers=num_of_workers,
            sampler=sampler,
            drop_last=False,
        )

        res_data = []
        for batch_ in tqdm(dataloader, disable=not accelerator.is_main_process):
            new_batch = deepcopy(batch_)

            batch = collate_fn(batch_, tokenizer, separator)
            input_ids = batch['input_ids'].to(accelerator.device)
            labels = batch['labels']
            score_ids = batch['score_ids']

            with accelerator.autocast(), torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
            
            for i, score_id in enumerate(score_ids):
                label = labels[i]
                prob = logits[i, score_id].softmax(dim=-1)  # (steps, 2)
                score = prob[:, 1] - prob[:, 0]  # (steps,)
                pred_or = ((-score / temperature).softmax(dim=-1) * score).sum()
                if (label != -1 and pred_or <= 0) or (label == -1 and pred_or > 0):
                    new_batch[i]['match'] = True
                else:
                    new_batch[i]['match'] = False
            
            res_data.extend(new_batch)
        
        accelerator.wait_for_everyone()
        gathered_data = gather_objects(res_data, accelerator)

        if accelerator.is_main_process:
            data1 = [e for e in gathered_data if e['label'] != -1]
            data2 = [e for e in gathered_data if e['label'] == -1]
            # dataset length check
            if len(data1) != num[0]:
                print(f'{config} error num mismatch: {len(data1)} != {num[0]}')
            if len(data2) != num[1]:
                print(f'{config} correct num mismatch: {len(data2)} != {num[1]}')
            
            acc1 = np.mean([e['match'] for e in data1]) * 100
            acc2 = np.mean([e['match'] for e in data2]) * 100
            f1 = 2 * acc1 * acc2 / (acc1 + acc2)
            print(f'{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')

            all_f1_scores.append(f1)

    if accelerator.is_main_process:
        print(f'ProcessBench. Average F1: {np.mean(all_f1_scores):.1f}')

    if accelerator.distributed_type == "MULTI_GPU":
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=24)
    parser.add_argument("-w", "--num_of_workers", type=int, default=4)
    parser.add_argument("-s", "--separator", type=str, default="\n", help="It's important to use the same separator as the one used during TRL training")
    parser.add_argument("-t", "--temperature", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(args)
