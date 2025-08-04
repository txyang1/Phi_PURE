# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version

import numpy as np
from scipy.special import softmax
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence


# Example 3:
# The question is: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
# The reasoning steps are:

# Originally, Leah had 32 chocolates. Her sister had 42. 
# So in total they had 32 + 42 = 74. 
# After eating 35, they had 74 - 35 = 39.
# The answer is: \\boxed{39}.<end_of_reasoning>


# Example 4:
# The question is: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
# The reasoning steps are:

# Jason started with 20 lollipops.
# Then he had 12 after giving some to Denny.
# So he gave Denny 20 - 12 = 8.
# The answer is: \\boxed{8}.<end_of_reasoning>


# Example 5:
# The question is: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
# The reasoning steps are:

# Shawn started with 5 toys. 
# If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. 
# The answer is: \\boxed{9}.<end_of_reasoning>


# Example 6:
# The question is: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
# The reasoning steps are:

# There were originally 9 computers.
# For each of 4 days, 5 more computers were added.
# So 5 * 4 = 20 computers were added. 9 + 20 is 29.
# The answer is: \\boxed{29}.<end_of_reasoning>


# Example 7:
# The question is: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
# The reasoning steps are:

# Michael started with 58 golf balls.
# After losing 23 on tuesday, he had 58 - 23 = 35.
# After losing 2 more, he had 35 - 2 = 33 golf balls.
# The answer is: \\boxed{33}.<end_of_reasoning>


# Example 8:
# The question is: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
# The reasoning steps are:

# Olivia had 23 dollars.
# 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.
# So she has 23 - 15 dollars left. 23 - 15 is 8.
# The answer is: \\boxed{8}.<end_of_reasoning>

GSM_COT_8_SHOT = """
Example 1:
The question is : There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
The reasoning steps are:

There are 15 trees originally. Then there were 21 trees after some more were planted. 
So there must have been 21 - 15 = 6.
The answer is: \\boxed{6}.<end_of_reasoning>

Example 2:
The question is: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
The reasoning steps are:

There are originally 3 cars. 
2 more cars arrive. 
3 + 2 = 5.
The answer is: \\boxed{5}.<end_of_reasoning>

""".strip()
DEFAULT_SYSTEM_PROMPT = (
    "Please solve the following problem step by step.\n"
    "When you reach the answer, please include the answer in the box format and finish the reasoning with <end_of_reasoning>.\n"
    "I will give you some examples for reference.\n"
    + GSM_COT_8_SHOT
)

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            seed = 42,# 加入seed，保证每次生成的结果一致
        )
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.model.path)

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    '''@torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

        response = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response.append(output.outputs[sample_id].token_ids)

        response = pad_2d_list_to_length(response, self.pad_token_id,
                                         max_length=self.config.response_length).to(idx.device)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)'''
 



    '''#rollout 成功版
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        多步 Phi-Decoding，batch 维度并行版
        """
        # —— 1. 超参数 ——
        beam_size       = int(kwargs.get("step_beam_size",   self.config.step_beam_size))
        num_rollout     = int(kwargs.get("num_rollout",      self.config.num_rollout))
        num_foresight   = int(kwargs.get("num_foresight",    self.config.num_foresight))
        threshold       = float(kwargs.get("threshold",      self.config.threshold))
        sigma_rate      = float(kwargs.get("sigma_rate",     self.config.sigma_rate))
        cluster_num     = int(kwargs.get("cluster_num",      self.config.cluster_num))
        temperature     = float(kwargs.get("temperature",    self.config.temperature))
        step_response_len = int(kwargs.get("step_response_length", self.config.step_response_length))
        response_len    = int(kwargs.get("response_length",  self.config.response_length))

        system_prompt = DEFAULT_SYSTEM_PROMPT

        # —— 2. 解码 raw prompts ——
        idx            = prompts.batch["input_ids"]         # [bs, Lp]
        device         = idx.device
        bs             = idx.size(0)
        raw_prompts    = self.tokenizer.batch_decode(idx, skip_special_tokens=True)

        # —— 3. SamplingParams for intermediate rollout ——
        base_sp = SamplingParams(
            max_tokens=step_response_len,
            logprobs=1,
            temperature=temperature,
            n=num_rollout,
            stop=["\n", "<end_of_reasoning>"]
        )

        # —— 4. 初始化 prev_steps/prev_values 为 [bs][beam_size] ——
        prev_steps  = [["" for _ in range(beam_size)] for _ in range(bs)]
        prev_values = [[0.0 for _ in range(beam_size)] for _ in range(bs)]

        # —— 5. 多步前瞻 (num_foresight) ——
        for depth in range(num_foresight):
            # —— 阶段1：并行构造所有 prompt ——
            all_inputs = []
            for b in range(bs):
                for k in range(beam_size):
                    prefix = (
                        #f"{system_prompt}\n\n"
                        f"User: {raw_prompts[b].strip()}\n"
                        f"Reasoning so far:\n{prev_steps[b][k]}"
                    )
                    #all_inputs += [prefix] * num_rollout
                    all_inputs.append(prefix)    # 不再 * num_rollout
                    

            prompt_ids = [self.tokenizer.encode(t, add_special_tokens=False)
                        for t in all_inputs]

            outs = self.inference_engine.generate(
                prompts=None,
                sampling_params=base_sp,
                prompt_token_ids=prompt_ids,
                use_tqdm=False
            )

            # # 扁平化收集 all_resp, all_lp
            # flat_R = bs * beam_size * num_rollout
            # all_resp = []
            # all_lp   = []
            # for i in range(flat_R):
            #     o    = outs[i].outputs[0]
            #     txt  = o.text.strip()
            #     lp   = o.cumulative_logprob / (len(o.token_ids) + 1e-8)
            #     all_resp.append(txt)
            #     all_lp.append(lp)

            # —— 扁平化收集 all_resp, all_lp
            all_resp = []
            all_lp   = []
            for out in outs:                     # outs 的长度 = bs * beam_size
                for o in out.outputs:            # 每个 out.outputs 的长度 = num_rollout
                    txt = o.text.strip()
                    lp  = o.cumulative_logprob / (len(o.token_ids) + 1e-8)
                    all_resp.append(txt)
                    all_lp.append(lp)

            # 计算 all_adv
            all_adv = []
            for b in range(bs):
                for k in range(beam_size):
                    base = (b*beam_size + k)*num_rollout
                    prev_v = prev_values[b][k]
                    for j in range(num_rollout):
                        all_adv.append(all_lp[base + j] - prev_v)

            # Stage2-4：对每个 b 做剪枝和选 beam
            new_steps  = [["" for _ in range(beam_size)] for _ in range(bs)]
            new_values = [[0.0 for _ in range(beam_size)] for _ in range(bs)]

            for b in range(bs):
                # slice for this batch element
                start = b * beam_size * num_rollout
                end   = start + beam_size * num_rollout
                resp_slice = all_resp[start:end]
                lp_slice   = all_lp[start:end]
                adv_slice  = np.array(all_adv[start:end])

                # 宽度剪枝 low_sigma
                mu, sigma = np.mean(lp_slice), np.std(lp_slice)
                keep = [i for i,v in enumerate(lp_slice) if v > mu - sigma_rate*sigma]
                if len(keep) < beam_size:
                    wts = np.exp(adv_slice/temperature)
                    wts /= wts.sum()
                    extra = np.random.choice(
                        len(adv_slice),
                        beam_size-len(keep),
                        replace=False,
                        p=wts
                    ).tolist()
                    keep += extra
                keep.sort()

                # 候选集合
                cand   = [resp_slice[i] for i in keep]
                adv_k  = adv_slice[keep]

                # # 聚类
                # X      = TfidfVectorizer().fit_transform(cand)
                # labels = KMeans(n_clusters=cluster_num).fit_predict(X)

                # # 计算簇比例
                # counts = np.bincount(labels, minlength=cluster_num)
                # cluster_ratios = counts / counts.sum()
                # cluster_weights = softmax(cluster_ratios[labels])

                # 计算增益权重
                adv_weights = softmax(adv_k/temperature)

                # 合并权重
                #combined_weights = 0.5*cluster_weights + 0.5*adv_weights
                combined_weights = adv_weights
                combined_weights /= combined_weights.sum()

                # 一次性抽 beam_size 条
                selected_idxs = np.random.choice(
                    len(keep), size=beam_size, replace=False, p=combined_weights
                ).tolist()
                picked = [keep[i] for i in selected_idxs]

                # 更新
                for k, sel in enumerate(picked):
                    origin_beam = sel // num_rollout
                    new_steps[b][k]  = prev_steps[b][origin_beam] + resp_slice[sel] + "\n"
                    new_values[b][k] = lp_slice[sel]

            prev_steps, prev_values = new_steps, new_values
        #print(f"prev_steps: {prev_steps}")###check prev_steps

        # —— 6. 最后一轮并行生成最终答案 ——(argmax)
        final_prompts = []
        for b in range(bs):
            # 选择最大的 prev_value 对应的 beam
            best_k = int(np.argmax(prev_values[b]))
            txt = (
                #f"{system_prompt}\n\n"
                f"User: {raw_prompts[b].strip()}\n"
                f"Reasoning so far:\n{prev_steps[b][best_k]}\n"
                f"Final step: Continue the reasoning above—include **all** intermediate steps—and write out the full reasoning process, ending with the answer in the format \\boxed{…} and finish the reasoning with <end_of_reasoning>."
            )
            #print(f"raw_prompts[{b}]:{raw_prompts[b]}")###check raw_prompts
            #print(f"prev_steps[{b}]:{prev_steps[b][best_k]}")###check prev_steps
            #print(f"final_prompt[{b}]:{txt}")###check final_prompt
            ids = self.tokenizer.encode(txt, add_special_tokens=False)
            final_prompts += [ids] * self.config.n

        final_sp = SamplingParams(
            max_tokens = response_len,
            logprobs   = 1,
            temperature= temperature,
            n          = 1,
            stop       = ["<end_of_reasoning>"]
        )
        final_outs = self.inference_engine.generate(
            prompts=None,
            sampling_params=final_sp,
            prompt_token_ids=final_prompts,
            use_tqdm=False
        )

        # 收集并返回
        responses = [ out.outputs[0].token_ids for out in final_outs ]
        resp_padded = pad_2d_list_to_length(
            responses, self.tokenizer.pad_token_id, max_length=response_len
        ).to(device)
        

        # # —— 6. 最后一轮并行生成最终答案 —— (beam search) batch problem
        # # 1) 构造 beam_size 个前缀，每个前缀只生成一次完整回答
        # final_prompts: List[List[int]] = []
        # for b in range(bs):
        #     for k in range(beam_size):
        #         prefix = (
        #             f"{system_prompt}\n\n"
        #             f"User: {raw_prompts[b].strip()}\n"
        #             f"Reasoning so far:\n{prev_steps[b][k]}"
        #         )
        #         ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        #         final_prompts.append(ids)

        # final_sp = SamplingParams(
        #     max_tokens = response_len,
        #     logprobs   = 1,
        #     temperature= temperature,
        #     n          = 1,
        #     stop       = ["<end_of_reasoning>"]
        # )
        # final_outs = self.inference_engine.generate(
        #     prompts=None,
        #     sampling_params=final_sp,
        #     prompt_token_ids=final_prompts,
        #     use_tqdm=False
        # )

        # # 2) 收集 M = bs * beam_size 条 response, logprob, advantage
        # all_resps = []
        # all_logps  = []
        # all_advs   = []
        # for idx, out in enumerate(final_outs):
        #     text = out.outputs[0].text.strip()
        #     lp   = out.outputs[0].cumulative_logprob / (len(out.outputs[0].token_ids) + 1e-8)
        #     b    = idx // beam_size      # batch 索引
        #     k    = idx % beam_size       # beam 索引
        #     adv  = lp - prev_values[b][k]

        #     all_resps.append(text)
        #     all_logps .append(lp)
        #     all_advs  .append(adv)

        # # 3) 针对每个 batch 元素 b，用簇+增益权重采样一个 k
        # selected_prefixes = [0] * bs
        # for b in range(bs):
        #     # slice 出这一组 beam_size 条完整回答
        #     start = b * beam_size
        #     end   = start + beam_size
        #     resps = all_resps[start:end]
        #     advs  = all_advs[start:end]

        #     # —— 聚类权重 —— 
        #     X = TfidfVectorizer().fit_transform(resps)
        #     labels = KMeans(n_clusters=cluster_num).fit_predict(X)
        #     counts = np.bincount(labels, minlength=cluster_num)
        #     size_ratios = counts[labels] / beam_size        # 每条属于的簇大小比例
        #     cluster_weights = softmax(size_ratios)          # 归一化

        #     # —— 增益权重 —— 
        #     adv_weights = softmax(np.array(advs) / temperature)

        #     # —— 合并权重 & 采样 —— 
        #     combined_weights = (cluster_weights + adv_weights) * 0.5
        #     combined_weights /= combined_weights.sum()
        #     sel_k = int(np.random.choice(beam_size, p=combined_weights))

        #     selected_prefixes[b] = sel_k

        # # 4) 将选出的前缀拼接对应回答，打包输出
        # responses = []
        # for b in range(bs):
        #     k = selected_prefixes[b]
        #     # pick 对应的 final_outs 索引
        #     idx = b * beam_size + k
        #     tok_ids = final_outs[idx].outputs[0].token_ids
        #     responses.append(tok_ids)

        # resp_padded = pad_2d_list_to_length(
        #     responses, self.tokenizer.pad_token_id, max_length=response_len
        # ).to(device)####

        # repeat 原 prompt tensors
        Bn = resp_padded.size(0)
        repeat_times = Bn // bs
        idx    = prompts.batch["input_ids"].repeat_interleave(repeat_times, dim=0)
        mask   = prompts.batch["attention_mask"].repeat_interleave(repeat_times, dim=0)
        pos_ids= prompts.batch["position_ids"].repeat_interleave(repeat_times, dim=0)
       
        seq = torch.cat([idx, resp_padded], dim=1)
        delta = torch.arange(1, response_len+1, device=device).unsqueeze(0).expand(Bn, -1)
        last_pos = pos_ids[:, -1:].expand(-1, response_len)
        pos_ids = torch.cat([pos_ids, last_pos+delta], dim=1)
        resp_attn = get_eos_mask(resp_padded, self.tokenizer.eos_token_id, mask.dtype)
        mask = torch.cat([mask, resp_attn], dim=1)

        # # 假设 seq 是一个 shape [Bn, L] 的 Tensor,检验结果
        # first_ids = seq[0].tolist()  
        # first_text = self.tokenizer.decode(first_ids, skip_special_tokens=True)
        # print(f"First sequence text: {first_text}")  # 打印第一个序列的文本

        batch = TensorDict({
            "prompts":        idx,
            "responses":      resp_padded, #response ids
            "input_ids":      seq, #complete input+response ids
            "attention_mask": mask,
            "position_ids":   pos_ids,
        }, batch_size=Bn)

        return DataProto(batch=batch)'''

    '''#rollout 拼接prompts和responses版 成功,但是没有提取combined_weights
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        多步 Phi-Decoding，batch 维度并行版
        """
        # —— 1. 超参数 ——
        beam_size       = int(kwargs.get("step_beam_size",   self.config.step_beam_size))
        num_rollout     = int(kwargs.get("num_rollout",      self.config.num_rollout))
        num_foresight   = int(kwargs.get("num_foresight",    self.config.num_foresight))
        threshold       = float(kwargs.get("threshold",      self.config.threshold))
        sigma_rate      = float(kwargs.get("sigma_rate",     self.config.sigma_rate))
        cluster_num     = int(kwargs.get("cluster_num",      self.config.cluster_num))
        temperature     = float(kwargs.get("temperature",    self.config.temperature))
        step_response_len = int(kwargs.get("step_response_length", self.config.step_response_length))
        response_len    = int(kwargs.get("response_length",  self.config.response_length))

        system_prompt = DEFAULT_SYSTEM_PROMPT

        # —— 2. 解码 raw prompts ——
        idx            = prompts.batch["input_ids"]         # [bs, Lp]
        device         = idx.device
        bs             = idx.size(0)
        raw_prompts    = self.tokenizer.batch_decode(idx, skip_special_tokens=True)

        # —— 3. SamplingParams for intermediate rollout ——
        base_sp = SamplingParams(
            max_tokens=step_response_len,
            logprobs=1,
            temperature=temperature,
            n=num_rollout,
            stop=["\n", "<end_of_reasoning>"]
        )
        # —— 初始化 weights_history ——
        #weights_history = [[[] for _ in range(beam_size)] for _ in range(bs)]

        # —— 4. 初始化 prev_steps/prev_values 为 [bs][beam_size] ——
        prev_steps  = [["" for _ in range(beam_size)] for _ in range(bs)]
        prev_values = [[0.0 for _ in range(beam_size)] for _ in range(bs)]

        # —— 5. 多步前瞻 (num_foresight) ——
        for depth in range(num_foresight):
            # —— 阶段1：并行构造所有 prompt ——
            all_inputs = []
            for b in range(bs):
                for k in range(beam_size):
                    prefix = (
                        #f"{system_prompt}\n\n"
                        f"User: {raw_prompts[b].strip()}\n"
                        f"Reasoning so far:\n{prev_steps[b][k]}"
                    )
                    #all_inputs += [prefix] * num_rollout
                    all_inputs.append(prefix)    # 不再 * num_rollout
                    

            prompt_ids = [self.tokenizer.encode(t, add_special_tokens=False)
                        for t in all_inputs]

            outs = self.inference_engine.generate(
                prompts=None,
                sampling_params=base_sp,
                prompt_token_ids=prompt_ids,
                use_tqdm=False
            )

            # —— 扁平化收集 all_resp, all_lp
            all_resp = []
            all_lp   = []
            for out in outs:                     # outs 的长度 = bs * beam_size
                for o in out.outputs:            # 每个 out.outputs 的长度 = num_rollout
                    txt = o.text.strip()
                    lp  = o.cumulative_logprob / (len(o.token_ids) + 1e-8)
                    all_resp.append(txt)
                    all_lp.append(lp)

            # 计算 all_adv
            all_adv = []
            for b in range(bs):
                for k in range(beam_size):
                    base = (b*beam_size + k)*num_rollout
                    prev_v = prev_values[b][k]
                    for j in range(num_rollout):
                        all_adv.append(all_lp[base + j] - prev_v)

            # Stage2-4：对每个 b 做剪枝和选 beam
            new_steps  = [["" for _ in range(beam_size)] for _ in range(bs)]
            new_values = [[0.0 for _ in range(beam_size)] for _ in range(bs)]
            #new_weights = [[[] for _ in range(beam_size)] for _ in range(bs)]

            for b in range(bs):
                # slice for this batch element
                start = b * beam_size * num_rollout
                end   = start + beam_size * num_rollout
                resp_slice = all_resp[start:end]
                lp_slice   = all_lp[start:end]
                adv_slice  = np.array(all_adv[start:end])

                # 宽度剪枝 low_sigma
                mu, sigma = np.mean(lp_slice), np.std(lp_slice)
                keep = [i for i,v in enumerate(lp_slice) if v > mu - sigma_rate*sigma]
                if len(keep) < beam_size:
                    wts = np.exp(adv_slice/temperature)
                    wts /= wts.sum()
                    extra = np.random.choice(
                        len(adv_slice),
                        beam_size-len(keep),
                        replace=False,
                        p=wts
                    ).tolist()
                    keep += extra
                keep.sort()

                # 候选集合
                cand   = [resp_slice[i] for i in keep]
                adv_k  = adv_slice[keep]

                # 计算增益权重
                adv_weights = softmax(adv_k/temperature)

                # 合并权重
                #combined_weights = 0.5*cluster_weights + 0.5*adv_weights
                combined_weights = adv_weights
                combined_weights /= combined_weights.sum()

                # 一次性抽 beam_size 条
                selected_idxs = np.random.choice(
                    len(keep), size=beam_size, replace=False, p=combined_weights
                ).tolist()
                picked = [keep[i] for i in selected_idxs]

                # 更新
                for k, sel in enumerate(picked):
                    origin_beam = sel // num_rollout
                    new_steps[b][k]  = prev_steps[b][origin_beam] + resp_slice[sel] + "\n"
                    new_values[b][k] = lp_slice[sel]

            prev_steps, prev_values = new_steps, new_values
        #print(f"prev_steps: {prev_steps}")###check prev_steps

        # —— 6. 最后一轮并行生成最终答案 ——(argmax)
        final_prompts = []
        history_list  = []
        for b in range(bs):
            # 选择最大的 prev_value 对应的 beam
            best_k = int(np.argmax(prev_values[b]))
            history = prev_steps[b][best_k]
            txt = (
                #f"{system_prompt}\n\n"
                f"User: {raw_prompts[b].strip()}\n"
                f"Reasoning so far:\n{prev_steps[b][best_k]}\n"
                #f"Final step: Continue the reasoning above—include **all** intermediate steps—and write out the full reasoning process, "
                f"Final step: ending with the answer in the format \\boxed{{…}} and finish the reasoning with <end_of_reasoning>."
            )
            #print(f"raw_prompts[{b}]:{raw_prompts[b]}")###check raw_prompts
            #print(f"prev_steps[{b}]:{prev_steps[b][best_k]}")###check prev_steps
            #print(f"final_prompt[{b}]:{txt}")###check final_prompt
            ids = self.tokenizer.encode(txt, add_special_tokens=False)
            # 重复 n 次，保证 final_prompts 和 history_list 都是 bs * n 长度
            for _ in range(self.config.n):
                final_prompts.append(ids)
                history_list.append(history)

        final_sp = SamplingParams(
            max_tokens = response_len,
            logprobs   = 1,
            temperature= temperature,
            n          = 1,
            stop       = ["<end_of_reasoning>"]
        )
        final_outs = self.inference_engine.generate(
            prompts=None,
            sampling_params=final_sp,
            prompt_token_ids=final_prompts,
            use_tqdm=False
        )

        # # 收集并返回,只关心最后生成的那一个，第一版本
        # responses = [ out.outputs[0].token_ids for out in final_outs ]
        # resp_padded = pad_2d_list_to_length(
        #     responses, self.tokenizer.pad_token_id, max_length=response_len
        # ).to(device)
        # 第二版本 用 out.text.strip() 拼接 history，得到完整文本列表，并 tokenize+pad
        full_texts = []
        for history, out in zip(history_list, final_outs):
            gen_text = out.outputs[0].text.strip()
            full_texts.append(history + gen_text)
        print("full_texts[0] 示例：\n", full_texts[0])####check full_texts
        full_resp_ids = [
            self.tokenizer.encode(t, add_special_tokens=False)
            for t in full_texts
        ]
        resp_padded = pad_2d_list_to_length(
            full_resp_ids,
            self.tokenizer.pad_token_id,
            max_length=response_len
        ).to(device)

        # # —— 6. 最后一轮并行生成最终答案 ——(sampling instead of argmax) —— 第二版 bug: batchsize对不齐 AssertionError: Two tensor dict must have identical batch size. Got torch.Size([256]) and torch.Size([64])
        # # 为每个 (b,k) 生成 num_rollout 条候选
        # final_prompts = []
        # origin_idx   = []  # 记录 (b, k) 对应关系
        # for b in range(bs):
        #     for k in range(beam_size):
        #         history = prev_steps[b][k]
        #         txt = (
        #             f"User: {raw_prompts[b].strip()}\n"
        #             f"Reasoning so far:\n{history}\n"
        #             f"Final step: ending with the answer in the format \\boxed{{…}} and finish the reasoning with <end_of_reasoning>."
        #         )
        #         ids = self.tokenizer.encode(txt, add_special_tokens=False)
        #         # 重复 n=num_rollout 次
        #         final_prompts += [ids] * num_rollout
        #         origin_idx += [(b, k)] * num_rollout

        # final_sp = SamplingParams(
        #     max_tokens = response_len,
        #     logprobs   = 1,
        #     temperature= temperature,
        #     n          = 1,             # 因为 prompt 已经重复了，这里 n=1
        #     stop       = ["<end_of_reasoning>"]
        # )
        # final_outs = self.inference_engine.generate(
        #     prompts=None,
        #     sampling_params=final_sp,
        #     prompt_token_ids=final_prompts,
        #     use_tqdm=False
        # )

        # # 收集所有候选文本和对应的 logprob
        # all_final_resp = []
        # all_final_lp   = []
        # for out in final_outs:
        #     txt = out.outputs[0].text.strip()
        #     lp  = out.outputs[0].cumulative_logprob / (len(out.outputs[0].token_ids) + 1e-8)
        #     all_final_resp.append(txt)
        #     all_final_lp.append(lp)

        # # 按 batch b 做剪枝 + 抽样
        # selected_texts = [""] * bs
        # for b in range(bs):
        #     # 收集属于 b 的所有 (k, j) 候选
        #     idxs = [i for i, (bb, kk) in enumerate(origin_idx) if bb == b]
        #     resp_slice = [all_final_resp[i] for i in idxs]
        #     lp_slice   = np.array([all_final_lp[i]   for i in idxs])
        #     # 计算增益 adv = lp - prev_values[b][k]
        #     prev_vs = np.array([prev_values[b][origin_idx[i][1]] for i in idxs])
        #     adv_slice = lp_slice - prev_vs

        #     # 低 sigma 剪枝
        #     mu, sigma = lp_slice.mean(), lp_slice.std()
        #     keep = [i for i,v in enumerate(lp_slice) if v > mu - sigma_rate*sigma]
        #     if len(keep) < 1:
        #         keep = list(range(len(lp_slice)))

        #     # 计算权重
        #     adv_weights = np.exp(adv_slice[keep] / temperature)
        #     combined_weights = adv_weights / adv_weights.sum()

        #     # 按权重抽一条
        #     pick = np.random.choice(keep, p=combined_weights)
        #     selected_texts[b] = prev_steps[b][origin_idx[idxs[pick]][1]] + resp_slice[pick]

        # # 把选中的文本编码、pad，拼回 batch
        # full_resp_ids = [
        #     self.tokenizer.encode(t, add_special_tokens=False)
        #     for t in selected_texts
        # ]
        # resp_padded = pad_2d_list_to_length(
        #     full_resp_ids,
        #     self.tokenizer.pad_token_id,
        #     max_length=response_len
        # ).to(device)



        # repeat 原 prompt tensors
        Bn = resp_padded.size(0)
        repeat_times = Bn // bs
        idx    = prompts.batch["input_ids"].repeat_interleave(repeat_times, dim=0)
        mask   = prompts.batch["attention_mask"].repeat_interleave(repeat_times, dim=0)
        pos_ids= prompts.batch["position_ids"].repeat_interleave(repeat_times, dim=0)
       
        seq = torch.cat([idx, resp_padded], dim=1)
        delta = torch.arange(1, response_len+1, device=device).unsqueeze(0).expand(Bn, -1)
        last_pos = pos_ids[:, -1:].expand(-1, response_len)
        pos_ids = torch.cat([pos_ids, last_pos+delta], dim=1)
        resp_attn = get_eos_mask(resp_padded, self.tokenizer.eos_token_id, mask.dtype)
        mask = torch.cat([mask, resp_attn], dim=1)

        # # 假设 seq 是一个 shape [Bn, L] 的 Tensor,检验结果
        # first_ids = seq[0].tolist()  
        # first_text = self.tokenizer.decode(first_ids, skip_special_tokens=True)
        # print(f"First sequence text: {first_text}")  # 打印第一个序列的文本

        batch = TensorDict({
            "prompts":        idx,
            "responses":      resp_padded, #response ids
            "input_ids":      seq, #complete input+response ids
            "attention_mask": mask,
            "position_ids":   pos_ids,
        }, batch_size=Bn)

        return DataProto(batch=batch)'''
    #rollout 成功版，使用 combined_weights
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        多步 Phi-Decoding，batch 维度并行版
        """
        # —— 1. 超参数 ——
        beam_size       = int(kwargs.get("step_beam_size",   self.config.step_beam_size))
        num_rollout     = int(kwargs.get("num_rollout",      self.config.num_rollout))
        num_foresight   = int(kwargs.get("num_foresight",    self.config.num_foresight))
        threshold       = float(kwargs.get("threshold",      self.config.threshold))
        sigma_rate      = float(kwargs.get("sigma_rate",     self.config.sigma_rate))
        cluster_num     = int(kwargs.get("cluster_num",      self.config.cluster_num))
        temperature     = float(kwargs.get("temperature",    self.config.temperature))
        step_response_len = int(kwargs.get("step_response_length", self.config.step_response_length))
        response_len    = int(kwargs.get("response_length",  self.config.response_length))

        system_prompt = DEFAULT_SYSTEM_PROMPT

        # —— 2. 解码 raw prompts ——
        idx            = prompts.batch["input_ids"]         # [bs, Lp]
        device         = idx.device
        bs             = idx.size(0)
        raw_prompts    = self.tokenizer.batch_decode(idx, skip_special_tokens=True)

        # —— 3. SamplingParams for intermediate rollout ——
        base_sp = SamplingParams(
            max_tokens=step_response_len,
            logprobs=1,
            temperature=temperature,
            n=num_rollout,
            stop=["\n", "<end_of_reasoning>"]
        )
        # —— 初始化 weights_history ——
        weights_history = [[[] for _ in range(beam_size)] for _ in range(bs)]

        # —— 4. 初始化 prev_steps/prev_values 为 [bs][beam_size] ——
        prev_steps  = [["" for _ in range(beam_size)] for _ in range(bs)]
        prev_values = [[0.0 for _ in range(beam_size)] for _ in range(bs)]

        # —— 5. 多步前瞻 (num_foresight) ——
        for depth in range(num_foresight):
            #skip_current = False      # 重置，下轮还能重新判断
            # —— 阶段1：并行构造所有 prompt ——
            all_inputs = []
            for b in range(bs):
                for k in range(beam_size):
                    prefix = (
                        #f"{system_prompt}\n\n"
                        f"User: {raw_prompts[b].strip()}\n"
                        f"Reasoning so far:\n{prev_steps[b][k]}"
                    )
                    #all_inputs += [prefix] * num_rollout
                    all_inputs.append(prefix)    # 不再 * num_rollout
                    

            prompt_ids = [self.tokenizer.encode(t, add_special_tokens=False)
                        for t in all_inputs]

            outs = self.inference_engine.generate(
                prompts=None,
                sampling_params=base_sp,
                prompt_token_ids=prompt_ids,
                use_tqdm=False
            )

            # —— 扁平化收集 all_resp, all_lp
            all_resp = []
            all_lp   = []
            for out in outs:                     # outs 的长度 = bs * beam_size
                for o in out.outputs:            # 每个 out.outputs 的长度 = num_rollout
                    txt = o.text.strip()
                    lp  = o.cumulative_logprob / (len(o.token_ids) + 1e-8)
                    all_resp.append(txt)
                    all_lp.append(lp)

            # 计算 all_adv
            all_adv = []
            for b in range(bs):
                for k in range(beam_size):
                    base = (b*beam_size + k)*num_rollout
                    prev_v = prev_values[b][k]
                    for j in range(num_rollout):
                        all_adv.append(all_lp[base + j] - prev_v)

            # Stage2-4：对每个 b 做剪枝和选 beam
            new_steps  = [["" for _ in range(beam_size)] for _ in range(bs)]
            new_values = [[0.0 for _ in range(beam_size)] for _ in range(bs)]
            new_weights = [[[] for _ in range(beam_size)] for _ in range(bs)]

            for b in range(bs):
                # slice for this batch element
                start = b * beam_size * num_rollout
                end   = start + beam_size * num_rollout
                resp_slice = all_resp[start:end]
                lp_slice   = all_lp[start:end]
                adv_slice  = np.array(all_adv[start:end])

                # 宽度剪枝 low_sigma
                mu, sigma = np.mean(lp_slice), np.std(lp_slice)
                keep = [i for i,v in enumerate(lp_slice) if v > mu - sigma_rate*sigma]
                if len(keep) < beam_size:
                    wts = np.exp(adv_slice/temperature)
                    wts /= wts.sum()
                    extra = np.random.choice(
                        len(adv_slice),
                        beam_size-len(keep),
                        replace=False,
                        p=wts
                    ).tolist()
                    keep += extra
                keep.sort()

                # 候选集合
                cand   = [resp_slice[i] for i in keep]
                adv_k  = adv_slice[keep]

                # # # 聚类###################################有深度剪枝
                # X      = TfidfVectorizer().fit_transform(cand)
                # labels = KMeans(n_clusters=cluster_num).fit_predict(X)

                # # 计算簇比例
                # counts = np.bincount(labels, minlength=cluster_num)
                # cluster_ratios = counts / counts.sum()
                # cluster_weights = softmax(cluster_ratios[labels])

                # # 计算增益权重
                # adv_weights = softmax(adv_k/temperature)
                # combined_weights= adv_weights

                # # 合并权重
                # combined_weights = 0.5*cluster_weights + 0.5*adv_weights

                # # —— 深度剪枝判定 ——  
                # # 计算最大的簇比例，如果超过阈值就停止前瞻
                # largest_ratio = counts.max() / len(resp_slice)
                # if largest_ratio >= threshold:
                #     # 跳出内层 batch 循环和外层 depth 循环
                #     skip_current = True
                #     break
                
                # # # —— 新增：检测是否已有完整轨迹 ——  
                # traj_complete = False
                # # # 这里的 candidates_list 就是 cand；如果你用其它变量名，请相应替换
                # traj_complete = any(tag in text
                #                    for text in cand
                #                    for tag in ["\\boxed{", "<boxed>", "<end_of_reasoning>"])
                # if traj_complete:
                #    skip_current = True
                #    break  
                ##################################################################
                # 计算增益权重，无深度剪枝
                combined_weights = softmax(adv_k/temperature)#########################################

                # #########################
                # if skip_current:
                #     continue                  # 跳过 Stage2–4，进入下一个 depth
                # 一次性抽 beam_size 条
                selected_idxs = np.random.choice(
                    len(keep), size=beam_size, replace=False, p=combined_weights
                ).tolist()
                picked = [keep[i] for i in selected_idxs]

                # 更新
                for k, sel in enumerate(picked):
                    origin_beam = sel // num_rollout

                    resp_text = resp_slice[sel]
                    # 记录本步权重并 replicate
                    idx_in_keep = keep.index(sel)
                    # w = float(combined_weights[idx_in_keep])
                    # token_ids = self.tokenizer.encode(resp_text, add_special_tokens=False)
                    # rep_ws = [w] * len(token_ids)
                    # new_weights[b][k] = weights_history[b][origin_beam] + rep_ws
                    # 新：直接取 raw advantage adv_slice[sel]####
                    raw_adv = float(adv_slice[sel])               # adv_slice = all_lp - prev_v
                    token_ids = self.tokenizer.encode(resp_text, add_special_tokens=False)
                    #rep_adv = [raw_adv] * len(token_ids)
                    rep_adv = [0.0] * (len(token_ids)-1)+ [raw_adv]      ## 只在最后一个 token 上记录 adv，其余为 0.0
                    new_weights[b][k] = weights_history[b][origin_beam] + rep_adv

                    new_steps[b][k]  = prev_steps[b][origin_beam] + resp_slice[sel] + "\n"
                    new_values[b][k] = lp_slice[sel]


            prev_steps, prev_values, weights_history = new_steps, new_values, new_weights
        #print(f"prev_steps: {prev_steps}")###check prev_steps

        # —— 6. 最后一轮并行生成最终答案 ——(argmax)
        final_prompts = []
        history_list  = []
        final_rewards = []
        final_probs   = []
        for b in range(bs):
            # # 选择最大的 prev_value 对应的 beam
            # best_k = int(np.argmax(prev_values[b]))
            # —— 基于 prev_values 重新计算增益权重（adv）并做 softmax 采样 —— 
            vals = np.array(prev_values[b])                     # shape: [beam_size]
            adv  = vals - vals.mean()                           # centered advantage
            beam_p = np.exp(adv / temperature)
            beam_p /= beam_p.sum()                                # 归一化概率
            best_k = int(np.random.choice(len(beam_p), p=beam_p))  # 按概率采样一个 beam
            p = float(beam_p[best_k])   # 把选中那条 beam 的概率取出来
            #新adv####
            adv_all = vals - vals.mean()####
            raw_adv_final = float(adv_all[best_k])####

            history = prev_steps[b][best_k]
            base_ws   = weights_history[b][best_k]
            txt = (
                #f"{system_prompt}\n\n"
                f"User: {raw_prompts[b].strip()}\n"
                f"Reasoning so far:\n{prev_steps[b][best_k]}\n"
                #f"Final step: Continue the reasoning above—include **all** intermediate steps—and write out the full reasoning process, "
                #f"Final step: ending with the answer in the format \\boxed{{…}} and finish the reasoning with <end_of_reasoning>."
            )
            #print(f"raw_prompts[{b}]:{raw_prompts[b]}")###check raw_prompts
            #print(f"prev_steps[{b}]:{prev_steps[b][best_k]}")###check prev_steps
            #print(f"final_prompt[{b}]:{txt}")###check final_prompt
            ids = self.tokenizer.encode(txt, add_special_tokens=False)
            # 重复 n 次，保证 final_prompts 和 history_list 都是 bs * n 长度
            for _ in range(self.config.n):
                final_prompts.append(ids)
                history_list.append(history)
                final_rewards.append(list(base_ws))
                #final_probs.append(p)  # 保存选中 beam 的概率
                final_probs.append(raw_adv_final)####adv版

        final_sp = SamplingParams(
            max_tokens = response_len,
            logprobs   = 1,
            temperature= temperature,
            n          = 1,
            stop       = ["<end_of_reasoning>"]
        )
        final_outs = self.inference_engine.generate(
            prompts=None,
            sampling_params=final_sp,
            prompt_token_ids=final_prompts,
            use_tqdm=False
        )

        
        full_texts = []
        final_rewards_padded = []
        for i, (history, out) in enumerate(zip(history_list, final_outs)):
            gen_text = out.outputs[0].text.strip()
            full_texts.append(history + gen_text)
            token_ids = self.tokenizer.encode(gen_text, add_special_tokens=False)
            #final_rewards_padded.append(final_rewards[i] + [1.0] * len(token_ids))
            prob = final_probs[i]
            #final_rewards_padded.append(final_rewards[i] + [prob] * len(token_ids))
            segment_reward = [0.0] * (len(token_ids) - 1)+ [prob] 
            final_rewards_padded.append(final_rewards[i] + segment_reward)  #最后一个 token 的 reward 是 prob，其余为 0.0

        # encode & pad responses
        full_resp_ids = [self.tokenizer.encode(t, add_special_tokens=False) for t in full_texts]
        #full_resp_ids = [ids[:response_len] for ids in full_resp_ids]#截断到 response_len
        resp_padded = pad_2d_list_to_length(
            full_resp_ids,
            self.tokenizer.pad_token_id,
            max_length=response_len
        ).to(device)

        # —— 8. 构建 prm_reward 张量 ——
        # 使得 prm_reward 的每行长度与 resp_padded 的响应长度一致 (response_len)
        pr_tensors = []
        for r in final_rewards_padded:
            # 截断或补齐到 response_len
            if len(r) >= response_len:
                row = r[:response_len]
            else:
                row = r + [0.0] * (response_len - len(r))
            pr_tensors.append(row)
        prm_reward = torch.tensor(pr_tensors, device=device)

        
        # ####neu reward r*
        # # 按r*公式算权重：w_i = exp(-r_i/T) / sum_j exp(-r_j/T)
        r = prm_reward
        T = 0.1 # 温度越小，最差一步越影响权重
        exp_neg = torch.exp(-r / T)           # [Bn, L]
        den = exp_neg.sum(dim=1, keepdim=True)  # [Bn, 1]
        w = exp_neg / den                       # [Bn, L]

        # 4) 最终 r*_i = w_i * r_i
        r_star = w * r                          # [Bn, L]

        # 5) 用 r_star 作为 prm_reward
        prm_reward = r_star

        # —— 9. repeat 原 prompt tensors & 拼 batch —— repeat 原 prompt tensors & 拼 batch ——
        Bn = resp_padded.size(0)
        repeat = Bn // bs
        idx    = prompts.batch["input_ids"].repeat_interleave(repeat, dim=0)
        mask   = prompts.batch["attention_mask"].repeat_interleave(repeat, dim=0)
        pos    = prompts.batch["position_ids"].repeat_interleave(repeat, dim=0)
        seq    = torch.cat([idx, resp_padded], dim=1)
        delta  = torch.arange(1, response_len+1, device=device).unsqueeze(0).expand(Bn, -1)
        last_p = pos[:, -1:].expand(-1, response_len)
        pos    = torch.cat([pos, last_p+delta], dim=1)
        attn   = get_eos_mask(resp_padded, self.tokenizer.eos_token_id, mask.dtype)
        mask   = torch.cat([mask, attn], dim=1)

        batch = TensorDict({
            "prompts":        idx,
            "responses":      resp_padded,
            "input_ids":      seq,
            "attention_mask": mask,
            "position_ids":   pos,
            "prm_reward":     prm_reward,
        }, batch_size=Bn)
        print(f"[DEBUG] resp_padded.shape: {resp_padded.shape}")####check resp_padded shape
        print(f"[DEBUG] prm_reward.shape: {prm_reward.shape}")####check prm_reward shape
        print(f"[DEBUG] prm_reward[0]: {prm_reward[0]}")####check prm_reward[0]
        with open("prm_reward0803.txt", "w", encoding="utf-8") as f:
            f.write(str(prm_reward[0].tolist()))
        # 将第一个样本的 response 文本保存到文件
        first_resp_ids = resp_padded[0].tolist()
        #print(f"[DEBUG] first_resp_ids: {first_resp_ids}")####check first_resp_ids
        first_resp_text = self.tokenizer.decode(first_resp_ids, skip_special_tokens=True)
        print(f"[DEBUG] first_resp_text: {first_resp_text}")####check first_resp_text
        with open("first_response0803.txt", "w", encoding="utf-8") as f:
            f.write(first_resp_text)
        # 可选：打印文件路径以确认
        print("[DEBUG] Saved prm_reward to prm_reward_0.txt and first response to first_response_0.txt")
        return DataProto(batch=batch)

