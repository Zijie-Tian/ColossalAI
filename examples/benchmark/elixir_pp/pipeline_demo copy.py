import os
from functools import partial

import warnings
# 忽略所有的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

import pytest
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc 
# from rpc_test_utils import DAG_MLP, MLP
from torch.distributed.rpc import _is_current_rpc_agent_set
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

from colossalai import launch
from colossalai.fx import ColoTracer
from colossalai.fx.passes.adding_split_node_pass import balanced_split_pass, split_with_split_nodes_pass
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.pipeline.middleware.adaptor import get_fx_topology
from colossalai.pipeline.pipeline_process_group import ppg
from colossalai.pipeline.rpc._pipeline_schedule import OneFOneBPipelineEngine
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn

class MLP(nn.Module):

    def __init__(self, dim: int, layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        for _ in range(layers):
            self.layers.append(nn.Linear(dim, dim, bias=False))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.sum()


class DAG_MLP(nn.Module):

    def __init__(self, dim: int, layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.dag_layer = nn.Linear(dim, dim, bias=False)

        for _ in range(layers):
            self.layers.append(nn.Linear(dim, dim, bias=False))

    def forward(self, x, y):
        for layer in self.layers:
            x = layer(x)
            y = self.dag_layer(y)
        return x.sum(), y.sum()

#> 创建一个Transformer解码器
def create_decoder_layers(num_layers):
    
    d_model = 512
    nhead = 8
    dim_feedforward = 2048
    
    return nn.TransformerDecoder(
        nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward),
        num_layers=num_layers
    )

# global variable for model created
batch_size = 16
dim = 10
rpc_is_initialized = _is_current_rpc_agent_set

logger = get_dist_logger()
logger.set_level('DEBUG')


def create_partition_module(pp_rank: int, stage_num: int, model, data_kwargs):
    model.eval()
    tracer = ColoTracer()
    meta_args = {k: v.to('meta') for k, v in data_kwargs.items()}
    
    logger.info(f"Start tracer.trace")
    graph = tracer.trace(root=model, meta_args=meta_args)
    gm = torch.fx.GraphModule(model, graph, model.__class__.__name__)
    
    #> 在每个分区的输入输出之间插入split和gather节点
    logger.info(f"Start balanced_split_pass, stage_num: {stage_num}")
    annotated_model = balanced_split_pass(gm, stage_num)
    
    
    logger.info(f"Start split_with_split_nodes_pass, merge_output: {True}")
    top_module, split_submodules = split_with_split_nodes_pass(annotated_model, merge_output=True)
    topo = get_fx_topology(top_module)
    for submodule in split_submodules:
        if isinstance(submodule, torch.fx.GraphModule):
            setattr(submodule, '_topo', topo)
    return split_submodules[pp_rank + 1]


def partition(model, data_kwargs: dict, pp_rank: int, chunk: int, stage_num: int):
    torch.manual_seed(1024)
    partition = create_partition_module(pp_rank, stage_num, model, data_kwargs)
    return partition


def run_master(world_size, forward_only):
    torch.manual_seed(100)
    
    model_name = "/datasets/models/llama-160m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
    
    # model = create_decoder_layers(12)
    model = MLP(12, 512)
    
    
    epoch = 3
    device = 'cuda'
    stage_num = world_size
    chunk = 1
    num_microbatches = 8
    use_checkpoint = 'store_true'

    # data_kwargs = {
    #     'tgt' : torch.randn((1, 512), dtype=torch.float32, device=device),
    #     'memory' : torch.randn((1, 512), dtype=torch.float32, device=device)
    # }
    
    data_kwargs = {
        'x': torch.randn((batch_size, dim), dtype=torch.float32, device=device)
    }
    
    #> Here we split the model.
    engine = OneFOneBPipelineEngine(
        partition_fn=partial(partition, model, data_kwargs),
        stage_num=stage_num,
        num_microbatches=num_microbatches,
        device=device,
        chunk=chunk,
        checkpoint=use_checkpoint,
    )
    
    if not forward_only:
        engine.initialize_optimizer(getattr(torch.optim, 'SGD'), lr=1e-3)

    for _ in range(epoch):
        input_x = torch.randn((batch_size, dim), device=device)
        input_y = torch.randn((batch_size, dim), device=device)
        # logits = engine.forward_backward({'x': input_x, 'y': input_y}, labels=labels, forward_only=forward_only)
    
    partition_models = partition(model, data_kwargs, 0, 1, 4)
    print(partition_models)
    
    import pdb; pdb.set_trace()
    
    return partition_models

def run_worker(rank, world_size, port, forward_only, master_func):
    master_addr = 'localhost'
    master_port = 29020
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)

    disable_existing_loggers()

    #> Run function 'run_master' in each process
    launch(dict(), rank, world_size, master_addr, master_port, 'nccl', verbose=False)
    ppg.set_global_info(rank=rank,
                        world_size=world_size,
                        dp_degree=1,
                        tp_degree=1,
                        num_worker_threads=128,
                        device='cuda')
    

    
    # example_text = "What's your name?"
    
    # if tokenizer is not None:
    #     input_ids = tokenizer.encode(example_text, return_tensors='pt').to('cuda')
    # if model is not None:
    #     print("model is not None")
        
    # result = model.generate(input_ids, max_length=256)
    # print(tokenizer.decode(result[0], skip_special_tokens=True))

    # in rpc mode, only rank 0 is needed to be coded
    if rank == 0:
        master_func(world_size, forward_only)
        
    # barrier here
    if rpc_is_initialized():
        rpc.shutdown()


@pytest.mark.skip("skip due to CI torch version 1.11")
@parameterize('forward_only', [True])
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_pp_middleware_fwd(forward_only):
    world_size = 1
    master_func = run_master
    spawn(
        run_worker,
        world_size,
        forward_only=forward_only,
        master_func=master_func,
    )


if __name__ == "__main__":
    test_pp_middleware_fwd()
