from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch
from accelerate import Accelerator
from functools import partial
from colossalai.pipeline.rpc._pipeline_schedule import OneFOneBPipelineEngine
from colossalai.pipeline.pipeline_process_group import ppg
from colossalai import launch
from colossalai.fx import ColoTracer
from colossalai.fx.passes.adding_split_node_pass import balanced_split_pass, split_with_split_nodes_pass
from colossalai.pipeline.middleware.adaptor import get_fx_topology

device = 'cuda'
# model_name = "/datasets/models/Llama2-7B"
model_name = "/datasets/models/llama-160m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize the accelerator
# accelerator = Accelerator()
# model = accelerator.prepare(model)

def create_partition_module(pp_rank: int, stage_num: int, model, data_kwargs):
    model.eval()
    tracer = ColoTracer()
    meta_args = {k: v.to('meta') for k, v in data_kwargs.items()}
    graph = tracer.trace(root=model, meta_args=meta_args)
    gm = torch.fx.GraphModule(model, graph, model.__class__.__name__)
    
    annotated_model = balanced_split_pass(gm, stage_num)
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

if __name__ == "__main__":
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    print(encodings)
        
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    # Initialize pipeline engine
    stage_num = 4  # Number of stages in the pipeline
    num_microbatches = 8
    chunk = 1
    use_checkpoint = 'store_true'
    
    data_kwargs = {
        'input_ids': encodings.input_ids[:, 0:512].to(device)
    }
    
    engine = OneFOneBPipelineEngine(
        partition_fn=partial(partition, model, data_kwargs),
        stage_num=stage_num,
        num_microbatches=num_microbatches,
        device=device,
        chunk=chunk,
        checkpoint=use_checkpoint,
    )

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + 512, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = engine.forward_backward({'x': input_ids}, labels=target_ids, forward_only=True)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    
    print(f"Perplexity: {ppl}")
