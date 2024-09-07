'''
Author: Zijie Tian tzj21@mails.tsinghua.edu.cn
Date: 2024-06-24 15:54:46
LastEditors: Zijie Tian tzj21@mails.tsinghua.edu.cn
LastEditTime: 2024-06-25 13:24:06
FilePath: /Elixir/Elixir/example/common/fsdp.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.distributed as dist
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

from elixir.ctx import MetaContext
from elixir.kernels.attn_wrapper import wrap_attention
from elixir.utils import get_model_size

import sys
sys.path.append('/home/tzj/Code/Elixir/Elixir/example/')
from example.common.models import get_model


def train_init(model_name: str):
    with MetaContext('cuda'):
        model = get_model(model_name)
    model_size = get_model_size(model)
    model = FSDP(module=model, mixed_precision=True, flatten_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    model.gradient_checkpointing_enable()

    model = wrap_attention(model)
    model.train()

    def forward(data):
        return model(**data)

    def backward(loss):
        loss.backward()

    def optim():
        optimizer.step()
        optimizer.zero_grad()

    return forward, backward, optim, model_size


if __name__ == '__main__':
    import colossalai
    colossalai.launch_from_torch(config={})
    print(train_init('opt-1b'))
