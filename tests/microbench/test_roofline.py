import torch
import torch.nn as nn
import time

import sys
from pathlib import Path
sys.path.append(str(Path(sys.argv[0]).absolute().parent.parent))
from memory import ContinuousMemoryAllocator

# 设置模型参数
d_model = 512
nhead = 8
dim_feedforward = 2048
seq_len = 50
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建不同层数的Transformer解码器
def create_decoder_layers(num_layers):
    return nn.TransformerDecoder(
        nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward),
        num_layers=num_layers
    )

# 测试不同层数的解码器
num_layers_list = [1, 2, 4, 8, 12]
cpu_to_gpu_times = []
gpu_compute_times = []
gpu_backward_times = []

# 输入数据
tgt = torch.rand(seq_len, batch_size, d_model)
memory = torch.rand(seq_len, batch_size, d_model)

for num_layers in num_layers_list:
    # 创建指定层数的解码器
    
    cpu_allocator = ContinuousMemoryAllocator(1024 * 1024 * 1024, torch.float32, 'cpu')
    gpu_allocator = ContinuousMemoryAllocator(1024 * 1024 * 1024, torch.float32, 'cuda')
    
    decoder = create_decoder_layers(num_layers)
    
    gpu_buffer = {}
    cpu_buffer = {}
    param_shapes = {}
    for param in decoder.parameters():
        param_shapes[param] = param.data.shape
        
        # print("param.data.numel() * param.data.element_size()", param.data.numel() * param.data.element_size())
        allocated_tensor = cpu_allocator.allocate_tensor(param.data.numel())
        
        # import pdb; pdb.set_trace()
        
        allocated_tensor.copy_(param.data.flatten())
        param.data = allocated_tensor
        
        cpu_buffer[param] = allocated_tensor
        gpu_tensor = gpu_allocator.allocate_tensor(param.data.numel())
        gpu_buffer[param] = gpu_tensor
        

    # 打印参数量
    num_params = sum(p.numel() * p.element_size() for p in decoder.parameters())
    print(f"Layers: {num_layers}, Number of parameters: {num_params / 1024 / 1024 / 1024:.2f} GB")

    # # 进行warmup
    # decoder = decoder.to(device)
    # tgt_gpu = tgt.to(device)
    # memory_gpu = memory.to(device)
    # with torch.no_grad():
    #     for _ in range(10):
    #         _ = decoder(tgt_gpu, memory_gpu)

    # 测试从CPU到GPU的传输时间
    tgt_gpu = tgt.to(device)
    memory_gpu = memory.to(device)
    
    start_time = time.time()
    
    for param in decoder.parameters():
        gpu_buffer[param].copy_(cpu_buffer[param])
        param.data = gpu_buffer[param].reshape(param_shapes[param])
        
        # print("Param shape : ", param.data.shape)
    
    torch.cuda.synchronize()
    cpu_to_gpu_time = time.time() - start_time
    cpu_to_gpu_times.append(cpu_to_gpu_time)

    # 测试在GPU上的计算时间
    start_time = time.time()
    with torch.no_grad():  # 不需要计算梯度
        output = decoder(tgt_gpu, memory_gpu)
    gpu_compute_time = time.time() - start_time
    gpu_compute_times.append(gpu_compute_time)

    # 测试在GPU上的反向传播时间
    tgt_gpu.requires_grad = True
    memory_gpu.requires_grad = True
    output = decoder(tgt_gpu, memory_gpu)
    grad_output = torch.rand_like(output)
    
    start_time = time.time()
    output.backward(grad_output)
    gpu_backward_time = time.time() - start_time
    gpu_backward_times.append(gpu_backward_time)

    print(f"Layers: {num_layers}, CPU to GPU time: {cpu_to_gpu_time:.6f} seconds, Bandwidth : {num_params / 1024 ** 3 / cpu_to_gpu_time:.6f}, GPU compute time: {gpu_compute_time:.6f} seconds, GPU backward time: {gpu_backward_time:.6f} seconds")

# 打印最终结果
print("\nSummary of results:")
for i, num_layers in enumerate(num_layers_list):
    print(f"Layers: {num_layers} -> CPU to GPU time: {cpu_to_gpu_times[i]:.6f} seconds, GPU compute time: {gpu_compute_times[i]:.6f} seconds, GPU backward time: {gpu_backward_times[i]:.6f} seconds")
