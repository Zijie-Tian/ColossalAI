import torch
import torch.fx as fx
import time

# 1. 定义计时器
class NodeTimer:
    def __init__(self):
        self.start_times = {}
        self.end_times = {}

    def start(self, node_name):
        self.start_times[node_name] = time.time()

    def end(self, node_name):
        self.end_times[node_name] = time.time()

    def get_execution_time(self, node_name):
        return self.end_times[node_name] - self.start_times[node_name]

# 2. 自定义 Tracer 类
class TimerTracer(fx.Tracer):
    def __init__(self, timer):
        super().__init__()
        self.timer = timer

    def call_module(self, module, forward, args, kwargs):
        # 重载 call_module 来记录模块的执行时间
        def timed_forward(*args, **kwargs):
            module_name = str(module.__class__.__name__)
            print(f"Executing module: {module_name}")
            # 记录开始时间
            self.timer.start(module_name)
            result = forward(*args, **kwargs)
            # 记录结束时间
            self.timer.end(module_name)
            
            print(f"Module execution time: {self.timer.get_execution_time(module_name)} seconds")
            return result

        return timed_forward(*args, **kwargs)
    
    def call_function(self, target, args, kwargs):
        def timed_forward(*args, **kwargs):
            function_name = str(target)
            print(f"Executing function: {function_name}")
            # 记录开始时间
            self.timer.start(function_name)
            result = target(*args, **kwargs)
            # 记录结束时间
            self.timer.end(function_name)
        
        return timed_forward(*args, **kwargs)

# 3. 自定义 GraphModule
class TimedGraphModule(fx.GraphModule):
    def __init__(self, root, graph, timer):
        super().__init__(root, graph)
        self.timer = timer

    def forward(self, *args, **kwargs):
        print("Starting forward pass with timing...")
        return super().forward(*args, **kwargs)

# 4. 示例模型
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(5, 5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        return self.relu(x)

# 5. 应用自定义 Tracer 和 Timer
timer = NodeTimer()
tracer = TimerTracer(timer)
module = MyModule()

# 追踪模型的计算图
graph = tracer.trace(module)

# 创建 GraphModule
timed_module = TimedGraphModule(module, graph, timer)

timed_module.graph.print_tabular()

# 运行模型，并记录时间
x = torch.randn(5)
output = timed_module(x)

# 输出模块的执行时间
for node in graph.nodes:
    if node.op == 'call_module':  # 仅输出模块节点的执行时间
        print(f"Node: {node}, Execution Time: {timer.get_execution_time(node.target)} seconds")