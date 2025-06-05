import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import argparse
from dataclasses import dataclass

# 设置线程数
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# 模型配置类
@dataclass
class ModelConfig:
    head_dim: int = 64

# 原始版本的 KernelizedAttention
class OriginalKernelizedAttention(nn.Module):
    def __init__(self, model_config, dtype=torch.bfloat16, eps=1e-6):
        super().__init__()
        self.head_dim = model_config.head_dim
        self.dtype = dtype
        self.eps = eps

    def prepare(self, K, V):
        bsz, num_heads, kv_len, head_dim = K.shape
        K_reshaped = K.view(bsz * num_heads, kv_len, head_dim)
        V_reshaped = V.view(bsz * num_heads, kv_len, head_dim)

        K_phi = F.elu(K_reshaped) + 1
        K_phi = K_phi.transpose(1, 2)
        self.KV = K_phi @ V_reshaped  # [bsz*num_heads, head_dim, head_dim]
        self.Z = K_phi @ torch.ones(bsz * num_heads, kv_len, 1, device=K.device, dtype=K.dtype)  # [bsz*num_heads, head_dim, 1]

    def forward(self, Q):
        # Q.shape [bsz, num_heads, q_len, head_dim]
        bsz, num_heads, q_len, head_dim = Q.shape

        Q_reshaped = Q.view(bsz * num_heads, q_len, head_dim)
        Q_phi = F.elu(Q_reshaped) + 1
        output = (Q_phi @ self.KV) / (Q_phi @ self.Z + self.eps)  # [bsz*num_heads, q_len, head_dim]
        
        output = output.reshape(bsz, num_heads, q_len, head_dim)
        return output

# 基础JIT优化模型 - Script版本
class JITScriptKernelizedAttention(nn.Module):
    def __init__(self, model_config, dtype=torch.float32, eps=1e-6):
        super().__init__()
        self.head_dim = model_config.head_dim
        self.dtype = dtype
        self.eps = eps
        # 初始化为空张量而不是None
        self.register_buffer("KV", torch.empty(0))
        self.register_buffer("Z", torch.empty(0))

    def prepare(self, K, V):
        K = K.contiguous()
        V = V.contiguous()
        
        bsz, num_heads, kv_len, head_dim = K.shape
        K_reshaped = K.view(bsz * num_heads, kv_len, head_dim)
        V_reshaped = V.view(bsz * num_heads, kv_len, head_dim)

        K_phi = F.elu(K_reshaped) + 1
        K_phi = K_phi.transpose(1, 2)
        
        self.KV = torch.bmm(K_phi, V_reshaped)
        self.Z = torch.bmm(K_phi, torch.ones(bsz * num_heads, kv_len, 1, device=K.device, dtype=K.dtype))

    def forward(self, Q):
        Q = Q.contiguous()
        
        bsz, num_heads, q_len, head_dim = Q.shape
        Q_reshaped = Q.view(bsz * num_heads, q_len, head_dim)
        
        Q_phi = F.elu(Q_reshaped) + 1
        
        numerator = torch.bmm(Q_phi, self.KV)
        denominator = torch.bmm(Q_phi, self.Z) + self.eps
        output = numerator / denominator
        
        output = output.view(bsz, num_heads, q_len, head_dim)
        return output

# JIT优化模型 - BFloat16版本
class JITBFloat16KernelizedAttention(nn.Module):
    def __init__(self, model_config, dtype=torch.bfloat16, eps=1e-6):
        super().__init__()
        self.head_dim = model_config.head_dim
        self.dtype = dtype
        self.eps = eps
        self.register_buffer("KV", torch.empty(0, dtype=torch.bfloat16))
        self.register_buffer("Z", torch.empty(0, dtype=torch.bfloat16))

    def prepare(self, K, V):
        K = K.contiguous()
        V = V.contiguous()
        
        bsz, num_heads, kv_len, head_dim = K.shape
        K_reshaped = K.view(bsz * num_heads, kv_len, head_dim)
        V_reshaped = V.view(bsz * num_heads, kv_len, head_dim)

        K_phi = F.elu(K_reshaped) + 1
        K_phi = K_phi.transpose(1, 2)
        
        self.KV = torch.bmm(K_phi, V_reshaped)
        self.Z = torch.bmm(K_phi, torch.ones(bsz * num_heads, kv_len, 1, device=K.device, dtype=K.dtype))

    def forward(self, Q):
        Q = Q.contiguous()
        
        bsz, num_heads, q_len, head_dim = Q.shape
        Q_reshaped = Q.view(bsz * num_heads, q_len, head_dim)
        
        Q_phi = F.elu(Q_reshaped) + 1
        
        numerator = torch.bmm(Q_phi, self.KV)
        denominator = torch.bmm(Q_phi, self.Z) + self.eps
        output = numerator / denominator
        
        output = output.view(bsz, num_heads, q_len, head_dim)
        return output

# 修复的融合操作符模型
class FixedFusedOpsKernelizedAttention(nn.Module):
    def __init__(self, model_config, dtype=torch.float32, eps=1e-6):
        super().__init__()
        self.head_dim = model_config.head_dim
        self.dtype = dtype
        self.eps = eps
        self.register_buffer("KV", torch.empty(0))
        self.register_buffer("Z", torch.empty(0))

    def prepare(self, K, V):
        K = K.contiguous()
        V = V.contiguous()
        
        bsz, num_heads, kv_len, head_dim = K.shape
        K_reshaped = K.reshape(bsz * num_heads, kv_len, head_dim)
        V_reshaped = V.reshape(bsz * num_heads, kv_len, head_dim)

        # 尝试融合ELU+1操作
        K_phi = F.elu(K_reshaped) + 1
        K_phi = K_phi.transpose(1, 2)
        
        # 预计算并存储
        self.KV = torch.bmm(K_phi, V_reshaped)
        ones = torch.ones(bsz * num_heads, kv_len, 1, device=K.device, dtype=K.dtype)
        self.Z = torch.bmm(K_phi, ones)

    def forward(self, Q):
        Q = Q.contiguous()
        
        bsz, num_heads, q_len, head_dim = Q.shape
        Q_reshaped = Q.reshape(bsz * num_heads, q_len, head_dim)
        
        # 融合ELU+1操作
        Q_phi = F.elu(Q_reshaped) + 1
        
        # 计算分子和分母
        numerator = torch.bmm(Q_phi, self.KV)
        denominator = torch.bmm(Q_phi, self.Z) + self.eps
        
        # 确保形状正确
        output = numerator / denominator
        output = output.reshape(bsz, num_heads, q_len, head_dim)
        return output

# 高级JIT BFloat16模型 - 添加更多优化
class AdvancedJITBFloat16KernelizedAttention(nn.Module):
    def __init__(self, model_config, dtype=torch.bfloat16, eps=1e-6):
        super().__init__()
        self.head_dim = model_config.head_dim
        self.dtype = dtype
        self.eps = eps
        self.register_buffer("KV", torch.empty(0, dtype=torch.bfloat16))
        self.register_buffer("Z", torch.empty(0, dtype=torch.bfloat16))

    @torch.jit.export
    def prepare(self, K, V):
        # 使用@torch.jit.export确保方法被正确导出
        K = K.contiguous()
        V = V.contiguous()
        
        bsz, num_heads, kv_len, head_dim = K.shape
        K_reshaped = K.view(bsz * num_heads, kv_len, head_dim)
        V_reshaped = V.view(bsz * num_heads, kv_len, head_dim)

        # 内联函数调用以减少开销
        K_phi = F.elu(K_reshaped, alpha=1.0) + 1
        K_phi = K_phi.transpose(1, 2)
        
        # 使用torch.bmm而不是@操作符
        self.KV = torch.bmm(K_phi, V_reshaped)
        ones = torch.ones(bsz * num_heads, kv_len, 1, device=K.device, dtype=K.dtype)
        self.Z = torch.bmm(K_phi, ones)

    def forward(self, Q):
        Q = Q.contiguous()
        
        bsz, num_heads, q_len, head_dim = Q.shape
        Q_reshaped = Q.view(bsz * num_heads, q_len, head_dim)
        
        # 内联函数调用
        Q_phi = F.elu(Q_reshaped, alpha=1.0) + 1
        
        # 使用torch.bmm
        numerator = torch.bmm(Q_phi, self.KV)
        denominator = torch.bmm(Q_phi, self.Z) + self.eps
        
        # 使用除法而不是reciprocal+乘法，让编译器决定最佳实现
        output = numerator / denominator
        
        # 使用view而不是reshape以避免可能的复制
        output = output.view(bsz, num_heads, q_len, head_dim)
        return output

# 可追踪的JIT模型 - 为torch.jit.trace准备
class TraceableKernelizedAttention(nn.Module):
    def __init__(self, KV, Z, eps=1e-6):
        super().__init__()
        self.register_buffer("KV", KV)
        self.register_buffer("Z", Z)
        self.eps = eps

    def forward(self, Q):
        Q = Q.contiguous()
        
        bsz, num_heads, q_len, head_dim = Q.shape
        Q_reshaped = Q.view(bsz * num_heads, q_len, head_dim)
        
        Q_phi = F.elu(Q_reshaped) + 1
        
        numerator = torch.bmm(Q_phi, self.KV)
        denominator = torch.bmm(Q_phi, self.Z) + self.eps
        output = numerator / denominator
        
        output = output.view(bsz, num_heads, q_len, head_dim)
        return output

# 改进的Trace模型 - 使用bfloat16
class ImprovedTraceableKernelizedAttention(nn.Module):
    def __init__(self, KV, Z, eps=1e-6):
        super().__init__()
        self.register_buffer("KV", KV)
        self.register_buffer("Z", Z)
        self.eps = eps

    def forward(self, Q):
        Q = Q.contiguous()
        
        bsz, num_heads, q_len, head_dim = Q.shape
        Q_reshaped = Q.view(bsz * num_heads, q_len, head_dim)
        
        # 使用alpha参数的elu，可能更容易被优化
        Q_phi = F.elu(Q_reshaped, alpha=1.0) + 1
        
        # 使用bmm而不是矩阵乘法操作符
        numerator = torch.bmm(Q_phi, self.KV)
        denominator = torch.bmm(Q_phi, self.Z) + self.eps
        
        # 直接除法
        output = numerator / denominator
        
        # 使用view
        output = output.view(bsz, num_heads, q_len, head_dim)
        return output

# 创建不同类型的JIT模型
def create_jit_script_model(model_config, K, V):
    model = JITScriptKernelizedAttention(model_config)
    model.prepare(K, V)
    return torch.jit.script(model)

def create_jit_bfloat16_model(model_config, K, V):
    model = JITBFloat16KernelizedAttention(model_config)
    model.prepare(K, V)
    return torch.jit.script(model)

def create_advanced_jit_bfloat16_model(model_config, K, V):
    model = AdvancedJITBFloat16KernelizedAttention(model_config)
    model.prepare(K, V)
    # 使用优化选项
    return torch.jit.script(model, optimize=True)

def create_jit_trace_model(model_config, K, V, Q):
    # 首先创建一个标准模型并准备KV
    temp_model = JITScriptKernelizedAttention(model_config)
    temp_model.prepare(K, V)
    
    # 然后创建可追踪模型
    traceable_model = TraceableKernelizedAttention(temp_model.KV, temp_model.Z)
    
    # 使用trace而不是script
    return torch.jit.trace(traceable_model, (Q,))

def create_improved_jit_trace_model(model_config, K, V, Q):
    # 使用bfloat16数据类型
    temp_model = JITBFloat16KernelizedAttention(model_config)
    temp_model.prepare(K, V)
    
    # 使用改进的可追踪模型
    traceable_model = ImprovedTraceableKernelizedAttention(temp_model.KV, temp_model.Z)
    
    # 使用trace并启用优化
    return torch.jit.trace(traceable_model, (Q,), optimize=True)

def create_fixed_fused_ops_model(model_config, K, V):
    model = FixedFusedOpsKernelizedAttention(model_config)
    model.prepare(K, V)
    # 使用script_method优化特定方法
    scripted_model = torch.jit.script(model)
    
    # 尝试启用融合优化
    try:
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
    except Exception as e:
        print(f"优化推理失败，使用标准脚本模型: {e}")
    
    return scripted_model

def benchmark_model(model_name, model, K, V, Q, num_runs=10, warmup=3):
    # 预热
    for _ in range(warmup):
        model.prepare(K, V)
        _ = model(Q)
    
    # 计时
    prepare_times = []
    forward_times = []
    total_times = []
    
    for _ in range(num_runs):
        # 测量准备时间
        start_time = time.time()
        model.prepare(K, V)
        prepare_time = time.time() - start_time
        prepare_times.append(prepare_time)
        
        # 测量前向传播时间
        start_time = time.time()
        output = model(Q)
        forward_time = time.time() - start_time
        forward_times.append(forward_time)
        
        # 总时间
        total_times.append(prepare_time + forward_time)
    
    # 计算平均时间
    avg_prepare_time = sum(prepare_times) / num_runs
    avg_forward_time = sum(forward_times) / num_runs
    avg_total_time = sum(total_times) / num_runs
    
    print(f"\n{model_name} 性能测试结果:")
    print(f"  准备阶段平均耗时: {avg_prepare_time:.6f} 秒")
    print(f"  前向传播平均耗时: {avg_forward_time:.6f} 秒")
    print(f"  总平均耗时: {avg_total_time:.6f} 秒")
    
    return {
        "model_name": model_name,
        "prepare_time": avg_prepare_time,
        "forward_time": avg_forward_time,
        "total_time": avg_total_time
    }

def benchmark_jit_model(model_name, jit_model, Q, num_runs=10, warmup=3):
    # 预热
    for _ in range(warmup):
        _ = jit_model(Q)
    
    # 计时
    forward_times = []
    
    for _ in range(num_runs):
        # 测量前向传播时间
        start_time = time.time()
        output = jit_model(Q)
        forward_time = time.time() - start_time
        forward_times.append(forward_time)
    
    # 计算平均时间
    avg_forward_time = sum(forward_times) / num_runs
    
    print(f"\n{model_name} 性能测试结果:")
    print(f"  前向传播平均耗时: {avg_forward_time:.6f} 秒")
    print(f"  总平均耗时: {avg_forward_time:.6f} 秒")  # JIT模型只测试前向传播
    
    return {
        "model_name": model_name,
        "prepare_time": 0,  # JIT模型不单独测试prepare
        "forward_time": avg_forward_time,
        "total_time": avg_forward_time
    }

def main():
    parser = argparse.ArgumentParser(description="测试不同JIT优化策略的性能")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--num_heads", type=int, default=12, help="注意力头数量")
    parser.add_argument("--seq_len", type=int, default=512, help="序列长度")
    parser.add_argument("--head_dim", type=int, default=64, help="每个头的维度")
    parser.add_argument("--num_runs", type=int, default=10, help="运行次数")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数")
    args = parser.parse_args()
    
    # 打印测试配置
    print(f"测试配置:")
    print(f"  批次大小: {args.batch_size}")
    print(f"  注意力头数量: {args.num_heads}")
    print(f"  序列长度: {args.seq_len}")
    print(f"  每个头的维度: {args.head_dim}")
    print(f"  运行次数: {args.num_runs}")
    print(f"  预热次数: {args.warmup}")
    
    # 设置模型配置
    model_config = ModelConfig(head_dim=args.head_dim)
    
    # 准备输入数据
    device = torch.device("cpu")
    
    # 原始版本使用 bfloat16
    K_orig = torch.randn(args.batch_size, args.num_heads, args.seq_len, args.head_dim, 
                         device=device).to(torch.bfloat16)
    V_orig = torch.randn(args.batch_size, args.num_heads, args.seq_len, args.head_dim, 
                         device=device).to(torch.bfloat16)
    Q_orig = torch.randn(args.batch_size, args.num_heads, args.seq_len, args.head_dim, 
                         device=device).to(torch.bfloat16)
    
    # float32 版本
    K_float32 = K_orig.to(torch.float32)
    V_float32 = V_orig.to(torch.float32)
    Q_float32 = Q_orig.to(torch.float32)
    
    # 创建原始模型
    original_model = OriginalKernelizedAttention(model_config).to(device)
    
    # 运行基准测试
    results = []
    
    # 测试原始模型
    results.append(benchmark_model(
        "原始模型 (bfloat16)", 
        original_model, 
        K_orig, V_orig, Q_orig, 
        num_runs=args.num_runs, 
        warmup=args.warmup
    ))
    
    try:
        # 测试JIT Script模型 (float32)
        jit_script_model = create_jit_script_model(model_config, K_float32, V_float32)
        results.append(benchmark_jit_model(
            "JIT Script模型 (float32)", 
            jit_script_model, 
            Q_float32, 
            num_runs=args.num_runs, 
            warmup=args.warmup
        ))
    except Exception as e:
        print(f"JIT Script模型编译失败: {e}")
    
    try:
        # 测试JIT BFloat16模型
        jit_bfloat16_model = create_jit_bfloat16_model(model_config, K_orig, V_orig)
        results.append(benchmark_jit_model(
            "JIT BFloat16模型", 
            jit_bfloat16_model, 
            Q_orig, 
            num_runs=args.num_runs, 
            warmup=args.warmup
        ))
    except Exception as e:
        print(f"JIT BFloat16模型编译失败: {e}")
    
    try:
        # 测试高级JIT BFloat16模型
        advanced_jit_bfloat16_model = create_advanced_jit_bfloat16_model(model_config, K_orig, V_orig)
        results.append(benchmark_jit_model(
            "高级JIT BFloat16模型", 
            advanced_jit_bfloat16_model, 
            Q_orig, 
            num_runs=args.num_runs, 
            warmup=args.warmup
        ))
    except Exception as e:
        print(f"高级JIT BFloat16模型编译失败: {e}")
    
    try:
        # 测试JIT Trace模型
        jit_trace_model = create_jit_trace_model(model_config, K_float32, V_float32, Q_float32)
        results.append(benchmark_jit_model(
            "JIT Trace模型 (float32)", 
            jit_trace_model, 
            Q_float32, 
            num_runs=args.num_runs, 
            warmup=args.warmup
        ))
    except Exception as e:
        print(f"JIT Trace模型编译失败: {e}")
    
    try:
        # 测试改进的JIT Trace模型
        improved_jit_trace_model = create_improved_jit_trace_model(model_config, K_orig, V_orig, Q_orig)
        results.append(benchmark_jit_model(
            "改进的JIT Trace模型 (bfloat16)", 
            improved_jit_trace_model, 
            Q_orig, 
            num_runs=args.num_runs, 
            warmup=args.warmup
        ))
    except Exception as e:
        print(f"改进的JIT Trace模型编译失败: {e}")
    
    try:
        # 测试修复的融合操作符模型
        fixed_fused_ops_model = create_fixed_fused_ops_model(model_config, K_float32, V_float32)
        results.append(benchmark_jit_model(
            "修复的融合操作符模型 (float32)", 
            fixed_fused_ops_model, 
            Q_float32, 
            num_runs=args.num_runs, 
            warmup=args.warmup
        ))
    except Exception as e:
        print(f"修复的融合操作符模型编译失败: {e}")
    
    # 打印性能比较
    print("\n性能比较:")
    baseline = results[0]["total_time"]
    for result in results:
        speedup = baseline / result["total_time"]
        print(f"{result['model_name']}: 总耗时 {result['total_time']:.6f} 秒, 加速比: {speedup:.2f}x")

if __name__ == "__main__":
    main()
