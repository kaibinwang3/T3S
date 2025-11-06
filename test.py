import torch
import time
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def run_inference_benchmark():
    """
    比较 Qwen 模型在两种不同输入形状下的单次前向传播（Prefill 阶段）耗时：
    1. Batch Size = 1, Sequence Length = bsz * n
    2. Batch Size = bsz, Sequence Length = n
    """
    # --- 1. 配置 ---
    n = 1000  # 基础序列长度
    bsz = 4
    # max_new_tokens 在此版本中不再需要，因为我们只做一次 forward pass
    warmup_iterations = 1
    test_iterations = 1

    print("--- 配置 ---")
    print(f"基础序列长度 (n): {n}")
    print(f"批处理大小 (bsz): {bsz}")
    print(f"预热轮数: {warmup_iterations}")
    print(f"测试轮数: {test_iterations}\n")

    # --- 2. 加载模型和处理器 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print("警告: 未检测到 CUDA，将在 CPU 上运行。计时结果不代表 GPU 性能。")
        n = 64
        print(f"已为 CPU 执行将 'n' 调整为 {n}。")


    print(f"正在加载模型到 {device}...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/mnt/afs/wangkaibin/models/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("/mnt/afs/wangkaibin/models/Qwen2.5-VL-7B-Instruct")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保您已安装 transformers, accelerate 等必要库，并根据需要登录到 huggingface。")
        return

    model.eval()

    # --- 3. 准备虚拟输入数据 ---
    # 场景 1: bs=1, seq_len=bsz*n
    # 总 token 数 = 1 * 8 * 10000 = 80000
    input_ids_1 = torch.randint(0, model.config.vocab_size, (1, bsz * n), device=device)
    attention_mask_1 = torch.ones_like(input_ids_1)

    # 场景 2: bs=bsz, seq_len=n
    # 总 token 数 = 8 * 1 * 10000 = 80000
    input_ids_2 = torch.randint(0, model.config.vocab_size, (bsz, n * 2), device=device)
    attention_mask_2 = torch.ones_like(input_ids_2)

    print("\n--- 开始基准测试 (仅 Prefill/Forward 阶段) ---")

    # --- 4. 预热 ---
    print("正在预热 GPU...")
    with torch.no_grad():
        for _ in range(warmup_iterations):
            # 修改: 直接调用 model() 来预热 forward pass
            _ = model(input_ids=input_ids_1, attention_mask=attention_mask_1)
            _ = model(input_ids=input_ids_2, attention_mask=attention_mask_2)
    if device == 'cuda':
        torch.cuda.synchronize()

    # --- 5. 测试场景 1 (bs=1, seq_len=bsz*n) ---
    total_time_1 = 0
    print(f"\n正在运行测试场景 1 (batch_size=1, seq_len={bsz*n})...")
    with torch.no_grad():
        for i in range(test_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()

            # 修改: 直接调用 model() 只测量 prefill 阶段的耗时
            # 这会返回 logits，我们忽略其输出，只关心时间
            _ = model(input_ids=input_ids_1, attention_mask=attention_mask_1)

            end_time = time.time()
            if device == 'cuda':
                torch.cuda.synchronize()
            total_time_1 += (end_time - start_time)
    avg_time_1 = total_time_1 / test_iterations

    # --- 6. 测试场景 2 (bs=bsz, seq_len=n) ---
    total_time_2 = 0
    print(f"正在运行测试场景 2 (batch_size={bsz}, seq_len={n})...")
    with torch.no_grad():
        for i in range(test_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()

            # 修改: 同样，只调用 forward()
            _ = model(input_ids=input_ids_2, attention_mask=attention_mask_2)

            end_time = time.time()
            if device == 'cuda':
                torch.cuda.synchronize()
            total_time_2 += (end_time - start_time)
    avg_time_2 = total_time_2 / test_iterations

    # --- 7. 打印结果 ---
    print("\n--- 基准测试结果 ---")
    print(f"场景1 (batch_size=1, seq_len={bsz*n}) 平均耗时: {avg_time_1:.4f} 秒")
    print(f"场景2 (batch_size={bsz}, seq_len={n}) 平均耗时: {avg_time_2:.4f} 秒")

    if avg_time_2 > 0 and avg_time_1 > avg_time_2:
        print(f"\n结论: 场景2 速度是场景1的 {avg_time_1 / avg_time_2:.2f} 倍。")
        print("这个结果更准确地反映了 Prefill 阶段的性能，批处理（Batching）在此阶段能显著提升计算效率。")
    elif avg_time_1 > 0 and avg_time_2 > avg_time_1:
        print(f"\n结论: 场景1 速度是场景2的 {avg_time_2 / avg_time_1:.2f} 倍。")
    else:
        print("\n结论: 耗时过短或为零，无法计算有意义的加速比。")


# 运行基准测试函数
if __name__ == "__main__":
    run_inference_benchmark()
