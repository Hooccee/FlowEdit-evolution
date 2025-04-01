import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
from diffusers import StableDiffusion3Pipeline, FluxPipeline
from PIL import Image
import argparse
import random 
import numpy as np
import yaml
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from FlowEdit_utils import FlowEditSD3, FlowEditFLUX
from datasets import get_dataloader
from utils.utils import *
from utils.metrics import *

# 自定义图像转换类（保持0-255范围）
class ToTensorWithoutScaling(object):
    """将PIL图像转换为张量而不进行归一化"""
    def __call__(self, pic):
        if pic is None:
            raise ValueError("输入图像为空")
        if isinstance(pic, Image.Image):
            pic.load()  # 确保图像已加载
            # 转换为字节张量并调整形状
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
            img = img.permute((2, 0, 1)).contiguous()  # 转为CHW格式
            return img.to(dtype=torch.uint8)
        else:
            raise TypeError(f"输入类型错误: {type(pic)}")

# 张量转PIL图像函数
def tensor_to_pil_uint8(tensor: torch.Tensor) -> Image.Image:
    """将张量转换回PIL图像"""
    tensor = tensor.detach().cpu()
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)  # CHW → HWC
    elif tensor.dim() == 4:
        tensor = tensor.squeeze(0).permute(1, 2, 0)  # 去除批次维度
    return Image.fromarray(tensor.numpy())

def main():
    # 1. 参数设置 ##################################################
    parser = argparse.ArgumentParser(description="图像编辑批量处理与评估")
    
    # 数据参数

    parser.add_argument("--device_number", type=int, default=0, help="device number to use")
    parser.add_argument("--dataset", type=str, default='EditEval_v1', 
                       help="选择数据集: EditEval_v1, PIE-Bench")
    parser.add_argument("--height", type=int, default=512, help="输出图像高度")
    parser.add_argument("--width", type=int, default=512, help="输出图像宽度")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    
    # 模型参数
    parser.add_argument("--model_type", type=str, default="FLUX",  
                       help="模型类型: FLUX 或 SD3")
    parser.add_argument("--model_path", type=str, 
                       default="/data/chx/FLUX.1-dev", 
                       help="模型路径")
    
    # 编辑参数
    parser.add_argument("--T_steps", type=int, default=28, help="总步数")  
    parser.add_argument("--n_avg", type=int, default=1, help="平均次数") 
    parser.add_argument("--src_guidance_scale", type=float, default=1.5,  
                       help="源图像引导尺度")
    parser.add_argument("--tar_guidance_scale", type=float, default=5.5,  
                       help="目标图像引导尺度")
    parser.add_argument("--n_min", type=int, default=0, help="最小步数")  
    parser.add_argument("--n_max", type=int, default=24, help="最大步数")  
    
    # 评估参数
    parser.add_argument("--eval_metrics", action="store_true", 
                       help="是否进行评估指标计算")
    parser.add_argument("--save_samples", action="store_true", 
                       help="是否保存样本图像")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                       help="输出目录")

    args = parser.parse_args()

    # 2. 设备设置 ##################################################
    device = torch.device(f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. 数据加载 ##################################################
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),  # 调整尺寸
        ToTensorWithoutScaling()  # 转为张量(0-255)
    ])
    
    # 获取数据加载器
    dataset = get_dataloader(
        dataset_name=args.dataset,
        default_transform=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,  # 批大小
        shuffle=False,  # 不随机打乱
        num_workers=args.num_workers  # 工作线程数
    )


    # 4. 模型加载 ##################################################
    if args.model_type == 'FLUX':
        pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    elif args.model_type == 'SD3':
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            torch_dtype=torch.float16)
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    # 启用CPU卸载以节省显存
    pipe.enable_sequential_cpu_offload()
    
    # 5. 评估指标初始化 ############################################
    if args.eval_metrics:
        metrics = {
            'clip_score': 0.0,  # 图像-文本对齐度
            'clip_score_i': 0.0,  # 图像-图像对齐度
            'mse': 0.0,         # 均方误差
            'psnr': 0.0,         # 峰值信噪比
            'lpips': 0.0,        # 感知相似度
            'ssim': 0.0,         # 结构相似性
            'dino': 0.0,         # 高级特征相似度
            'count': 0           # 样本计数
        }
        metric_calculator = metircs()  # 指标计算器

    # 6. 处理循环 ##################################################
    progress = tqdm(dataloader, desc="Processing")
    for batch_idx, (images, source_prompts, target_prompts) in enumerate(progress):
        batch_metrics = {
            'clip_score': 0.0,
            'clip_score_i': 0.0,
            'mse': 0.0,
            'psnr': 0.0,
            'lpips': 0.0,
            'ssim': 0.0,
            'dino': 0.0,
            'count': 0
        }
        
        for idx in range(len(images)):
            #try:
                # 6.1 准备输入图像 ---------------------------------
                # 将张量转为PIL图像
                init_image_pil = tensor_to_pil_uint8(images[idx])
                # 确保尺寸是16的倍数
                width = init_image_pil.width - init_image_pil.width % 16
                height = init_image_pil.height - init_image_pil.height % 16
                init_image_pil = init_image_pil.crop((0, 0, width, height))
                
                # 预处理图像
                image_src = pipe.image_processor.preprocess(init_image_pil)
                image_src = image_src.to(device).half()
                
                # 编码到潜在空间
                with torch.autocast("cuda"), torch.inference_mode():
                    x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
                x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                x0_src = x0_src.to(device)
                
                # 6.2 处理每个目标提示 -----------------------------

                # 执行图像编辑
                if args.model_type == 'SD3':
                    x0_tar = FlowEditSD3(
                        pipe, pipe.scheduler, x0_src,
                        source_prompts[idx], target_prompts[idx], "",
                        args.T_steps, args.n_avg, 
                        args.src_guidance_scale, args.tar_guidance_scale,
                        args.n_min, args.n_max
                    )
                else:  # FLUX
                    x0_tar = FlowEditFLUX(
                        pipe, pipe.scheduler, x0_src,
                        source_prompts[idx], target_prompts[idx], "",
                        args.T_steps, args.n_avg, 
                        args.src_guidance_scale, args.tar_guidance_scale,
                        args.n_min, args.n_max
                    )
                
                # 解码回像素空间
                x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                with torch.autocast("cuda"), torch.inference_mode():
                    image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
                edited_img = pipe.image_processor.postprocess(image_tar)[0]
                
                # 6.3 计算评估指标 ----------------------------
                if args.eval_metrics:
                    # 转换图像为评估用的张量
                    edited_tensor = transforms.Compose([
                        transforms.Resize((args.height, args.width)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5])
                    ])(edited_img).unsqueeze(0).to(device)
                    
                    orig_tensor = transforms.Compose([
                        transforms.Resize((args.height, args.width)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5])
                    ])(init_image_pil).unsqueeze(0).to(device)
                    
                    # 计算各项指标
                    clip_score = metric_calculator.clip_scores(edited_tensor, target_prompts[idx])
                    clip_score_i = metric_calculator.clip_scores(edited_tensor, orig_tensor)
                    mse = metric_calculator.mse_scores(edited_tensor, orig_tensor)
                    psnr_val = metric_calculator.psnr_scores(edited_tensor, orig_tensor)
                    lpips_val = metric_calculator.lpips_scores(edited_tensor, orig_tensor)
                    ssim_val = metric_calculator.ssim_scores(edited_tensor, orig_tensor)
                    dino_val = metric_calculator.dino_scores(edited_tensor, orig_tensor)
                    
                    # 打印样本指标
                    print(f"\n样本 {batch_idx}-{idx} 指标:")
                    print(f"源提示: {source_prompts[idx]}")
                    print(f"目标提示: {target_prompts[idx]}")
                    print(f"CLIP-T分数: {clip_score:.4f}")
                    print(f"CLIP-I分数: {clip_score_i:.4f}")
                    print(f"MSE: {mse:.4f}")
                    print(f"PSNR: {psnr_val:.4f} dB")
                    print(f"LPIPS: {lpips_val:.4f}")
                    print(f"SSIM: {ssim_val:.4f}")
                    print(f"DINO: {dino_val:.4f}")
                    
                    # 更新批次指标
                    batch_metrics['clip_score'] += clip_score
                    batch_metrics['clip_score_i'] += clip_score_i
                    batch_metrics['mse'] += mse
                    batch_metrics['psnr'] += psnr_val
                    batch_metrics['lpips'] += lpips_val
                    batch_metrics['ssim'] += ssim_val
                    batch_metrics['dino'] += dino_val
                    batch_metrics['count'] += 1
                
                # 6.4 保存结果 -------------------------------
                if args.save_samples:
                    # 创建保存目录
                    src_prompt_txt = source_prompts[idx][:20].replace(" ", "_")
                    target_prompts_txt = target_prompts[idx][:20].replace(" ", "_")
                    save_dir = os.path.join(
                        args.output_dir, 
                        f"batch_{batch_idx}", 
                        f"sample_{idx}"
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # 保存编辑后的图像
                    edited_img.save(os.path.join(save_dir, f"edited_.png"))
                    # 保存原始图像
                    init_image_pil.save(os.path.join(save_dir, "original.png"))
                    # 保存提示文本
                    with open(os.path.join(save_dir, "prompts.txt"), "w") as f:
                        f.write(f"Source: {source_prompts[idx]}\n")
                        f.write(f"Target: {target_prompts[idx]}\n")
            
            # except Exception as e:
            #     print(f"处理样本 {batch_idx}-{idx} 时出错: {str(e)}")
            #     continue
        
        # 6.5 打印批次指标摘要 --------------------------------
        if args.eval_metrics and batch_metrics['count'] > 0:
            print(f"\n{'='*40}")
            print(f"批次 {batch_idx} 指标摘要:")
            
            # 计算并更新总指标
            for key in ['clip_score','clip_score_i', 'mse', 'psnr', 'lpips', 'ssim', 'dino']:
                batch_avg = batch_metrics[key] / batch_metrics['count']
                metrics[key] += batch_avg * batch_metrics['count']
                
                print(f"[{key.upper()}]:")
                print(f"  批次平均: {batch_avg:.4f}")
                print(f"  累计总值: {metrics[key]:.4f}")
                print("-"*30)
            
            # 更新总样本数
            metrics['count'] += batch_metrics['count']
        
        # 清理显存
        torch.cuda.empty_cache()
    
    # 7. 最终评估结果 #############################################
    if args.eval_metrics and metrics['count'] > 0:
        print("\n最终评估结果:")
        print(f"总样本数: {metrics['count']}")
        for key in ['clip_score','clip_score_i', 'mse', 'psnr', 'lpips', 'ssim', 'dino']:
            final_metric = metrics[key] / metrics['count']
            print(f"{key.upper():<10} | {final_metric:.4f}")

    print("处理完成")

if __name__ == "__main__":
    main()