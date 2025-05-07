from typing import Optional, Tuple, Union
import torch.nn as nn
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import numpy as np
import accelerate


from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from torchvision import transforms



from utils.metrics import *

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps



def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Foward process in flow-matching

    Args:
        sample (`torch.FloatTensor`):
            The input sample.
        timestep (`int`, *optional*):
            The current timestep in the diffusion chain.

    Returns:
        `torch.FloatTensor`:
            A scaled input sample.
    """
    # if scheduler.step_index is None:
    scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample


# for flux
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu



def calc_v_sd3(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(src_tar_latent_model_input.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i


    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # perform guidance source
        if pipe.do_classifier_free_guidance:
            src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_src_tar.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
            noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar



def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(latents.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i


    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred



class DifferentiableMetrics:  # 输入图像值范围均为[-1,1]
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载CLIP模型
        try:
            import clip
            self.clip = clip  # 保存为类属性
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            # self.clip_model.eval()
        except ImportError:
            print("请安装OpenAI CLIP库: pip install git+https://github.com/openai/CLIP.git")
            self.clip_model = None
        
        # 加载DINOv2模型和处理器
        model_folder = '/data/chx/dinov2-base'
        self.dino_processor = AutoImageProcessor.from_pretrained(model_folder)
        self.dino_model = AutoModel.from_pretrained(model_folder).to(self.device)
        self.dino_model.eval()

        # 加载LPIPS VGG模型
        self.lpips_vgg = self._load_lpips_vgg().to(self.device)
    
    def _load_lpips_vgg(self):
        """加载预训练的VGG模型用于LPIPS计算"""
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features
            vgg_layers = [0, 4, 9, 16, 23, 30]
            model = nn.ModuleList()
            for i in range(len(vgg_layers)-1):
                model.append(vgg[vgg_layers[i]:vgg_layers[i+1]])
            return nn.Sequential(*model)
        except:
            print("无法加载VGG模型,LPIPS度量可能不可用")
            return nn.Sequential()  # 返回空模型

    def _tensor_to_pil(self, tensor):
        """将[-1,1]范围的Tensor转换为PIL图像"""
        # 处理可能的批量维度（当输入是4维时）
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # 从[B C H W] -> [C H W]
        
        # 转换为[0,1]范围
        image_01 = (tensor + 1) / 2
        # 调整维度顺序并转换为uint8
        image_uint8 = (image_01.permute(1, 2, 0) * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
        # 转换为PIL图像
        return Image.fromarray(image_uint8)

    def _normalize_to_0_1(self, tensor):
        """将[-1,1]归一化到[0,1]"""
        return (tensor + 1) / 2
    
    def mse_scores(self, image1, image2):
        """可微分MSE评分"""
        # 确保输入为连续内存
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        
        # 直接计算均方误差
        mse = F.mse_loss(image1, image2)
        return mse
    
    def psnr_scores(self, image1, image2):
        """可微分PSNR评分,图像范围[-1,1],data_range=2.0"""
        # 计算MSE
        mse = self.mse_scores(image1, image2)
        # 计算PSNR（峰值信噪比）
        data_range = 2.0  # 因为输入在[-1,1]范围
        psnr = 10 * torch.log10(data_range**2 / mse)
        return psnr
    
    def ssim_scores(self, image1, image2):
        """可微分SSIM评分,图像范围[-1,1]"""
        # 转换到[0,1]范围
        img1 = self._normalize_to_0_1(image1)
        img2 = self._normalize_to_0_1(image2)
        
        # SSIM参数
        C1 = (0.01 * 1) ** 2
        C2 = (0.03 * 1) ** 2
        kernel_size = 11
        sigma = 1.5
        
        # 创建高斯核
        coords = torch.arange(kernel_size, device=img1.device).float() - kernel_size // 2
        x = coords.repeat(kernel_size, 1)
        y = x.t()
        gaussian_kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(img1.size(1), 1, 1, 1)
        
        # 应用卷积获取均值和方差
        padding = kernel_size // 2
        mu1 = F.conv2d(img1, gaussian_kernel, padding=padding, groups=img1.size(1))
        mu2 = F.conv2d(img2, gaussian_kernel, padding=padding, groups=img2.size(1))
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, gaussian_kernel, padding=padding, groups=img1.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, gaussian_kernel, padding=padding, groups=img2.size(1)) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, gaussian_kernel, padding=padding, groups=img1.size(1)) - mu1_mu2
        
        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    
    def lpips_scores(self, image1, image2):
        """可微分LPIPS评分"""
        # 转换输入范围从[-1,1]到[0,1]
        img1 = self._normalize_to_0_1(image1)
        img2 = self._normalize_to_0_1(image2)
        
        # 需要重新调整大小到224x224以匹配VGG输入
        if img1.size(-1) != 224 or img1.size(-2) != 224:
            img1 = F.interpolate(img1, size=(224, 224), mode='bilinear', align_corners=False)
        if img2.size(-1) != 224 or img2.size(-2) != 224:
            img2 = F.interpolate(img2, size=(224, 224), mode='bilinear', align_corners=False)
            
        # 从[0,1]转到ImageNet归一化
        mean = torch.tensor([0.485, 0.456, 0.406], device=img1.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img1.device).view(1, 3, 1, 1)
        img1 = (img1 - mean) / std
        img2 = (img2 - mean) / std
        
        # 计算每层特征的距离
        
        dist = 0
        for layer in self.lpips_vgg:
            img1 = layer(img1)
            img2 = layer(img2)
            dist += F.l1_loss(img1, img2)
        
        return dist


    def dino_transform(self,image: torch.Tensor) -> torch.Tensor:
        """
        可微分图像预处理流程（适配dino模型，输入范围[-1,1]）
        输入: 张量范围[-1,1], 支持形状 (C,H,W) 或 (B,C,H,W)
        输出: 标准化后的张量，形状 (C,224,224) 或 (B,C,224,224)
        """
        # 确保输入在[-1,1]范围内（可导操作）
        image = torch.tanh(image) * 0.9999  # 软截断，避免梯度爆炸
        
        # 统一处理批次维度
        is_batched = image.ndim == 4
        if not is_batched:
            image = image.unsqueeze(0)  # [B,C,H,W]

        # Step 1: 调整大小（短边缩放到256，保持宽高比）
        B, C, H, W = image.shape
        shortest_edge = min(H, W)
        scale = 256.0 / shortest_edge
        new_H = int(round(H * scale))
        new_W = int(round(W * scale))
        
        # 双三次插值调整大小
        resized = F.interpolate(
            image, 
            size=(new_H, new_W), 
            mode='bicubic', 
            align_corners=False
        )

        # Step 2: 中心裁剪224x224
        _, _, H_resized, W_resized = resized.shape
        start_y = (H_resized - 224) // 2
        start_x = (W_resized - 224) // 2
        cropped = resized[..., start_y:start_y+224, start_x:start_x+224]

        # Step 3: 缩放 + 归一化（数学等价原流程）
        scaled = cropped * 0.5  # 将[-1,1]映射到[-0.5,0.5]
        
        # BiT标准化参数（原均值减去0.5）
        mean = torch.tensor([-0.015, -0.044, -0.094], 
                        dtype=scaled.dtype, 
                        device=scaled.device).view(1, 3, 1, 1)
        
        std = torch.tensor([0.229, 0.224, 0.225], 
                        dtype=scaled.dtype, 
                        device=scaled.device).view(1, 3, 1, 1)
        
        normalized = (scaled - mean) / std

        # 恢复原始形状
        if not is_batched:
            normalized = normalized.squeeze(0)

        return normalized

    def dino_scores(self, image1, image2):
        """计算两幅图像之间的DINO特征相似度"""
        # 转换Tensor到PIL图像
        # image1_pil = self._tensor_to_pil(image1)
        image2_pil = self._tensor_to_pil(image2)
        
        # 处理图像并提取特征

        # inputs1 = self.dino_processor(images=image1_pil, return_tensors="pt").to(self.device)
        inputs1 = self.dino_transform(image1)
        inputs2 = self.dino_processor(images=image2_pil, return_tensors="pt").to(self.device)
        
        outputs1 = self.dino_model(inputs1)
        outputs2 = self.dino_model(**inputs2)
    
        # 提取并平均特征
        features1 = outputs1.last_hidden_state.mean(dim=1)
        features2 = outputs2.last_hidden_state.mean(dim=1)
        
        # 计算余弦相似度并归一化
        sim = F.cosine_similarity(features1, features2, dim=1)
        return (sim + 1) / 2  # 归一化到[0,1]
    
    
    def clip_transform(self, image):
        """
        将[-1,1]范围的张量直接转换为CLIP期望的归一化格式
        输入: 张量范围[-1,1], 形状可以是 (C,H,W) 或 (B,C,H,W)
        输出: 归一化后的张量
        """
        # 确保输入在[-1,1]范围内（可导操作）
        image = torch.tanh(image) * 0.9999  # 软截断，避免梯度爆炸
        
        # 转换到[0,1]范围
        image_01 = (image + 1) / 2.0
        
        # 调整尺寸到224x224（CLIP-ViT的标准输入）
        if image.dim() == 4:  # 批处理模式 [B,C,H,W]
            image_resized = F.interpolate(image_01, size=(224,224), mode='bicubic')
        else:  # 单图像模式 [C,H,W]
            image_resized = F.interpolate(image_01.unsqueeze(0), size=(224,224), mode='bicubic').squeeze(0)
        
        # CLIP标准化参数
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], 
                        device=image.device).view(-1, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], 
                        device=image.device).view(-1, 1, 1)
        
        # 应用标准化
        return (image_resized - mean) / std

    def clip_scores(self, image, txt):
        """使用CLIP模型计算图像和文本/图像的相似度"""
        if self.clip_model is None:
            return torch.tensor(0.0)
            
        # 将图像从[-1,1]转换到CLIP期望的格式
        image_clip_transform = self.clip_transform(image)
        

        if isinstance(txt, torch.Tensor):  # 图像-图像相似度
            text_pil = self._tensor_to_pil(txt)
            image_features = self.clip_model.encode_image(image_clip_transform.to(self.device))
            text_features = self.clip_model.encode_image(self.clip_preprocess(text_pil).unsqueeze(0).to(self.device))
        else:  # 图像-文本相似度
            image_features = self.clip_model.encode_image(image_clip_transform.to(self.device))
            text_features = self.clip_model.encode_text(self.clip.tokenize(txt).to(self.device))
            
        # 归一化特征
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 计算相似度分数
        similarity = (image_features @ text_features.T).squeeze()
            
        return similarity.mean()#.item()


class MetricGuidance:
    def __init__(self, pipe, device):
        self.pipe = pipe
        self.device = device
        self.metric_calculator = DifferentiableMetrics()  # 指标计算器
        
    def compute_guidance_grad(self, z_fe, src_img, tar_prompt,orig_height, orig_width):
        """
        计算指标引导梯度 (完整7指标版本)
        Args:
            z_fe: 当前编辑的潜在变量 (B,C,H,W)
            x_src: 源图像潜在变量
            src_img: 原始图像张量(已归一化)
            tar_prompt: 目标文本提示
        Returns:
            guidance_grad: 综合指标梯度 (B,C,H,W)
            metrics: 各指标值字典
        """
        with torch.enable_grad():
            # 启用梯度计算
            z_fe = z_fe.detach().requires_grad_(True)
            
            # 解包潜在变量为图像格式
            unpacked_out = self.pipe._unpack_latents(z_fe, orig_height, orig_width, self.pipe.vae_scale_factor)
            
            # 解码回像素空间
            x0_tar_denorm = (unpacked_out / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
            # with torch.autocast("cuda"):
            # self.pipe.vae.to("cpu")
            # print("VAE device:", self.pipe.vae.device)  # 如果 vae 本身有 device 属性
            # # 或者更详细地检查 encoder/decoder 的设备
            # print("VAE encoder device:", next(self.pipe.vae.encoder.parameters()).device)
            # print("VAE decoder device:", next(self.pipe.vae.decoder.parameters()).device)

            # x0_tar_denorm = x0_tar_denorm.to("cpu")
            # accelerate.hooks.remove_hook_from_module(self.pipe.vae)


            image_tar = self.pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            image_tar = torch.clamp(image_tar, -1.0, 1.0) 
            edited_img = image_tar
            # edited_img = self.pipe.image_processor.postprocess(image_tar)[0]

            edited_img= edited_img.to(self.device)
            # self.pipe.enable_model_cpu_offload()
            # print(f"范围: [{edited_img.min().item():.3f}, {edited_img.max().item():.3f}]")

            # 转换图像为评估用的张量
            edited_tensor = transforms.Compose([
                # transforms.Resize((orig_height, orig_width)),
                # transforms.ToTensor(),
                transforms.Normalize([0], [1])
            ])(edited_img).to(self.device)

            # print(f"范围: [{edited_tensor.min().item():.3f}, {edited_tensor.max().item():.3f}]")
            
            orig_tensor = transforms.Compose([
                transforms.Resize((orig_height, orig_width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])(src_img).unsqueeze(0).to(self.device)

            # edited_tensor = edited_tensor.squeeze()
            # orig_tensor = orig_tensor.squeeze()
            edited_tensor=edited_tensor.float()
            # 计算所有指标
            metrics = {
                'clip': self.metric_calculator.clip_scores(edited_tensor, tar_prompt),
                'clip_i': self.metric_calculator.clip_scores(edited_tensor, orig_tensor),
                'mse': self.metric_calculator.mse_scores(edited_tensor, orig_tensor),
                'psnr': self.metric_calculator.psnr_scores(edited_tensor, orig_tensor),
                'lpips': self.metric_calculator.lpips_scores(edited_tensor, orig_tensor),
                'ssim': self.metric_calculator.ssim_scores(edited_tensor, orig_tensor),
                'dino': self.metric_calculator.dino_scores(edited_tensor, orig_tensor)
            }

            del self.metric_calculator
            torch.cuda.empty_cache()

            # 构建多目标损失函数 (可配置权重)
            loss = (
                1.0 * (1 - metrics['clip']) +    # 最大化文本对齐
                0.8 * (1-metrics['clip_i'])          # 保持图像相似性
                # 0.5 * metrics['lpips']          # 最小化感知差异
                # 0.3 * metrics['mse'] +           # 降低像素误差
                # 1.0 * metrics['psnr']           # 提高峰值信噪比
                # 0.2 * (1 - metrics['ssim'])     # 提高结构相似性
                # 0.1 * (1 - metrics['dino'])      # 增强高级特征匹配
            )
            
            # 反向传播计算梯度
            loss.backward()

            print("clip:", metrics['clip'].item())
            print("clip_i:", metrics['clip_i'].item())
            print("mse:", metrics['mse'].item())
            print("psnr:", metrics['psnr'].item())
            print("lpips:", metrics['lpips'].item())
            print("ssim:", metrics['ssim'].item())
            print("dino:", metrics['dino'].item())
            # 在 loss.backward() 后添加梯度打印逻辑
            if z_fe.grad is not None:
                # 打印梯度基本信息
                print("\n梯度详细信息:")
                print(f"梯度形状: {z_fe.grad.shape}")
                print(f"梯度数据类型: {z_fe.grad.dtype}")
                print(f"梯度设备位置: {z_fe.grad.device}")
                
                # 打印统计信息
                print(f"绝对值均值: {z_fe.grad.abs().mean().item():.6f}")
                print(f"标准差: {z_fe.grad.std().item():.6f}")
                print(f"最大值: {z_fe.grad.max().item():.6f}")
                print(f"最小值: {z_fe.grad.min().item():.6f}")
                print(f"L2范数: {z_fe.grad.norm().item():.6f}")  # 整体梯度大小
                
                # 检查异常值
                print(f"NaN数量: {torch.isnan(z_fe.grad).sum().item()}")
                print(f"Inf数量: {torch.isinf(z_fe.grad).sum().item()}")
                
                # 可选：打印前10个元素的梯度值
                # print("前10个梯度值:", z_fe.grad.flatten()[:10].cpu().numpy())
            else:
                print("警告：梯度为None，未成功计算梯度！")
            return z_fe.grad.data.clone(), metrics

@torch.no_grad()
def FlowEditSD3(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,
    orig_height: int = 1024,
    orig_width: int = 1024,
    ):
    
    device = x_src.device

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)

    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    pipe._guidance_scale = src_guidance_scale
    
    # src prompts
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )
 
    # CFG prep
    src_tar_prompt_embeds = torch.cat([src_negative_prompt_embeds, src_prompt_embeds, tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat([src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)
    
    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):
        
        if T_steps - i > n_max:
            continue
        
        t_i = t/1000
        if i+1 < len(timesteps): 
            t_im1 = (timesteps[i+1])/1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)
        
        if T_steps - i > n_min:

            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src)
            for k in range(n_avg):

                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                
                zt_src = (1-t_i)*x_src + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_model_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar]) if pipe.do_classifier_free_guidance else (zt_src, zt_tar) 

                Vt_src, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input,src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

                V_delta_avg += (1/n_avg) * (Vt_tar - Vt_src) # - (hfg-1)*( x_src))

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else: # i >= T_steps-n_min # regular sampling for last n_min steps

            if i == T_steps-n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src
                
            src_tar_latent_model_input = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar]) if pipe.do_classifier_free_guidance else (xt_src, xt_tar)

            _, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input,src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(noise_pred_tar.dtype)

            xt_tar = prev_sample
        
    return zt_edit if n_min == 0 else xt_tar



@torch.no_grad()  
def FlowEditFLUX(pipe,
                scheduler,
                x_src,          # 源图像在潜在空间中的表示
                src_prompt,     # 源图像的文本提示
                tar_prompt,     # 目标图像的文本提示
                negative_prompt, # 负面提示（代码中未使用）
                T_steps: int = 28,    # 总扩散步数
                n_avg: int = 1,       # 速度场平均次数
                src_guidance_scale: float = 1.5,  # 源提示的引导强度
                tar_guidance_scale: float = 5.5,  # 目标提示的引导强度
                n_min: int = 0,       # 最小常规采样步数
                n_max: int = 24,      # 最大ODE编辑步数
                orig_height: int = 1024,  # 原始图像高度
                orig_width: int = 1024,   # 原始图像宽度
                init_image_pil=None,  # 原始图像（PIL格式）
                ):     

    # 设备设置和图像尺寸获取

    device = x_src.device
    num_channels_latents = pipe.transformer.config.in_channels // 4  # 潜在变量通道数

    # 验证输入参数合法性
    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    # 准备源图像的潜在变量

    x_src, latent_src_image_ids = pipe.prepare_latents(
        batch_size=x_src.shape[0],
        num_channels_latents=num_channels_latents,
        height=orig_height,
        width=orig_width,
        dtype=x_src.dtype,
        device=device,
        generator=None,
        latents=x_src  # 直接使用输入的潜在变量
    )
    # 将潜在变量打包为序列形式
    x_src_packed = pipe._pack_latents(x_src, x_src.shape[0], num_channels_latents, x_src.shape[2], x_src.shape[3])
    latent_tar_image_ids = latent_src_image_ids  # 目标图像ID保持与源相同

    # 准备时间步长参数
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)  # 噪声级别序列
    image_seq_len = x_src_packed.shape[1]  # 图像序列长度
    # 计算位移参数mu
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    # 获取时间步长和调整后的总步数
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
    )
    
    # 预热步数设置
    num_warmup_steps = max(len(timesteps) - T_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    # pipe.text_encoder.to('cuda')
    # pipe.text_encoder_2.to('cuda')
    # 编码源提示文本
    (
        src_prompt_embeds,        # 源提示的嵌入表示
        src_pooled_prompt_embeds, # 池化后的源提示嵌入
        src_text_ids,             # 源文本的token ID
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,            #默认与prompt一致
        # device=device,
    )

    # 编码目标提示文本
    pipe._guidance_scale = tar_guidance_scale  # 设置目标引导强度
    (
        tar_prompt_embeds,        # 目标提示的嵌入表示
        tar_pooled_prompt_embeds, # 池化后的目标提示嵌入
        tar_text_ids,             # 目标文本的token ID
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,            #默认与prompt一致
        # device=device,
    )

    # 处理引导参数
    if pipe.transformer.config.guidance_embeds:  # 如果模型支持引导嵌入
        src_guidance = torch.tensor([src_guidance_scale], device=device).expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device).expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    # 初始化ODE的编辑状态
    zt_edit = x_src_packed.clone()  # 初始化为源潜在变量


    # pipe.vae.to('cpu')
    # pipe.text_encoder.to('cpu')
    # pipe.text_encoder_2.to('cpu')
    torch.cuda.empty_cache() 


    # 主循环：迭代处理每个时间步
    for i, t in tqdm(enumerate(timesteps)):
        
        # 跳过超出n_max范围的步骤
        if T_steps - i > n_max:
            continue
        
        # 初始化调度器步索引
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]  # 当前时间步的sigma值
        t_im1 = scheduler.sigmas[scheduler.step_index + 1] if i < len(timesteps) else t_i  # 下一时间步sigma值
        
        

        # ODE编辑阶段（当剩余步数大于n_min时）
        if T_steps - i > n_min:

            V_delta_avg = torch.zeros_like(x_src_packed)  # 速度差平均值

            # 多次计算速度场取平均
            for k in range(n_avg):
                # 生成前向噪声
                fwd_noise = torch.randn_like(x_src_packed).to(device)
                
                # 构造源噪声潜在变量
                zt_src = (1 - t_i) * x_src_packed + t_i * fwd_noise
                # 构造目标噪声潜在变量
                zt_tar = zt_edit + zt_src - x_src_packed       #(zt_edit - t_i*x_src_packed)+ t_i * fwd_noise


                Vt_src = calc_v_flux(
                    pipe,
                    latents=zt_src,
                    prompt_embeds=src_prompt_embeds,
                    pooled_prompt_embeds=src_pooled_prompt_embeds,
                    guidance=src_guidance,
                    text_ids=src_text_ids,
                    latent_image_ids=latent_src_image_ids,
                    t=t
                )
                
                # 计算目标提示的速度场
                Vt_tar = calc_v_flux(
                    pipe,
                    latents=zt_tar,
                    prompt_embeds=tar_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance=tar_guidance,
                    text_ids=tar_text_ids,
                    latent_image_ids=latent_tar_image_ids,
                    t=t
                )

                # 累加速度差
                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)

                # 添加指标引导
                guide_freq = 2  # 指标引导频率
                if i % guide_freq == 0:  # 每3步应用一次指标引导
                    metric_guide = MetricGuidance(pipe, device='cuda')
                    guide_grad, _ = metric_guide.compute_guidance_grad(
                        zt_edit, init_image_pil, tar_prompt,orig_height, orig_width
                        )
                    
                    # 混合流编辑和指标梯度 (可配置混合权重)
                    V_delta_avg += 0.2 * guide_grad / (guide_grad.norm() + 1e-6)
                    
                    del metric_guide
                    del guide_grad
                    torch.cuda.empty_cache()

            # 更新ODE状态（使用欧拉方法）
            zt_edit = zt_edit.to(torch.float32) + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        # 常规采样阶段（最后n_min步）
        else:
            # 初始化采样阶段的噪声潜在变量
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src_packed).to(device)
                xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed
                
            # 计算目标速度场
            Vt_tar = calc_v_flux(
                pipe,
                latents=xt_tar,
                prompt_embeds=tar_prompt_embeds,
                pooled_prompt_embeds=tar_pooled_prompt_embeds,
                guidance=tar_guidance,
                text_ids=tar_text_ids,
                latent_image_ids=latent_tar_image_ids,
                t=t
            )

            # 更新采样状态
            prev_sample = xt_tar.to(torch.float32) + (t_im1 - t_i) * Vt_tar
            xt_tar = prev_sample.to(Vt_tar.dtype)

    # 选择最终输出（根据是否进入常规采样阶段）
    out = zt_edit if n_min == 0 else xt_tar
    # 解包潜在变量为图像格式
    unpacked_out = pipe._unpack_latents(out, orig_height, orig_width, pipe.vae_scale_factor)
    return unpacked_out

