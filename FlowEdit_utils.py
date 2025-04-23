from typing import Optional, Tuple, Union
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import numpy as np

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
                ):     

    # 设备设置和图像尺寸获取
#****************************************************************
    # print(f"FlowEditFLUX: x_src.shape={x_src.shape}, pipe.vae_scale_factor={pipe.vae_scale_factor}")
#****************************************************************
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
#****************************************************************
    # print("\n===== prepare_latents 参数 =====")
    # print(f"batch_size: {x_src.shape[0]}")
    # print(f"num_channels_latents: {num_channels_latents}")
    # print(f"height: {orig_height}")
    # print(f"width: {orig_width}")
    # print(f"dtype: {x_src.dtype}")
    # print(f"device: {device}")
    # print(f"generator: {None}")  # 这里显式传入的是 None
    # print(f"latents: shape={x_src.shape}, dtype={x_src.dtype}, device={x_src.device}")
#****************************************************************
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

    pipe.text_encoder.to('cuda')
    pipe.text_encoder_2.to('cuda')
    # 编码源提示文本
    (
        src_prompt_embeds,        # 源提示的嵌入表示
        src_pooled_prompt_embeds, # 池化后的源提示嵌入
        src_text_ids,             # 源文本的token ID
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,            #默认与prompt一致
        device=device,
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
        device=device,
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


    pipe.vae.to('cpu')
    pipe.text_encoder.to('cpu')
    pipe.text_encoder_2.to('cpu')
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

