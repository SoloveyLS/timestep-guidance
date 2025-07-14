import torch
from torch import Tensor

from typing import (
    List, 
    Dict, 
    Callable, 
    Any, 
    Union, 
    Optional
)
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.utils import (
    replace_example_docstring,
    is_torch_xla_available
)
from diffusers.image_processor import (
    PipelineImageInput
)

XLA_AVAILABLE = False
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusion3Pipeline

        >>> pipe = StableDiffusion3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        >>> image.save("sd3.png")
        ```
"""

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class Guider:
    """
    A helper that captures a 'skip' prediction at certain timesteps
    and then applies guidance (orthogonal or linear) on subsequent steps.
    """
    def __init__(
        self,
        skip_logic: Callable[[int], bool] = lambda ind: not bool(ind % 2),
        skip_guidance: Optional[float]   = None,
        orth: bool = False,
        norm: bool = True,
    ):
        self.ind = 0
        self.skip_logic = skip_logic
        self.skip_guidance = skip_guidance
        self.orth = orth
        self.norm = norm

        # Will hold the last 'skip' prediction
        self.skip_pred: Optional[Tensor] = None

    @torch.no_grad()
    def update(self, noise_pred: Tensor, timestep_ind: int) -> None:
        """
        Call this at each step to potentially record a 'skip' prediction.
        """
        self.ind = timestep_ind
        if self.skip_logic(self.ind) and not self.skip_logic(self.ind - 1):
            self.skip_pred = noise_pred.clone()

    @torch.no_grad()
    def guide(self, noise_pred: Tensor, edit_coef=1) -> Tensor:
        """
        Modify `noise_pred` based on the last recorded skip_pred
        using either orthogonal or linear guidance.
        """
        # Make a copy for later norm-preservation
        original = noise_pred.clone()

        # Only guide when skip_logic is False (i.e. 'non-skip' steps)
        if not self.skip_logic(self.ind) and self.skip_pred is not None:
            g = self.skip_guidance
            if g is None:
                raise ValueError("skip_guidance must be set to a float before guiding")
            
            guidance_addition = (noise_pred - self.skip_pred)

            # Orthogonalize if neccesary:
            if self.orth:
                spf = self.skip_pred.float()
                npf = noise_pred.float()

                dot_sp_np = (spf * npf).sum()
                dot_np_np = (npf * npf).sum()

                guidance_addition = spf - (npf * dot_sp_np / dot_np_np)
                guidance_addition = -guidance_addition.to(noise_pred.dtype)

            # apply guidance:
            noise_pred = noise_pred + (g - 1) * guidance_addition * edit_coef

            # normilize if necessary:
            if self.norm:
                noise_pred = noise_pred / noise_pred.norm() * original.norm()

        return noise_pred
    # def __init__(
    #     self,
    #     skip_logic: Callable[[int], bool] = lambda ind: not bool(ind % 2),
    #     skip_guidance: Optional[float]   = None,
    #     orth: bool = False,
    #     norm: bool = True,
    # ):
    #     self.ind = 0
    #     self.skip_logic = skip_logic
    #     self.skip_guidance = skip_guidance
    #     self.orth = orth
    #     self.norm = norm

    #     # Will hold the last 'skip' prediction, shape (B,C,H,W) or (C,H,W)
    #     self.skip_pred: Optional[Tensor] = None

    # @torch.no_grad()
    # def update(self, noise_pred: Tensor, timestep_ind: int) -> None:
    #     """
    #     Call this at each step to potentially record a 'skip' prediction.
    #     Stores a detached clone of noise_pred whenever
    #     skip_logic(t) is True AND skip_logic(t-1) was False.
    #     """
    #     self.ind = timestep_ind
    #     if self.skip_logic(self.ind) and not self.skip_logic(self.ind - 1):
    #         # detach + clone so we keep this tensor around without gradients
    #         self.skip_pred = noise_pred.detach().clone()

    # @torch.no_grad()
    # def guide(self, noise_pred: Tensor) -> Tensor:
    #     """
    #     Modify `noise_pred` based on the last recorded skip_pred
    #     using either orthogonal or linear guidance.
    #     """
    #     # nothing to do if we haven't recorded skip_pred yet
    #     if self.skip_pred is None or self.skip_logic(self.ind):
    #         return noise_pred

    #     if self.skip_guidance is None:
    #         raise ValueError("skip_guidance must be set to a float before guiding")

    #     # Ensure shapes match
    #     if noise_pred.shape != self.skip_pred.shape:
    #         raise RuntimeError(f"Shape mismatch: noise_pred is {noise_pred.shape}, "
    #                            f"but skip_pred is {self.skip_pred.shape}")

    #     # Keep original for possible norm‐preservation
    #     original = noise_pred.clone()

    #     # Compute the raw “direction” we want to guide along
    #     # (B,C,H,W) - (B,C,H,W)
    #     guidance_addition = noise_pred - self.skip_pred

    #     if self.orth:
    #         # convert to float for stable dot products
    #         spf = self.skip_pred.float()
    #         npf = noise_pred.float()

    #         # dot products per‐batch: sum over C,H,W, keepdims -> (B,1,1,1)
    #         dot_sp_np = (spf * npf).sum(dim=[*range(1, npf.ndim)], keepdim=True)
    #         dot_np_np = (npf * npf).sum(dim=[*range(1, npf.ndim)], keepdim=True)

    #         # project spf onto subspace orthogonal to npf
    #         #   spf - (npf * (spf·npf)/(npf·npf))
    #         guidance_addition = spf - npf * (dot_sp_np / dot_np_np)
    #         # restore original dtype and flip sign
    #         guidance_addition = -guidance_addition.to(noise_pred.dtype)

    #     # linear combination: new = old + (g-1)*direction
    #     g = self.skip_guidance
    #     noise_pred = noise_pred + (g - 1) * guidance_addition

    #     if self.norm:
    #         # compute norms per‐batch over C,H,W -> shape (B,1,1,1)
    #         orig_norm = original.pow(2).sum(dim=[*range(1, original.ndim)],
    #                                      keepdim=True).sqrt()
    #         new_norm  = noise_pred.pow(2).sum(dim=[*range(1, noise_pred.ndim)],
    #                                       keepdim=True).sqrt()
    #         # rescale so ||new|| == ||orig||
    #         noise_pred = noise_pred * (orig_norm / new_norm).clamp(min=0.0)

    #     return noise_pred

@torch.compile()
@torch.no_grad()
def skip_guidance_call(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    skip_guidance: float = 0.0,
    skip_steps_count: int = 0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_guidance_layers: List[int] = None,
    skip_layer_guidance_scale: float = 2.8,
    orthogonalize: bool = False,
    normilize: bool = False,
    mu: Optional[float] = None,
):
    
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            will be used instead
        prompt_3 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
            will be used instead
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image. This is set to 1024 by default for the best results.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image. This is set to 1024 by default for the best results.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        sigmas (`List[float]`, *optional*):
            Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
            their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
            will be used.
        guidance_scale (`float`, *optional*, defaults to 7.0):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        negative_prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
            `text_encoder_2`. If not defined, `negative_prompt` is used instead
        negative_prompt_3 (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
            `text_encoder_3`. If not defined, `negative_prompt` is used instead
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
            input argument.
        ip_adapter_image (`PipelineImageInput`, *optional*):
            Optional image input to work with IP Adapters.
        ip_adapter_image_embeds (`torch.Tensor`, *optional*):
            Pre-generated image embeddings for IP-Adapter. Should be a tensor of shape `(batch_size, num_images,
            emb_dim)`. It should contain the negative image embedding if `do_classifier_free_guidance` is set to
            `True`. If not provided, embeddings are computed from the `ip_adapter_image` input argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] instead of
            a plain tuple.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        callback_on_step_end (`Callable`, *optional*):
            A function that calls at the end of each denoising steps during the inference. The function is called
            with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
            callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.
        skip_guidance_layers (`List[int]`, *optional*):
            A list of integers that specify layers to skip during guidance. If not provided, all layers will be
            used for guidance. If provided, the guidance will only be applied to the layers specified in the list.
            Recommended value by StabiltyAI for Stable Diffusion 3.5 Medium is [7, 8, 9].
        skip_layer_guidance_scale (`int`, *optional*): The scale of the guidance for the layers specified in
            `skip_guidance_layers`. The guidance will be applied to the layers specified in `skip_guidance_layers`
            with a scale of `skip_layer_guidance_scale`. The guidance will be applied to the rest of the layers
            with a scale of `1`.
        skip_layer_guidance_stop (`int`, *optional*): The step at which the guidance for the layers specified in
            `skip_guidance_layers` will stop. The guidance will be applied to the layers specified in
            `skip_guidance_layers` until the fraction specified in `skip_layer_guidance_stop`. Recommended value by
            StabiltyAI for Stable Diffusion 3.5 Medium is 0.2.
        skip_layer_guidance_start (`int`, *optional*): The step at which the guidance for the layers specified in
            `skip_guidance_layers` will start. The guidance will be applied to the layers specified in
            `skip_guidance_layers` from the fraction specified in `skip_layer_guidance_start`. Recommended value by
            StabiltyAI for Stable Diffusion 3.5 Medium is 0.01.
        mu (`float`, *optional*): `mu` value used for `dynamic_shifting`.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
        `tuple`. When returning a tuple, the first element is a list with the generated images.
    """

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._skip_guidance = skip_guidance
    self._skip_steps_count = skip_steps_count
    self._skip_layer_guidance_scale = skip_layer_guidance_scale
    self._orthogonalize = orthogonalize
    self._normilize     = normilize
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    if self._skip_guidance != 1 and self._skip_steps_count > 0:
        self.do_skip_guidance = True
        skip_next_step = lambda ind: not bool(ind % self._skip_steps_count)
    else:
        self.do_skip_guidance = False

    if self.do_classifier_free_guidance:
        if skip_guidance_layers is not None:
            original_prompt_embeds = prompt_embeds
            original_pooled_prompt_embeds = pooled_prompt_embeds
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        _, _, height, width = latents.shape
        image_seq_len = (height // self.transformer.config.patch_size) * (
            width // self.transformer.config.patch_size
        )
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.16),
        )
        scheduler_kwargs["mu"] = mu
    elif mu is not None:
        scheduler_kwargs["mu"] = mu
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # 6. Prepare image embeddings
    if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
        ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            self.do_classifier_free_guidance,
        )

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
        else:
            self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

    # 7. Denoising loop
    if self.do_skip_guidance:
        _guide = Guider(
            skip_logic=skip_next_step, 
            skip_guidance=self._skip_guidance,
            orth=self._orthogonalize, 
            norm=self._normilize
        )
        _guide_uncond = Guider(
            skip_logic=skip_next_step, 
            skip_guidance=self._skip_guidance,
            orth=self._orthogonalize, 
            norm=self._normilize
        )
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]


            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                if self.do_skip_guidance:
                    # _guide.update(
                    #     noise_pred=noise_pred_text,
                    #     timestep_ind=i,
                    # )
                    # noise_pred_text = _guide.guide(
                    #     noise_pred=noise_pred_text,
                    #     edit_coef = self.guidance_scale ** -1
                    # )

                    _guide_uncond.update(
                        noise_pred=noise_pred_uncond,
                        timestep_ind=i,
                    )
                    noise_pred_uncond = _guide_uncond.guide(
                        noise_pred=noise_pred_uncond,
                        edit_coef = self.guidance_scale ** -1
                    )
                
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # if self.do_skip_guidance:
            #     _guide.update(
            #         noise_pred=noise_pred,
            #         timestep_ind=i,
            #     )
            #     noise_pred = _guide.guide(
            #         noise_pred=noise_pred
            #     )
            

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                negative_pooled_prompt_embeds = callback_outputs.pop(
                    "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                )

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                raise ValueError("XLA is not implemented for custom pipeline!")

    if output_type == "latent":
        image = latents

    else:
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return StableDiffusion3PipelineOutput(images=image)
    