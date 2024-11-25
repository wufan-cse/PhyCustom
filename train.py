import os

import kiui
from lightning import seed_everything
from lightning.app.components.serve.serve import instance
from markdown_it.rules_block import reference
from omegaconf import OmegaConf
from torch.fx.experimental.proxy_tensor import track_tensor

from models.data.dreambooth_dataset import DreamBoothDataset
from utils.model_utils import import_model_class_from_model_name_or_path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
import copy
import gc
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from peft import get_peft_model_state_dict

from models.multi_booth.multi_booth_unet import MultiViewUNetModel
from models.multi_booth.multi_booth_pipeline import MVDreamPipeline

from models.reference_attention.hack_attention import hack_self_attention_to_mrsa
from models.reference_attention.mrsa import MultiReferenceSelfAttention

from utils.data_utils import load_image, collate_fn, tokenize_prompt, encode_prompt


logger = get_logger(__name__)


def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def create_model_with_accelerator(args):
    # Preparation
    logging_dir = args["output_dir"].replace("ckpts", "logs")

    accelerator_project_config = ProjectConfiguration(project_dir=args["output_dir"], logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args["gradient_accumulation_steps"],
        mixed_precision=None,
        log_with=args["report_to"],
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args["seed"] is not None:
        seed_everything(args["seed"])

    # Handle the repository creation
    if accelerator.is_main_process:
        if args["output_dir"] is not None:
            os.makedirs(args["output_dir"], exist_ok=True)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Model
    if args["tokenizer_name"]:
        tokenizer = AutoTokenizer.from_pretrained(args["tokenizer_name"], revision=args["revision"], use_fast=False)
    elif args["pretrained_model_name_or_path"]:
        tokenizer = AutoTokenizer.from_pretrained(
            args["pretrained_model_name_or_path"],
            subfolder="tokenizer",
            revision=args["revision"],
            use_fast=False,
        )
    else:
        tokenizer = None

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args["pretrained_model_name_or_path"],
                                                                  args["revision"])

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args["pretrained_model_name_or_path"], subfolder="scheduler")

    text_encoder_object = text_encoder_cls.from_pretrained(
        args["pretrained_model_name_or_path"], subfolder="text_encoder", revision=args["revision"],
        variant=args["variant"]
    )

    text_encoder_verb = text_encoder_cls.from_pretrained(
        args["pretrained_model_name_or_path"], subfolder="text_encoder", revision=args["revision"],
        variant=args["variant"]
    )

    try:
        vae = AutoencoderKL.from_pretrained(
            args["pretrained_model_name_or_path"], subfolder="vae", revision=args["revision"], variant=args["variant"]
        )
    except OSError:
        vae = None

    # MVDream
    # unet = MultiViewUNetModel.from_pretrained(
    #     args["pretrained_model_name_or_path"], subfolder="unet", revision=args["revision"], variant=args["variant"]
    # )
    unet = UNet2DConditionModel.from_pretrained(
        args["pretrained_model_name_or_path"], subfolder="unet", revision=args["revision"], variant=args["variant"]
    )

    # We only train the additional adapter LoRA layers
    if vae is not None:
        vae.requires_grad_(False)
    text_encoder_object.requires_grad_(False)
    text_encoder_verb.requires_grad_(False)
    unet.requires_grad_(False)

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_object.to(accelerator.device, dtype=weight_dtype)
    text_encoder_verb.to(accelerator.device, dtype=weight_dtype)

    # now we will add new LoRA weights to the attention layers
    unet_lora_config = LoraConfig(
        r=args["rank"],
        lora_alpha=args["rank"],
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
    )


    # Two LoRA modules
    # diffusers.loaders.peft._SET_ADAPTER_SCALE_FN_MAPPING['MultiViewUNetModel'] = lambda model_cls, weights: weights

    unet.add_adapter(unet_lora_config, adapter_name="obj1")
    unet.add_adapter(unet_lora_config, adapter_name='obj2')
    unet.set_adapters(["obj1", "obj2"], weights=args["multi_lora_weights"])


    # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
    text_lora_config = LoraConfig(
        r=args["rank"],
        lora_alpha=args["rank"],
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    text_encoder_verb.add_adapter(text_lora_config, adapter_name='text_lora')

    ###################################### Register save and load hook ######################################

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(accelerator, unet))):
                    unet_lora_layers_to_save_1 = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model, adapter_name="obj1"))
                    unet_lora_layers_to_save_2 = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model, adapter_name="obj2"))

                elif isinstance(model, type(unwrap_model(accelerator, text_encoder_verb))):
                    text_encoder_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model, adapter_name="text_lora")
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            StableDiffusionLoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save_1,
                text_encoder_lora_layers=text_encoder_lora_layers_to_save,
                weight_name="pytorch_lora_weights_1.safetensors"
            )
            StableDiffusionLoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save_2,
                text_encoder_lora_layers=text_encoder_lora_layers_to_save,
                weight_name="pytorch_lora_weights_2.safetensors"
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(accelerator, unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(accelerator, text_encoder_verb))):
                text_encoder_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)

        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args["train_text_encoder"]:
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_)

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [unet_]
            if args.train_text_encoder:
                models.append(text_encoder_)

            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    ###################################### Register save and load hook ######################################

    return text_encoder_object, text_encoder_verb, tokenizer, vae, unet, noise_scheduler, weight_dtype, accelerator

def create_unet_optimizer(args, unet, text_encoder=None):
    # Optimizer creation
    optimizer_class = torch.optim.AdamW

    # params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters()))
    params_to_optimize_unet_lora_1 = list(p for n, p in unet.named_parameters() if p.requires_grad is True and 'obj1' in n)
    params_to_optimize_unet_lora_2 = list(p for n, p in unet.named_parameters() if p.requires_grad is True and 'obj2' in n)

    params_to_optimize = list(p for n, p in unet.named_parameters() if p.requires_grad is True)

    if text_encoder is not None:
        params_to_optimize = params_to_optimize + list(filter(lambda p: p.requires_grad, text_encoder.parameters()))

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args["learning_rate"],
        betas=(args["adam_beta1"], args["adam_beta2"]),
        weight_decay=args["adam_weight_decay"],
        eps=args["adam_epsilon"],
    )

    return params_to_optimize, params_to_optimize_unet_lora_1, params_to_optimize_unet_lora_2, optimizer

def create_dreambooth_dataset(args, tokenizer):
    obj = args["concepts"][0]
    verb = args["concepts"][1]

    instance_data_dir = [os.path.join(args["data_dir"], "objects", obj), os.path.join(args["data_dir"], "verbs", verb)]


    reference_prompt = ""
    db_datasets = []
    db_dataloaders = []
    for i in range(len(instance_data_dir)):
        prompt_path = os.path.join(instance_data_dir[i], "prompt.yaml")
        prompt = OmegaConf.load(prompt_path)
        prompt = OmegaConf.to_object(prompt)

        if prompt["reference_prompt"] is not None:
            reference_prompt = prompt["reference_prompt"]

        dataset = DreamBoothDataset(
            instance_data_root=instance_data_dir[i],
            instance_prompt=prompt["instance_prompt"],
            class_data_root=None,
            class_prompt=None,
            class_num=args["num_class_images"],
            tokenizer=tokenizer,
            size=args["image_resolution"],
            center_crop=args["center_crop"],
            encoder_hidden_states=None,
            class_prompt_encoder_hidden_states=None,
            tokenizer_max_length=args["tokenizer_max_length"],
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args["train_batch_size"],
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, args["with_prior_preservation"]),
            num_workers=args["dataloader_num_workers"],
        )
        db_datasets.append(dataset)
        db_dataloaders.append(dataloader)

    return db_datasets, db_dataloaders, reference_prompt

def main(args):
    args["output_dir"] = os.path.join(args["output_dir"], args["concepts"][0]+"_"+args["concepts"][1], "ckpts")

    # Model
    text_encoder_object, text_encoder_verb, tokenizer, vae, unet, noise_scheduler, weight_dtype, accelerator = create_model_with_accelerator(args)

    # Opimizer
    params_to_optimize, params_to_optimize_1, params_to_optimize_2, optimizer = create_unet_optimizer(args, unet, text_encoder_verb)

    # Scheduler
    lr_scheduler = get_scheduler(
        args["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=args["lr_warmup_steps"] * accelerator.num_processes,
        num_training_steps=args["max_train_steps"] * accelerator.num_processes,
        num_cycles=args["lr_num_cycles"],
        power=args["lr_power"]
    )

    # Training data
    train_dataset, train_dataloader, reference_prompt = create_dreambooth_dataset(args, tokenizer)

    # Prepare everything
    unet, text_encoder_verb, optimizer, lr_scheduler = accelerator.prepare(
        unet, text_encoder_verb, optimizer, lr_scheduler
    )
    train_dataloader = [accelerator.prepare(dl) for dl in train_dataloader]

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(min(len(train_dataloader[0]), len(train_dataloader[1])) / args["gradient_accumulation_steps"])
    args["num_train_epochs"] = math.ceil(args["max_train_steps"] / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # tracker_config = vars(copy.deepcopy(args))
        # tracker_config = copy.deepcopy(args)
        # tracker_config.pop("validation_images")
        accelerator.init_trackers("dreambooth-lora", config=None)



    # Train!
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num docs = {len(train_dataset)}")
    # logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    # logger.info(f"  Num Epochs = {args["num_train_epochs"]}")
    # logger.info(f"  Instantaneous batch size per device = {args["train_batch_size"]}")
    # logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    # logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    # logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args["max_train_steps"]),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args["num_train_epochs"]):
        unet.train()
        # if args["train_text_encoder"]:
        #     text_encoder.train()

        for step, batches in enumerate(zip(*train_dataloader)):
            with accelerator.accumulate(unet):
                var_losses, orth_losses, obj1_losses, obj2_losses, losses = 0., 0., 0., 0., 0.
                for b in range(len(batches)):
                    batch = batches[b]
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                    if vae is not None:
                        # Convert images to latent space
                        # model_input -> (1, 4, 64, 64)
                        model_input = vae.encode(pixel_values).latent_dist.sample()
                        model_input = model_input * vae.config.scaling_factor
                    else:
                        model_input = pixel_values

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)

                    bsz, channels, height, width = model_input.shape
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                    timesteps = timesteps.long()

                    if b == 0:
                        encoder_hidden_states = encode_prompt(
                            text_encoder_verb,
                            batch["input_ids"],
                            batch["attention_mask"],
                            text_encoder_use_attention_mask=args["text_encoder_use_attention_mask"],
                        )
                    else:
                        encoder_hidden_states = encode_prompt(
                            text_encoder_verb,
                            batch["input_ids"],
                            batch["attention_mask"],
                            text_encoder_use_attention_mask=args["text_encoder_use_attention_mask"],
                        )

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)

                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)


                    # unet_in_channels = 4
                    if unwrap_model(accelerator, unet).config.in_channels == channels * 2:
                        noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                    # Predict the noise residual
                    unet_inputs = {
                        'sample': noisy_model_input,
                        'timestep': timesteps,
                        'encoder_hidden_states': encoder_hidden_states,
                    }
                    # predict the noise residual
                    model_pred = unet.forward(**unet_inputs)
                    model_pred = model_pred[0]


                    # if model predicts variance, throw away the prediction. we will only train on the
                    # simplified training objective. This means that all schedulers using the fine tuned
                    # model must be configured to use one of the fixed variance variance types.
                    if model_pred.shape[1] == 6:
                        model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if args["with_prior_preservation"]:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # upweight the object
                    if b == 0:
                        obj1_losses += args["object_weight"] * loss
                    else:
                        obj2_losses += args["verb_weight"] * loss

                    count = 0
                    orth_loss = []
                    for v1, v2 in zip(params_to_optimize_1, params_to_optimize_2):
                        if count % args["orth_frequency"] == 0:
                            orth_loss.append(torch.mean(torch.abs(torch.matmul(v1.T, v2))))
                        count += 1
                    orth_loss = torch.sum(torch.stack(orth_loss))

                    orth_losses += args["orth_weight"] * orth_loss

                    if b == 1:
                        ref_inputs = tokenize_prompt(
                            train_dataset[1].tokenizer, reference_prompt, tokenizer_max_length=train_dataset[1].tokenizer_max_length
                        )
                        ref_encoder_hidden_states = encode_prompt(
                            text_encoder_verb,
                            ref_inputs.input_ids,
                            ref_inputs.attention_mask,
                            text_encoder_use_attention_mask=args["text_encoder_use_attention_mask"],
                        )

                        ref_encoder_hidden_states = ref_encoder_hidden_states.repeat(bsz, 1, 1)
                        diff = torch.abs(encoder_hidden_states - ref_encoder_hidden_states)
                        diff_var = torch.var(diff, dim=0, unbiased=False)
                        diff_var_loss = torch.mean(diff_var)

                        var_losses += args["var_weight"] * diff_var_loss

                losses = var_losses + orth_losses + obj1_losses + obj2_losses

                accelerator.backward(losses)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args["max_grad_norm"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args["checkpoint_steps"] == 0:
                        save_path = os.path.join(args["output_dir"], f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")


            logs = {"loss": losses.detach().item(),
                    "loss_object": obj1_losses.detach().item(),
                    "loss_verb": obj2_losses.detach().item(),
                    "loss_orth": orth_losses.detach().item(),
                    "loss_diff": var_losses.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args["max_train_steps"]:
                break


    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(accelerator, unet)
        unet = unet.to(torch.float32)

        unet_lora_layers_to_save_1 = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unet, adapter_name="obj1"))
        unet_lora_layers_to_save_2 = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unet, adapter_name="obj2"))

        # if args["train_text_encoder"]:
        text_encoder_verb = unwrap_model(accelerator, text_encoder_verb)
        text_encoder_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder_verb, adapter_name="text_lora"))

        StableDiffusionLoraLoaderMixin.save_lora_weights(
            save_directory=args["output_dir"],
            unet_lora_layers=unet_lora_layers_to_save_1,
            text_encoder_lora_layers=text_encoder_state_dict,
            weight_name="pytorch_lora_weights_1.safetensors"
        )
        StableDiffusionLoraLoaderMixin.save_lora_weights(
            save_directory=args["output_dir"],
            unet_lora_layers=unet_lora_layers_to_save_2,
            text_encoder_lora_layers=text_encoder_state_dict,
            weight_name="pytorch_lora_weights_2.safetensors"
        )

    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument(
        "--obj",
        type=str,
        default=None
    )
    parser.add_argument(
        "--verb",
        type=str,
        default=None
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config_path = "configs/train_v1.yaml"
    args = OmegaConf.load(config_path)
    args = OmegaConf.to_object(args)

    # Benchmarking
    benchmark_args = parse_args()
    if benchmark_args.obj is not None:
        print(f"Doing training for {benchmark_args.obj} and {benchmark_args.verb}")
        args["concepts"] = [benchmark_args.obj, benchmark_args.verb]

    main(args)
