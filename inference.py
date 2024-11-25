import argparse
import os
import random
import torch
import kiui

import numpy as np

import diffusers
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

from pytorch_lightning import seed_everything

from omegaconf import OmegaConf



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
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
    config_path = "configs/inference.yaml"
    args = OmegaConf.load(config_path)
    args = OmegaConf.to_object(args)

    # Benchmarking
    benchmark_args = parse_args()
    if benchmark_args.obj is not None:
        print(f"Doing inference for {benchmark_args.obj} and {benchmark_args.verb}")
        args["concepts"] = [benchmark_args.obj, benchmark_args.verb]

        prompt_path = os.path.join("datasets/gpt_prompt", benchmark_args.verb+".yaml")
        prompt_list = OmegaConf.load(prompt_path)
        prompt_list = OmegaConf.to_object(prompt_list)

        for i in prompt_list['prompts']:
            if i['object'] == benchmark_args.obj:
                args["target_prompt"] = i['prompt']
        print(args["target_prompt"])


        # verb_ing = verb_dict[benchmark_args.verb]
        # args["target_prompt"] = f"A photo of aka {benchmark_args.obj}, sks {verb_ing}"
        # print(f"A photo of aka {benchmark_args.obj}, sks {verb_ing}")

    args["output_dir"] = os.path.join(args["output_dir"], args["concepts"][0]+"_"+args["concepts"][1], "inference")
    args["lora_weight_path_1"] = os.path.join(args["lora_dir"], args["concepts"][0] + "_" + args["concepts"][1], "ckpts",
                                              "checkpoint-"+str(args["weight_step"]), "pytorch_lora_weights_1.safetensors")
    args["lora_weight_path_2"] = os.path.join(args["lora_dir"], args["concepts"][0] + "_" + args["concepts"][1], "ckpts",
                                              "checkpoint-"+str(args["weight_step"]), "pytorch_lora_weights_2.safetensors")
    args["lora_weight_dir"] = os.path.join(args["lora_dir"], args["concepts"][0] + "_" + args["concepts"][1], "ckpts", "checkpoint-"+str(args["weight_step"]))

    os.makedirs(args["output_dir"], exist_ok=True)

    # Register MultiViewUNetModel
    diffusers.loaders.peft._SET_ADAPTER_SCALE_FN_MAPPING['MultiViewUNetModel'] = lambda model_cls, weights: weights

    pipeline = StableDiffusionPipeline.from_pretrained(args["pretrained_model_name_or_path"], torch_dtype=torch.float).to("cuda")
    unet = UNet2DConditionModel.from_pretrained(args["pretrained_model_name_or_path"], subfolder="unet", torch_dtype=torch.float).to("cuda")

    unet.load_attn_procs(args["lora_weight_path_1"], weight_name="pytorch_lora_weights_1.safetensors", adapter_name="obj1")
    unet.load_attn_procs(args["lora_weight_path_2"], weight_name="pytorch_lora_weights_2.safetensors", adapter_name="obj2")
    unet.set_adapters(["obj1", "obj2"], weights=args["multi_lora_weights"])


    pipeline.load_lora_weights(args["lora_weight_dir"], weight_name="pytorch_lora_weights_1.safetensors")

    # TODO(Bug, Fixed): After pipeline.load_lora_weights
    pipeline.unet = unet


    # Data
    target_prompt = args["target_prompt"]
    # Seed
    if args["is_random_seed"]:
        seeds = [random.randint(1, 9999) for _ in range(args["seed_num"])]
        seeds += [0]
    else:
        if type(args["seed"]) is list:
            seeds = args["seed"]
        else:
            seeds = [args["seed"]]


    # Guidance scale
    if args["is_random_gs"]:
        guidance_scales = [random.uniform(1.0, 7.5) for _ in range(args["guidance_scale_num"])]
    else:
        if type(args["guidance_scale"]) is list:
            guidance_scales = args["guidance_scale"]
        else:
            guidance_scales = [args["guidance_scale"]]


    for s in seeds:
        for gs in guidance_scales:

            print("Using seeds {}, guidance scale {}, lora weights {}".format(s, gs, args["multi_lora_weights"][0]))
            seed_everything(s)

            image = pipeline(prompt=target_prompt,
                             guidance_scale=gs,
                             num_inference_steps=args["num_inference_steps"])

            grid = image.images[0]

            if args["is_save_for_reconstruct"]:
                for i in range(len(image)):
                    kiui.write_image(os.path.join(args["output_dir"],
                                                  f'seed_{s}_gs_{gs}_lora_weights_{args["multi_lora_weights"][0]}/{i}.jpg'),
                                     np.array(image[i] * 255).astype(np.uint8))
            else:
                kiui.write_image(os.path.join(args["output_dir"], f'seed_{s}_gs_{gs}_lora_weights_{args["multi_lora_weights"][0]}.jpg'), grid)

