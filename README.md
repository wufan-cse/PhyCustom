<div align="center">

<h1>PhyDiff: Towards Realistic Physical Transformations in Text-to-Image Diffusion Models</h1>


[Fan Wu](https://wufan-cse.github.io/)<sup>1</sup>, Cheng Chen<sup>1</sup>, Zhoujie Fu<sup>1</sup>, Xulei Yang<sup>2</sup>, Yi Xu<sup>3</sup>, [Guosheng Lin](https://guosheng.github.io/)<sup>1</sup><sup>&#8224;</sup>

<sup>1</sup>Nanyang Technological University,
<sup>2</sup>A*STAR,
<sup>3</sup>OPPO US Research Center,
<sup>&#8224;</sup>Corresponding author

[//]: # (CVPR 2024)

<a href="">
<img src='https://img.shields.io/badge/arxiv-PhyDiff-blue' alt='Paper PDF'></a>
<a href="https://github.com/wufan-cse/PhyDiff">
<img src='https://img.shields.io/badge/Project-Website-orange' alt='Project Page'></a>

</div>

## 📖 Abstract
<div style="text-align: center;">
    <img src="docs/figures/intro_results.png" alt="results_of_phydiff" width="1200">
</div>
<p>
    In this paper, we address a realistic task, physical transformations image generation, where we aim to freely combine physical concepts on open-world objects to generate natural and meaningful images. 
    Text-to-image diffusion models have demonstrated their superior performance in generating realistic images and recent diffusion-based customization methods, which aim at controlling the generation processes of diffusion models, have achieved significant success in customizing the shapes or styles of objects. 
    However, despite the tremendous success witnessed nowadays, diffusion models lack a fundamental understanding of physical processes, often viewing physical states as fixed attributes of a given object rather than dynamic changes happening on the object itself due to the rare training data, which contain such kinds of abstract knowledge (physical knowledge). 
    Furthermore, diffusion-based customization methods mostly focus on the appearance of the given objects and struggle to generate a combination of physical concepts to open-world objects since the knowledge has yet to be fully learned by diffusion models, failing to perform physical transformations image generation. 
    To address this limitation, we propose PhyDiff, a diffusion-based model fine-tuning framework using a few images and corresponding text prompts as inputs to perform realistic and meaningful physical transformations on open-world objects. 
    PhyDiff comprises two novel regularization loss functions. 
    One is concept decouple loss, which helps to decouple the mixture of independent features from multiple input concept data, ensuring the diffusion model learns the representations, respectively. 
    The other is isometric loss, which helps to extract the invariant features existing in the cross-object physical concept data. 
    Experiments are conducted on a newly constructed dataset, which consists of 25 object concepts and 6 physical concepts, in a total of 150 unique combinations. 
    The results demonstrate that PhyDiff outperforms previous state-of-the-art and popular methods in terms of performing physical transformations on open-world objects quantitatively and qualitatively.
</p>


## 🚀 Run
1. install
```
conda create -n freecustom python=3.10 -y
conda activate freecustom
pip install -r requirements.txt
```

2. run the following command to view the results
```
python freecustom_stable_diffusion.py
```

**At this point, you can already see the customized results, but you can also try the following two methods:**
1. try another config
- replace `./configs/config_stable_diffusion.yaml` with one of configuration files in the `./datasets/freecustom/multi_concept` folder. 
- run as step 2.

2. prepare your own data
- Select 2 to 3 images that represent the concepts you wish to customize, ensuring that each concept has contextual interaction.
- Use [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) or other segmentation tools to obtain concept masks for filtering out irrelevant pixels.
- Store your images and masks according to the structure in the dataset folder, making sure that the filenames and extensions of the images and masks are identical.
- Modify the `./configs/config_stable_diffusion.yaml` file by updating the "ref_image_infos" and "target_prompt" fields to align with your prepared data.
- Execute `python freecustom_stable_diffusion.py` to view the results.
- Feel free to experiment with adjusting the "seeds" and "mask_weights" fields in the `./configs/config_stable_diffusion.yaml` to achieve satisfactory results.

## 🌄 Demo of customized image generation
### multi-concept composition 
![results_of_multi_concept](docs/static/images/results_of_multi_concept.png)

### single-concept customization
![results_of_single_concept](docs/static/images/results_of_single_concept.png)

Our method excels at *rapidly* generating high-quality images with multiple concept combinations and single concept customization, without any model parameter tuning. The identity of each concept is remarkably preserved. Furthermore, our method exhibits great versatility and robustness when dealing with different categories of concepts. This versatility allows users to generate customized images that involve diverse combinations of concepts, catering to their specific needs and preferences. Best viewed on screen.

## 🗓️ TODO
- [x] Release code and datasets
- [x] Release FreeCustom on Stable Diffusion pipeline and running script
- [x] Release FreeCustom on BLIP Diffusion pipeline
- [ ] Release FreeCustom on BLIP Diffusion running script
- [ ] Release FreeCustom on ControlNet pipeline and running script


## 🎫 License
For non-commercial academic use, this project is licensed under [the 2-clause BSD License](https://opensource.org/license/bsd-2-clause). 
For commercial use, please contact: [Chunhua Shen](mailto:chhshen@gmail.com)




## 🖊️ BibTeX
If you find this project useful in your research, please consider cite:

```bibtex
@inproceedings{ding2024freecustom,
  title={FreeCustom: Tuning-Free Customized Image Generation for Multi-Concept Composition}, 
  author={Ganggui Ding and Canyu Zhao and Wen Wang and Zhen Yang and Zide Liu and Hao Chen and Chunhua Shen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## 🙏 Acknowledgements
We thank to [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [MasaCtrl](https://github.com/TencentARC/MasaCtrl), [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything), [HuggingFace](https://huggingface.co), [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt), [ControlNet](https://github.com/lllyasviel/ControlNet)

## 📧 Contact

If you have any technical comments or questions, please open a new issue or feel free to contact [Ganggui Ding](https://dingangui.github.io)