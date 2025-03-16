<p align="center">
  <h1 align="center">
  Progressive Detail Injection for Training-Free Semantic Binding in Text-to-Image Generation<br>
  </h1>
  <p align="center">
    <strong>Lifeng Chen</strong><sup>1</sup>
    &nbsp;&nbsp;
    <strong>Jiner Wang</strong><sup>2</sup>
    &nbsp;&nbsp;
    <a href="https://pan-zihao.github.io/"><strong>Zihao Pan</strong></a><sup>1</sup>
    &nbsp;&nbsp;
    <a href="https://beierzhu.github.io/"><strong>Beier Zhu</strong></a><sup>1, 2</sup>
    &nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=99ZjBGoAAAAJ&hl=en"><strong>Xiaofeng Yang</strong></a><sup>1, 2</sup>
    &nbsp;&nbsp;
    <a href="https://icoz69.github.io/"><strong>Chi Zhang</strong></a><sup>1✉</sup>
    <br>
    <br>
    <sup>1</sup>AGI Lab, Westlake University,</span>&nbsp;
    <sup>2</sup>Nanyang Technological University</span>&nbsp;
    <br>
    <br>
    <a href='https://arxiv.org/abs/2412.08503'><img src='https://img.shields.io/badge/ArXiv-2412.08503-red'></a>&nbsp;
    <a href='https://stylestudio-official.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>&nbsp;
    <a href="https://huggingface.co/spaces/Westlake-AGI-Lab/StyleStudio"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-HF-orange"></a>
    <br>
    <img src="assets/teaser.jpg" width="800">
  </p>
</p>

## 📑 Introduction
This paper provides a new method for resolving the problem of **semantic binding** in text-to-image (T2I) generation, which use a progressive injection way to make the attributes in a correct subject region. It is worth noting that it cannot only resolve the problem of semantic overflow, but also **style blending**, which cannot be handled by existing methods. The main idea of our method is to remove all attributes firstly, then add them one by one, in a unified self-attention framework.


<img src="assets/method.jpg" width="800">

For technical details, please refer to our paper.

## 🚀 Usage

1. **Environment Setup**

   **Create and activate the Conda virtual environment:**

   ```bash
   conda env create -f environment.yaml
   conda activate tome
   ```
   Alternatively, install dependencies via `pip`:
    ```bash
    pip install -r requirements.txt
    ```

   Additionally, download the SpaCy model for syntax parsing:

   ```bash
   python -m spacy download en_core_web_trf
   ```

2. **Configure Parameters**

   Modify the `configs/demo_config.py` file to adjust runtime parameters as needed. This file includes two example configuration classes: `RunConfig1` for object binding and `RunConfig2` for attribute binding. Key parameters are as follows:

   - `prompt`: Text prompt for guiding image generation.
   - `model_path`: Path to the Stable Diffusion model; set to `None` to download the pretrained model automatically.
   - `use_nlp`: Whether to use an NLP model for token parsing.
   - `token_indices`: Indices of tokens to merge.
   - `prompt_anchor`: Split text prompt.
   - `prompt_merged`: Text prompt after token merging.
   - For further parameter details, please refer to the comments in the configuration file and our paper.

3. **Run the Example**

   Execute the main script `run_demo.py`:

   ```bash
   python run_demo.py
   ```

   The generated images will be saved in the `demo` directory.

## 📸 Example Outputs

If everything is set up correctly, `RunConfig1` and `RunConfig2` should produce the left and right images below, respectively:

<img src="pics\demo.png" width="1000">

## ⚠️ Notes

- **Custom Configurations**: To use custom text prompts and parameters, add a new configuration class in `configs/demo_config.py` and make necessary adjustments in `run_demo.py`.
- **Parameter Sensitivity**: This method inherits the sensitivity of inference-based optimization techniques, meaning that the generated results are highly dependent on hyperparameter settings. Careful tuning may be required to achieve optimal results.
- **NLP Models**: When using NLP models like SpaCy for token parsing, ensure the correct language model is installed.

## 🙏 Acknowledgments

This project builds upon valuable work and resources from the following repositories:

- [Attend-and-Excite](https://github.com/yuval-alaluf/Attend-and-Excite) 
- [Linguistic Binding in Diffusion Models](https://github.com/RoyiRa/Linguistic-Binding-in-Diffusion-Models)
- [🤗 Diffusers](https://github.com/huggingface/diffusers) 

We extend our sincere thanks to the creators of these projects for their contributions to the field and for making their code available. 🙌