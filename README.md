<p align="center">
  <h1 align="center">SSD-LLM <sub><sup>[ECCV 2024]</sup></sub></h1>

  <a href="https://arxiv.org/abs/2405.02363">
    <img src="https://img.shields.io/badge/arXiv-2405.02363-b31b1b.svg?style=flat&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://link.springer.com/chapter/10.1007/978-3-031-73414-4_14">
    <img src="https://img.shields.io/badge/ECCV-2024-blue.svg?style=flat&logo=spring&logoColor=white" alt="ECCV 2024">
  </a>
  <a href="https://www.youtube.com/watch?v=pw-ZPzlXtQA">
    <img src="https://img.shields.io/badge/Video-Demo-red.svg?style=flat&logo=youtube&logoColor=white" alt="Video Demo">
  </a>
  <a href="https://llm-as-dataset-analyst.github.io/">
    <img src="https://img.shields.io/badge/Project-Website-brightgreen.svg?style=flat&logo=githubpages&logoColor=white" alt="Project Website">
  </a>
</p>

---

## 🔗 Featured Resources

- 📄 **Paper (arXiv)**: [LLM as Dataset Analyst](https://arxiv.org/abs/2405.02363)  
- 📘 **ECCV 2024 Camera Ready**: [Springer Link](https://link.springer.com/chapter/10.1007/978-3-031-73414-4_14)  
- 🌐 **Project Website**: [llm-as-dataset-analyst.github.io](https://llm-as-dataset-analyst.github.io/)  
- 🎬 **Video Presentation**: [YouTube](https://www.youtube.com/watch?v=pw-ZPzlXtQA)

> 🧠 *Discover subpopulation structures using Large Language Models (LLMs) with linguistic interpretability and automation.*

---

## 🧩 Overview

<div align="center">
  <img src="fig/ssd_llm.png" alt="SSD-LLM Overview" width="800"/>
</div>

**SSD-LLM** is an innovative framework for discovering subpopulation structures within datasets using Large Language Models (LLMs). By leveraging LLMs' extensive world knowledge and advanced reasoning capabilities, SSD-LLM offers:

- ✨ **Linguistically Interpretable Subpopulation Discovery**: Understand dataset structures through natural language  
- 🤖 **Automated Dataset Analysis**: Use LLMs to uncover subpopulation patterns with minimal human effort  
- 🔄 **Comprehensive Workflow**: End-to-end pipeline for subpopulation discovery and evaluation  
- 🔌 **Flexible Integration**: Compatible with various MLLMs and LLMs

**Future Directions**:
- 📊 **Diverse structure discovery**: Tailoring subpopulation forms to specific downstream needs  
- 🖼️ **Multimodal extension**: Applying SSD-LLM to vision and multimodal datasets  
- ✅ **Unbiased data generation**: Supporting fair and balanced dataset development



## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/llm-as-dataset-analyst/SSDLLM.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd SSDLLM
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```



## 📁 Project Structure

```
SSDLLM/
├── captions/               # Pre-captioned datasets (e.g., with LLaVA1.5-7B)
├── config/                 # YAML configs for datasets and pipeline settings
├── run.sh                  # Main entry script
├── utils.py                # Configuration and utility functions
└── step1_image_caption/    # Custom dataset captioning logic
```

- `captions/`: Pre-captioned datasets using `llava1.5-7b`; ready for direct use.  
- `config/`: Contains YAML configs:  
  - `0_summary.yaml`: Sets pipeline hyperparameters (auto-saved to `output/` after running).  
  - Other files define task name, dataset name, class count, etc. Follow provided examples to format your own.  
- `run.sh`: Pipeline launcher script. Supports switching `mllm` and `llm`—ensure corresponding models are prepared.  
- `utils.py`: Contains OpenAI API key and helper functions.  
- `step1_image_caption/`: Batch captioning scripts for custom datasets (supports ImageFolder format).




## 🛠️ Configuration

1. **Set your OpenAI API Key** in `utils.py`:
   ```python
   api_key = "your-openai-api-key"
   ```

2. **Adjust `run.sh` Parameters**:
   ```bash
   mllm_name=llava1.5-7b
   llm_name=gpt-3.5-turbo
   for class_name in mood
   ```
   - `mllm_name`: Multimodal LLM for captioning  
   - `llm_name`: Main LLM for reasoning & subpopulation discovery  
   - `class_name`: Dataset to analyze (check `/config` for available ones)

3. **Prepare Custom Dataset (Optional)**:
   - Format your images using `ImageFolder` structure:
     ```
     dataset/
     ├── class_a/
     ├── class_b/
     └── ...
     ```
   - Modify `step1_image_caption/scripts_infer_batch.sh` for inference logic.



## 🚀 Usage

Run the pipeline using:
```bash
bash run.sh
```



## 🙏 Acknowledgements

This project is benefited from the following repositories:
- [ICTC](https://github.com/sehyunkwon/ICTC)
- [LLAVA](https://github.com/haotian-liu/LLaVA)

Thanks for their great works!

## 📚 Citation

If you find our work helpful, please cite us:

```bibtex
@inproceedings{luo2025llm,
  title={LLM as dataset analyst: Subpopulation structure discovery with large language model},
  author={Luo, Yulin and An, Ruichuan and Zou, Bocheng and Tang, Yiming and Liu, Jiaming and Zhang, Shanghang},
  booktitle={European Conference on Computer Vision},
  pages={235--252},
  year={2025},
  organization={Springer}
}
```


🌟 **Star this repo** if you find it useful! Contributions and feedback are welcome 🙌