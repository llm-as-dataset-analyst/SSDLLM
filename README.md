# SSD-LLM [ECCV 2024]

![SSD-LLM Overview](fig/ssd_llm.png)

## Overview
SSD-LLM is an innovative framework for discovering subpopulation structures within datasets using Large Language Models (LLMs). By harnessing LLMs' broad world knowledge and advanced capabilities in reasoning, summarization, and instruction-following, this approach has several key features:
- **Linguistically Interpretable Subpopulation Discovery**: Provides insights into dataset structures in an understandable language format
- **Automated Dataset Analysis**: Utilizes LLMs’ capabilities to automate the discovery of subpopulation patterns
- **Comprehensive Workflow**: Designed to address a wide range of downstream tasks related to subpopulations
- **Flexible Integration**: Supports different MLLMs and LLMs

Future Directions:
- **Exploration of Diverse Subpopulation Structures**: Tailoring subpopulation structure forms to meet specific task requirements
- **Extension to Vision and Multimodal Tasks**: Broadening the scope of SSD-LLM to more visual and multimodal datasets
- **Development of Unbiased Datasets**: Promoting the creation of unbias datasets, enabling more reliable analysis and model training

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/llm-as-dataset-analyst/SSDLLM.git
   ```

2. Navigate to the project directory:

   ```bash
   cd SSDLLM
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

1. **Captions Directory**: Contains pre-captioned datasets using `llava1.5`. These can be used directly.

2. **Custom Dataset Captioning**: 
   - To caption your own dataset, refer to `step1_image_caption/scripts_infer_batch.sh` for modifications.
   - Prepare your dataset in a format readable by PyTorch's `ImageFolder`, e.g.,

     ```
     imagenet
     ├── bird
     ├── boat
     ├── cat
     ├── dog
     ```

   - Different MLLMs can be used for captioning, and you can modify them for batch inference from Hugging Face demos.

3. **Config Directory**:
   - `0_summary.yaml` sets hyperparameters for different stages of the SSDLLM pipeline. Adjust these when running different datasets. The used hyperparameters are automatically saved in the `output` directory when running the code.
   - Other YAML files correspond to different datasets, including task name, dataset name, class count, and average count per class. These settings affect the final processing form, so they need to be configured properly. Examples are provided for reference.
   - You can format your own dataset similarly in YAML.

4. **run.sh**: 
   - Sets the execution logic for the entire codebase. You can modify the `mllm` and `llm` names, but ensure the corresponding MLLM and LLM exist.

## Configuration

1. Open the `utils.py` file and locate the following code snippet:

   ```python
   api_key="sk-tC6amT2brKjK11nAC0830178D6E748F18072A17dE9D53784"
   ```

   Replace `api_key` with your OpenAI API key.

2. Open the `run.sh` file and modify the following parameters as needed:

   ```bash
   mllm_name=llava1.5-7b
   llm_name=gpt-3.5-turbo
   
   for class_name in mood
   ```

   `mllm_name`: Specify the multimodal language model to use (e.g., llava1.5-7b)
   
   `llm_name`: Choose the language model (e.g., gpt-3.5-turbo, gpt-4)
   
   `class_name`: Set the dataset name you need to process. The available dataset names that can be executed directly are listed in the config folder.

   Make sure the corresponding configurations for your chosen models are properly set up.

## Usage

Run the following command to start the script:

   ```bash
   bash run.sh
   ```

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request.

## License

This project is licensed under the MIT License.