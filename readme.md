# [VR Mini Project 2](https://github.com/vr-assignments/VR_MiniProject2_IMT2022_073_086_103): Multimodal Visual Question Answering and Fine-tuning with LoRA on the Amazon Berkeley Objects Dataset

This repository contains the code and resources for **Mini Project 2** of the AIM825 Visual Recognition course. The project focuses on creating a Visual Question Answering (VQA) dataset using the Amazon Berkeley Objects (ABO) dataset, evaluating baseline models such as BLIP and ViLT, fine-tuning BLIP using Low-Rank Adaptation (LoRA), and assessing performance using standard metrics.

This README provides details on the project's file structure and instructions for running the inference script. For comprehensive information regarding the project, please refer to the [report](report.pdf).

## 📁 Folder Structure
```
📂data
 ┣ 📂dataset
 ┃   ┣ 📂images
 ┃   ┣ 📄cleaned.csv
 ┃   ┣ 📄qna.csv
 ┃   ┗ 📄qna_2.csv
 ┣ 📂baseline
 ┃   ┣📄 blip_baseline_pred.csv
 ┃   ┗📄 vilt_baseline_pred.csv
 ┗ 📂finetuned
   ┣📄 fintuned_pred_model1.csv
   ┣📄 fintuned_pred_model2.csv
   ┗📄 fintuned_pred_model3.csv
📂code
 ┣ 📄part1-dataset.ipynb
 ┣ 📄part2-baseline.ipynb
 ┣ 📄part3-finetuning.ipynb
 ┣ 📄eval.ipynb
 ┗ 📄predict.ipynb
📂saved_models
 ┣ 📂finetuned_blip_model1
 ┣ 📂finetuned_blip_model2
 ┗ 📂finetuned_blip_model3
📄inference.py
📄requirements.txt
```



## 🔍 File & Folder Descriptions

-   **`code/`**
    -   `part1-dataset.ipynb`: Filters images and metadata from the ABO dataset to create `cleaned.csv`. It then utilizes the Gemini API to generate question-answer pairs, saved as `qna.csv` and `qna_2.csv`.
    -   `part2-baseline.ipynb`: Performs baseline evaluation on `salesforce/blip-vqa-base` and `dandelin/vilt-b32-finetuned-vqa` using the generated dataset.
    -   `part3-finetuning.ipynb`: Fine-tunes the `salesforce/blip-vqa-base` model on the generated dataset using LoRA.
    -   `eval.ipynb`: A helper script that, given a CSV file with predictions and ground truth answers, calculates various performance metrics.
    -   `predict.ipynb`: A helper script that, given a saved model and a dataset, performs predictions on the entire dataset and saves the results.

-   **`data/`**
    -   `dataset/`: Contains a subset of data from the ABO dataset used in this project.
        -   `images/`: Contains images filtered from the ABO dataset. Filenames correspond to those in the original ABO dataset.
        -   `cleaned.csv`: The filtered metadata extracted from the complete ABO dataset.
        -   `qna.csv`: Generated question-answer pairs using the Gemini API, containing approximately 50,000 easy difficulty Q&A pairs.
        -   `qna_2.csv`: Generated question-answer pairs using the Gemini API, containing approximately 25,000 moderate to hard difficulty Q&A pairs.
    -   `baseline/`: Contains prediction outputs from the baseline models (before fine-tuning).
    -   `finetuned/`: Contains prediction outputs from the fine-tuned models.

-   **`saved_models/`**
    Contains three different variants of the BLIP model, fine-tuned from `salesforce/blip-vqa-base`.

-   **`inference.py`**
    The primary inference script used for submission. It predicts answers for a given metadata CSV file and an image folder, then saves the output.

-   **`requirements.txt`**
    A list of Python dependencies required to run the project.

## Steps to Run the `inference.py` script

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vr-assignments/VR_MiniProject2_IMT2022_073_086_103.git
    cd VR_MiniProject2_IMT2022_073_086_103
    ```

2.  **Create and activate a virtual environment:**
    It is recommended to use Python 3.9.
    ```bash
    conda create -n myenv python=3.9
    conda activate myenv
    ```
    Alternatively, using `venv`:
    ```bash
    python3.9 -m venv myenv
    source myenv/bin/activate # On Linux/macOS
    # myenv\Scripts\activate # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the inference script:**
    ```bash
    python inference.py --image_dir data/dataset/images --csv_path data/dataset/qna_2.csv
    ```
    You can also provide different paths to test predictions on your own dataset.
    -   `--image_dir`: Path to the directory containing images.
    -   `--csv_path`: Path to a CSV file containing image filenames, questions, and (optionally) ground truth answers for evaluation purposes by other scripts. The `inference.py` script primarily uses image filenames and questions.

**Note:** The Jupyter notebooks in the `code/` folder were primarily developed and run on Kaggle or Google Colab. If you intend to run them locally, you may need to adjust file paths and resource allocations accordingly.

## Authors

-   [IMT2022073 Vasu Aggarwal](https://github.com/vasuganesha2)
-   [IMT2022086 Ananthakrishna K](https://github.com/Ananthakrishna-K-13)
-   [IMT2022103 Anurag Ramaswamy](https://github.com/Anurag9507)

---