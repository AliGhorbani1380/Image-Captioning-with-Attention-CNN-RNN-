# Image-Captioning-with-Attention-CNN-RNN-
This repository contains the solution for the Bonus Assignment of the Deep Learning course at the University of Tehran. The project implements an Encoder-Decoder model with Attention to generate textual descriptions (captions) for images.
# Image Captioning with Attention (CNN-RNN)

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg) 
![Course](https://img.shields.io/badge/Course-Deep%20Learning%20(Bonus)-blueviolet.svg)
![University](https://img.shields.io/badge/University-University%20of%20Tehran-red.svg)

This repository contains the solution for the **Bonus Assignment** of the Deep Learning course at the University of Tehran. The project implements an **Encoder-Decoder** model with **Attention** to generate textual descriptions (captions) for images.

This is a classic "sequence-to-sequence" problem bridging Computer Vision (CV) and Natural Language Processing (NLP). We use a pre-trained CNN as an Encoder to understand the image and an RNN (LSTM/GRU) as a Decoder with an Attention mechanism to generate the caption word by word.

<img width="844" height="753" alt="image" src="https://github.com/user-attachments/assets/ff87c99c-3b73-454d-aa30-8845204d6fa7" />


---

## 🚀 Project Goals

This assignment provided practical experience with:
* Implementing a complete **Encoder-Decoder (CNN-RNN)** pipeline.
* Preprocessing and building a vocabulary from text data (MS COCO dataset).
* Implementing **Attention Mechanisms** (like Additive and Scaled-Dot Product) to allow the model to "focus" on relevant parts of an image while generating text.
* Training and evaluating a sequence-generation model using metrics like **BLEU Score**.
* Exploring advanced training techniques like **Scheduled Sampling** to improve model stability and performance.

---
## 🏛️ Model Architecture

The model follows a standard Encoder-Decoder architecture:

1.  **CNN Encoder:** A pre-trained CNN (e.g., ResNet) is used to extract high-level feature maps from the input image. These features represent *what* is in the image and *where*.
2.  **Attention Mechanism:** At each step of generation, the Attention module weighs the importance of different regions in the image feature map, based on the decoder's current hidden state.
3.  **RNN Decoder:** An LSTM (or GRU) cell takes the previously generated word and the "context vector" (from Attention) to predict the next word in the sequence.

<img width="357" height="11" alt="image" src="https://github.com/user-attachments/assets/0c11262a-1db8-42a5-8f30-a59b44b9be24" />

---

## 📊 Key Findings

### 1. The Power of Attention
The core finding was that the Attention mechanism is crucial for generating relevant captions. The model learned to ground its generated words in visual evidence, as shown by the attention heatmaps.

### 2. Scheduled Sampling
As noted in the report, standard "Teacher Forcing" can lead to instability during inference. We explored **Scheduled Sampling**, which gradually shifts the model from using ground-truth words to using its own generated words as input during training, making it more robust.

### 3. Quantitative Results (BLEU Score)
The models were evaluated using the BLEU score to measure the similarity between generated captions and human-written ground truths.


---
## 🛠️ Getting Started

To run this project, you'll need the MS COCO dataset and the required Python libraries.

### Prerequisites

* Python 3.9+
* PyTorch
* Torchvision
* Matplotlib
* NumPy
* Tqdm
* NLTK (for BLEU score and tokenization)
* PIL (Pillow)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YourUsername]/[Your-Repo-Name].git
    cd [Your-Repo-Name]
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch torchvision matplotlib numpy tqdm nltk Pillow
    ```
4.  **Download Data:** You will need to download the MS COCO dataset (images and annotations) and place them in the appropriate directory.

### Usage

The project is split into two notebooks:
* `Q2_part1.ipynb`: Contains the implementation of the dataset, model, and training loop, likely with the first attention mechanism.
* `Q2_part2.ipynb`: Contains further experiments, possibly with a different attention mechanism (e.g., Scaled-Dot Product) and Scheduled Sampling.

Run the cells sequentially in Jupyter Notebook to train the model and generate captions.
```bash
jupyter notebook Q2_part1.ipynb
```
