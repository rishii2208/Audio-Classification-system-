# Audio Classification System

## Overview

I created this project a part of evaluation scheme under Dr. Sachin Taran for my courwork on subject titled " Speech Recognition". This project implements an audio classification system using deep learning techniques to categorize urban sounds into different classes. The system processes audio files using Mel-Frequency Cepstral Coefficients (MFCC) for feature extraction and employs various neural network architectures for classification.

The project is divided into three main parts:
1. **Exploratory Data Analysis (EDA)**: Understanding the dataset characteristics
2. **Data Preprocessing**: Feature extraction and preparation
3. **Model Creation and Testing**: Building and evaluating classification models

## Dataset

This project uses the UrbanSound8K dataset, which contains 8,732 labeled sound excerpts of urban sounds from ten categories:
- Air conditioner
- Car horn
- Children playing
- Dog bark
- Drilling
- Engine idling
- Gunshot
- Jackhammer
- Siren
- Street music

Each audio clip is a WAV file with a duration of up to 4 seconds.

You can download the dataset from: [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/download-urbansound8k.html)

## Features

- Audio feature extraction using MFCC
- Comprehensive data analysis and visualization
- Implementation of multiple deep learning models:
  - Artificial Neural Networks (ANN)
  - Convolutional Neural Networks (CNN)
- Performance evaluation and model comparison

## Requirements

```
librosa==0.8.0
numpy>=1.19.5
pandas
matplotlib
scikit-learn
tensorflow>=2.0.0
scipy>=1.4.1
soundfile>=0.9.0
```

## Project Structure

```
├── Audio-Classification-EDA.ipynb       # Exploratory Data Analysis
├── Audio-Classification-Data-Preprocessing.ipynb  # Data preprocessing
├── Audio-Classification-Model.ipynb     # Model creation and evaluation
├── README.md                            # Project documentation
└── requirements.txt                     # Required packages
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/rishii2208/Audio-Classification-system-
cd Audio-Classification-system-
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the UrbanSound8K dataset from the link provided above and extract it to the project directory.

## Usage

1. **Exploratory Data Analysis**:
   - Run the `Audio-Classification-EDA.ipynb` notebook to explore the dataset characteristics.
   - This notebook visualizes audio waveforms, spectrograms, and class distributions.

2. **Data Preprocessing**:
   - Run the `Audio-Classification-Data-Preprocessing.ipynb` notebook to extract MFCC features from the audio files.
   - The notebook processes the raw audio data and prepares it for model training.

3. **Model Creation and Testing**:
   - Run the `Audio-Classification-Model.ipynb` notebook to train and evaluate the classification models.
   - The notebook implements both ANN and CNN architectures and compares their performance.

## Technologies Used

- **Python**: Primary programming language
- **Librosa**: Audio processing and feature extraction
- **SciPy**: Scientific computing and signal processing
- **NumPy**: Numerical operations
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **TensorFlow/Keras**: Deep learning model implementation
- **Scikit-learn**: Machine learning utilities and evaluation metrics

## Results

The project demonstrates the effectiveness of deep learning models in classifying urban sounds. The CNN model generally outperforms the ANN model, particularly in terms of accuracy and F1-score. Detailed performance metrics and visualizations are provided in the model evaluation notebook.

## Future Work

- Implement more advanced architectures such as recurrent neural networks (RNNs) or transformer-based models
- Explore different feature extraction techniques beyond MFCC
- Develop a real-time audio classification system
- Implement data augmentation techniques to improve model robustness

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The UrbanSound8K dataset creators for providing a comprehensive urban sound dataset
- The developers of Librosa, TensorFlow, and other open-source libraries used in this project

##Citations:

- https://urbansounddataset.weebly.com/download-urbansound8k.html.
- https://github.com/abishek-as/Audio-Classification-Deep-Learning
- https://medium.datadriveninvestor.com/how-to-write-a-good-readme-for-your-data-science-project-on-github-ebb023d4a50e
- https://ai.google.dev/edge/mediapipe/solutions/audio/audio_classifier/python
- https://www.jonnor.com/tag/audio-classification/
- https://github.com/cyrta/UrbanSounds/blob/master/data/UrbanSound8K/README.txt
- https://towhee.io/audio-classification/panns/src/branch/main/README.md
- https://www.youtube.com/watch?v=QVUthvMzkbo
- https://github.com/ravising-h/Urbansound8k/blob/master/README.md
- https://github.com/catiaspsilva/README-template
- https://www.digitalocean.com/community/tutorials/audio-classification-with-deep-learning
- https://urbansounddataset.weebly.com/urbansound8k.html
- https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/
- https://colab.research.google.com/github/jsalbert/sound_classification_ml_production/blob/main/notebooks/UrbanSound8k_data_exploration.ipynb
- https://soundata.readthedocs.io/en/stable/_modules/soundata/datasets/urbansound8k.html



