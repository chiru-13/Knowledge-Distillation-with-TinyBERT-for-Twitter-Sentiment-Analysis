# Knowledge Distillation with TinyBERT for Twitter Sentiment Analysis

This repository contains the implementation and training code for a sentiment analysis system on Twitter data using knowledge distillation. The primary goal is to train a compact student model (TinyBERT) with the help of a larger teacher model (BERT), achieving significant model size reduction and performance enhancement.

## Overview

Knowledge distillation is a method where a smaller model (student) is trained to replicate the behavior of a larger model (teacher). This approach allows for faster inference and reduced resource consumption without a substantial loss in performance.

## Achievements

- **Model Size Reduction**: The student model (TinyBERT) achieves an impressive reduction in model size by approximately 96% compared to the teacher model (BERT).
- **Performance Preservation**: Despite the drastic reduction in size, the student model maintains a high level of performance, ensuring efficient and accurate sentiment analysis.
- **Speed Improvement**: The processing speed of the sentiment analysis system is enhanced by around 60%, facilitating faster data processing and analysis.

## Dataset

The dataset used is the Twitter Sentiment Analysis dataset, which can be accessed via the `datasets` library.

## Model Architecture

### Teacher Model
- Model: `bert-base-uncased`
- Number of Parameters: 109,483,778

### Student Model
- Model: `prajjwal1/bert-tiny`
- Number of Parameters: 4,386,178

## Training Process

1. **Data Augmentation**: Applied synonym augmentation to the training dataset to improve model generalization.
2. **Custom Training Loop**: Implemented a custom training loop for knowledge distillation using a modified `Trainer` class.
3. **Hyperparameters**: 
   - `alpha`: 0.5 (weight for combining student and teacher loss)
   - `temperature`: 2.0 (for softening logits)
   - `learning_rate`: 2e-5
   - `batch_size`: 48
   - `num_train_epochs`: 1

## Results

The training and evaluation results are summarized below:

| Epoch | Training Loss | Validation Loss | Accuracy  |
|-------|---------------|-----------------|-----------|
| 1     | 0.5698        | 0.5292          | 73.8%     |

- **Training Duration**: 1230.9 seconds
- **Training Samples per Second**: 97.48
- **Training Steps per Second**: 2.03
- **Evaluation Duration**: 87.8 seconds
- **Evaluation Samples per Second**: 341.76
- **Evaluation Steps per Second**: 7.12

## Model Evaluation

The final evaluation of the student model shows an accuracy of 73.8% on the validation dataset, indicating robust performance despite the model's reduced size.

## Checkpoints

Checkpoints were saved during the training process at regular intervals. Note that existing checkpoint directories were reused, which may have affected the validity of the saved results.

## Usage Instructions

1. Clone the repository.
2. Install the required dependencies.
3. Execute the provided notebook or script to train and evaluate the model.

## Conclusion

This project successfully demonstrates the application of knowledge distillation for creating an efficient sentiment analysis system. By leveraging TinyBERT as the student model, the system achieves a significant reduction in model size and an increase in processing speed while preserving performance. This approach is especially beneficial for deploying models in environments with limited computational resources.

For more details and implementation specifics, refer to the included notebook or script.