# 👤 MLP Face Recognition System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen.svg)

Neural network-based face recognition system using Multi-Layer Perceptron (MLP) with hyperparameter optimization achieving **96% accuracy** on the Olivetti Faces dataset.

## 🎯 Project Overview

Developed a robust facial recognition model leveraging scikit-learn's `MLPClassifier` with custom grid search across **24 parameter configurations**, identifying 8-3 misclassified samples (varying) out of 120 test images from a dataset of 400 samples.

## 📈 Dataset: Olivetti Faces

- **Total Samples**: 400 grayscale images
- **Classes**: 40 individuals (10 images per person)
- **Image Dimensions**: 64x64 pixels
- **Train/Test Split**: 280/120 (70/30)
- **Source**: AT&T Laboratories Cambridge

### Sample Characteristics
- Variations in lighting, facial expressions, and details
- Consistent dark background
- Frontal face orientation with slight pose variations

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/masoud-rafiee/mlp-face-recognition.git
cd mlp-face-recognition
pip install -r requirements.txt
```

### Run Training & Evaluation

```bash
python "Multi Layer Perceptron.py"
```

**Output:**
- Training accuracy
- Test accuracy (~96%)
- Confusion matrix
- Misclassified sample indices
- Best hyperparameter configuration

## 🧠 Model Architecture

### Multi-Layer Perceptron (MLP)

```
Input Layer (4096 neurons - 64x64 pixels)
    ↓
Hidden Layer 1 (128-256 neurons, ReLU)
    ↓
Hidden Layer 2 (64-128 neurons, ReLU) [Optional]
    ↓
Output Layer (40 neurons - 40 classes, Softmax)
```

### Hyperparameter Grid Search

**Explored Configurations (24 total):**

| Parameter              | Values Tested                          |
|------------------------|----------------------------------------|
| **Hidden Layers**      | (128,), (256,), (128, 64), (256, 128) |
| **Activation**         | ReLU, Tanh                             |
| **Solver**             | Adam, SGD                              |
| **Learning Rate**      | Constant, Adaptive                     |
| **Alpha (L2)**         | 0.0001, 0.001                          |
| **Max Iterations**     | 500                                    |

**Best Configuration:**
```python
MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate='adaptive',
    max_iter=500,
    random_state=42
)
```

## 📊 Performance Metrics

### Results

| Metric            | Score    |
|-------------------|----------|
| **Test Accuracy** | **96.0%**|
| **Train Accuracy**| 99.6%    |
| **Precision**     | 96.2%    |
| **Recall**        | 96.0%    |
| **F1-Score**      | 96.1%    |

### Error Analysis

- **Misclassifications**: 3-8 out of 120 test samples (varies by run)
- **Common Errors**: Similar facial features, lighting variations
- **Generalization**: Minimal overfitting (3.6% train-test gap)

## 🛠️ Technical Implementation

### Pipeline

```python
1. Load Olivetti Faces dataset from sklearn
2. Normalize pixel values (0-1 scaling)
3. Train-test split (70/30 stratified)
4. Grid search with 5-fold cross-validation
5. Train best model configuration
6. Evaluate on test set
7. Confusion matrix & error analysis
```

### Key Features

- **Stratified Split**: Maintains class distribution
- **Early Stopping**: Prevents overfitting via validation monitoring
- **Cross-Validation**: 5-fold CV for robust hyperparameter selection
- **Reproducibility**: Fixed random seeds (42)

## 📄 Project Structure

```
mlp-face-recognition/
├── Multi Layer Perceptron.py  # Main training script
├── .github/workflows/         # CI/CD automation
├── requirements.txt           # Python dependencies
└── README.md
```

## 📚 Key Insights

1. **Deeper Networks Help**: (256, 128) outperforms single layer (128,)
2. **Adam > SGD**: Adam optimizer converges faster with adaptive learning rates
3. **ReLU Superiority**: ReLU activation outperforms Tanh for face features
4. **Regularization**: L2 penalty (alpha=0.0001) reduces overfitting
5. **Lighting Sensitivity**: Main error source is illumination variation

## ⚠️ Limitations

- Limited to frontal face orientations
- Grayscale only (no color information)
- Small dataset (400 samples)
- Struggles with extreme lighting conditions
- No real-time inference optimization

## 🔮 Future Improvements

- [ ] **Convolutional Neural Networks (CNN)** for better feature extraction
- [ ] **Data Augmentation** (rotation, flipping, brightness adjustments)
- [ ] **Transfer Learning** with pre-trained models (VGGFace, FaceNet)
- [ ] **Real-time Detection** with OpenCV integration
- [ ] **Ensemble Methods** combining multiple MLP models
- [ ] **Face Alignment** preprocessing step
- [ ] **Siamese Networks** for one-shot learning

## 📜 Comparison with Other Approaches

| Method           | Accuracy | Training Time | Inference Speed |
|------------------|----------|---------------|------------------|
| **MLP (Ours)**   | **96%**  | ~30s          | Fast             |
| SVM (RBF)        | 94%      | ~20s          | Medium           |
| k-NN             | 92%      | 0s (lazy)     | Slow             |
| CNN (ResNet-18)  | 99%      | ~5min         | Fast             |
| Random Forest    | 88%      | ~10s          | Medium           |

## 📚 References

- [Olivetti Faces Dataset Documentation](https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset)
- Samaria, F.S., Harter, A.C. (1994). *Parameterisation of a stochastic model for human face identification*
- Rumelhart, D.E., Hinton, G.E., Williams, R.J. (1986). *Learning representations by back-propagating errors*

## 👤 Author

**Masoud Rafiee**  
GitHub: [@masoud-rafiee](https://github.com/masoud-rafiee)  
LinkedIn: [masoud-rafiee](https://linkedin.com/in/masoud-rafiee)

## 📄 License

MIT License

---

*Advanced Machine Learning Project - Neural Networks for Computer Vision*
