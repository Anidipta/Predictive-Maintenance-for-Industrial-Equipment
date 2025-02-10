# Gearbox Fault Diagnosis using CNN

Diagnosing gearbox failures using sensor data collected under various loading conditions. In this project, Convolutional Neural Networks (CNNs) is implemented to classify sensor readings as coming from either healthy or faulty (broken) gears. The study evaluates several optimizers (Adam, SMORMS3, and RMSProp) and provides comprehensive reporting, including exploratory data analysis, training metrics, confusion matrices, and a SWOT analysis with a discussion on future scope.

---

## 📌 Project Overview

The goal of this project is to develop a predictive maintenance system that can identify gearbox failures—specifically, broken gear teeth—using time-series sensor data. Using multiple CSV files that record sensor readings under various loading conditions (0%–90%), we preprocessed the data, engineered features, and trained several CNN models for binary classification. Detailed evaluation metrics and visualizations are provided for each model.

---

## 📁 Dataset Description

The dataset is composed of sensor readings from healthy and broken gears collected under ten different loading conditions (0%, 10%, 20%, …, 90%). For each loading condition, the following steps were applied:

- **Healthy Gears:** Files named `h30hzX.csv` (where X represents the load percentage) contain sensor readings from gears in good condition.
- **Broken Gears:** Files named `b30hzX.csv` contain sensor readings from gears with a broken tooth.

### Data Structure
Each CSV file includes:
- **Sensor Readings:** `a1`, `a2`, `a3`, `a4`
- **Additional Columns Added:**
  - `load`: The loading condition (e.g., 0, 10, …, 90)
  - `failure`: A binary indicator (0 = healthy, 1 = broken)

After labeling, the individual CSV files were concatenated into two dataframes (one for healthy and one for broken gears) and finally merged into one aggregated dataset.

---

## Data Preprocessing

1. **Data Ingestion:**  
   The CSV files were read using [Pandas](https://pandas.pydata.org/) and then merged to form a comprehensive dataset.

2. **Labeling:**  
   Additional columns (`load` and `failure`) were added to each dataset to indicate the operating load and gear condition.

3. **Concatenation:**  
   Healthy and broken datasets were concatenated to form a single dataframe.

4. **Exploratory Analysis:**  
   Descriptive statistics were computed (using `df.describe()`), and visualizations (boxplots and histograms) were generated to understand the distribution of each sensor reading.

5. **Feature Scaling & Splitting:**  
   - The dataset was split into training and testing sets (67% training, 33% testing).
   - Features were standardized using [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
   - The data was reshaped to add a channel dimension required for the CNN input.

*Example Code Snippet:*
```python
X = df.drop(columns=['failure']).values
y = df['failure'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for CNN (1D input with channel dimension)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
```

---

## Exploratory Data Analysis (EDA)

EDA was performed using [Seaborn](https://seaborn.pydata.org/) and [Matplotlib](https://matplotlib.org/):

- **Boxplots:** Used to detect outliers and visualize the spread of sensor values.
- **Histograms with KDE:** Showed the distribution of each sensor reading.
- **Summary Statistics:** Provided insights into central tendencies and variances.

---

## Model Architecture

Three CNN models were developed with a similar architecture but different optimizers:

### Common Architecture:
- **Input Layer:** 1D input corresponding to sensor readings.
- **Convolutional Layers:**
  - First Conv1D layer: 64 filters, kernel size 2, ReLU activation.
  - Second Conv1D layer: 128 filters, kernel size 2, ReLU activation.
- **Batch Normalization:** Applied after each convolutional layer.
- **MaxPooling:** Downsampling after the first convolution.
- **Dropout Layers:** Used to prevent overfitting (dropout rates of 0.25 and 0.5).
- **Flatten Layer:** To transition from convolutional to dense layers.
- **Dense Layers:** 
  - A hidden dense layer with 64 neurons (ReLU activation).
  - Output layer with 1 neuron (sigmoid activation for binary classification).

### Optimizers Evaluated:
1. **Adam Optimizer**
2. **SMORMS3 Optimizer** (via TensorFlow Addons)
3. **RMSProp Optimizer**

*Model Summary (Example using Adam):*
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d (Conv1D)              (None, 4, 64)             192       
batch_normalization (BatchNo (None, 4, 64)             256       
max_pooling1d (MaxPooling1D) (None, 2, 64)             0         
dropout (Dropout)            (None, 2, 64)             0         
conv1d_1 (Conv1D)            (None, 1, 128)            16512     
batch_normalization_1 (Batch (None, 1, 128)            512       
dropout_1 (Dropout)          (None, 1, 128)            0         
flatten (Flatten)            (None, 128)               0         
dense (Dense)                (None, 64)                8256      
dropout_2 (Dropout)          (None, 64)                0         
dense_1 (Dense)              (None, 1)                 65        
=================================================================
Total params: 25,793
Trainable params: 25,409
Non-trainable params: 384
_________________________________________________________________
```

---

## Experiments and Results

Each model was trained for 30 epochs with a batch size of 512. Below are highlights from the three experiments:

### Adam Optimizer
- **Test Accuracy:** ~61.77%
- **Loss:** ~0.6387 on the test set.
- **Classification Report:**
  - *No Failure*: Precision ~0.66, Recall ~0.48, F1-Score ~0.56
  - *Failure*: Precision ~0.59, Recall ~0.75, F1-Score ~0.66
- **Visualizations:**
  - **Training Curves:** Plots of training and validation accuracy/loss over epochs.
  - **Confusion Matrix:** A heatmap showing true vs. predicted labels.

### SMORMS3 Optimizer
- **Test Accuracy:** ~60.05%
- **Classification Report:**
  - *No Failure*: Precision ~0.71, Recall ~0.35, F1-Score ~0.47
  - *Failure*: Precision ~0.57, Recall ~0.85, F1-Score ~0.68

### RMSProp Optimizer
- **Test Accuracy:** ~61.65%
- **Classification Report:**
  - *No Failure*: Precision ~0.68, Recall ~0.45, F1-Score ~0.54
  - *Failure*: Precision ~0.59, Recall ~0.78, F1-Score ~0.67


---

## SWOT Analysis

### Strengths
- **Robust Data Aggregation:** Multiple sensor files across different load conditions are merged and labeled consistently.
- **Effective CNN Architecture:** Incorporates Batch Normalization, Dropout, and multiple convolutional layers for robust feature extraction.
- **Comprehensive Evaluation:** Detailed metrics, visualizations, and multiple optimizers provide in-depth performance insights.

### Weaknesses
- **Moderate Classification Accuracy:** Test accuracy hovers around 61–62%, suggesting room for improvement.
- **Overfitting Concerns:** Despite using dropout layers, further regularization may be required.
- **Limited Feature Engineering:** The analysis is based solely on raw sensor readings without advanced signal processing.

### Opportunities
- **Enhanced Feature Extraction:** Incorporate techniques like Fourier or wavelet transforms for richer feature representation.
- **Hyperparameter Optimization:** Employ automated methods (Grid Search, Bayesian Optimization) to fine-tune model parameters.
- **Ensemble Methods:** Combine multiple models for improved robustness and accuracy.
- **Real-Time Deployment:** Develop real-time fault detection systems integrated with IoT devices.

### Threats
- **Data Variability:** Sensor noise and environmental factors could impact model reliability.
- **Scalability:** Scaling to larger datasets or more complex systems may demand higher computational resources.
- **Operational Risks:** Misclassifications in real-world settings could lead to costly maintenance errors.

---

## Future Scope

- **Advanced Preprocessing:**  
  Explore data augmentation, denoising techniques, and more sophisticated normalization to improve the quality of sensor inputs.

- **Alternative Architectures:**  
  Investigate recurrent neural networks (RNNs), transformers, or hybrid models to capture temporal dependencies in sensor data.

- **Hyperparameter and Architecture Search:**  
  Utilize automated tuning frameworks (e.g., Keras Tuner) to systematically improve model performance.

- **Real-Time Fault Detection:**  
  Integrate the trained model with IoT platforms for continuous monitoring and predictive maintenance.

- **Explainability:**  
  Incorporate interpretability methods (e.g., SHAP, LIME) to understand the model’s decision-making process and build trust with end users.

---

## Conclusion

This project demonstrates the application of CNNs for gearbox fault diagnosis using sensor data. Although the current models achieve moderate accuracy (~61–62%), the comprehensive analysis—including exploratory data analysis, multiple optimizer evaluations, and a SWOT analysis—highlights both the potential and the challenges. Future work will focus on advanced preprocessing, model tuning, and real-time integration to enhance predictive maintenance systems.

---

### Contributor

[Anidipta Pal](https://github.com/Anidipta)
