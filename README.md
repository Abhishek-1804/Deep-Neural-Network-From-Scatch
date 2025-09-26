# Deep Neural Network From Scratch

A complete implementation of an N-Layered Neural Network built from scratch using NumPy for multi-class classification on the MNIST handwritten digits dataset.

## 🎯 Project Overview

This project demonstrates a deep understanding of neural network fundamentals by implementing every component from the ground up - no deep learning frameworks like TensorFlow or PyTorch for the core neural network logic. The implementation includes:

- **Forward Propagation**: Data flow from input to output through multiple layers
- **Backward Propagation**: Gradient calculation and error propagation back through the network
- **Multiple Activation Functions**: ReLU, Sigmoid, and Softmax implementations
- **Cost Function**: Cross-entropy loss for multi-class classification
- **Parameter Optimization**: Gradient descent for weight and bias updates

## 🏗️ Architecture

The neural network supports **N-layered architecture** with the following components:

### **Activation Functions**
- **ReLU (Rectified Linear Unit)**: `f(x) = max(0, x)` - Used in hidden layers for its versatility and computational efficiency
- **Sigmoid**: `f(x) = 1/(1 + e^(-x))` - Suitable for binary classification tasks
- **Softmax**: `f(x) = e^(xi)/Σe^(xj)` - Used in the output layer for multi-class classification

### **Core Components**
- **Weights and Biases**: Learnable parameters that determine connection strength and decision flexibility
- **Cost Function**: Cross-entropy loss measuring prediction accuracy
- **Gradient Descent**: Optimization algorithm minimizing loss by iteratively adjusting parameters

## 📊 Dataset

The model is trained and tested on the **MNIST dataset**:
- **Training Set**: 5,000 samples (subset of original 60,000)
- **Test Set**: 5,000 samples (subset of original 10,000)
- **Input Features**: 784 (28×28 pixel images flattened)
- **Output Classes**: 10 (digits 0-9)
- **Data Preprocessing**: Pixel normalization (0-255 → 0-1) and one-hot encoding for labels

## 🚀 Implementation Details

### Network Configuration
```python
layers_dims = [784, 10, 10]  # Input: 784, Hidden: 10, Output: 10
learning_rate = 0.05
num_iterations = 2500
```

### Key Functions
- `initialize_parameters_deep()` - Initialize weights and biases
- `L_model_forward()` - Complete forward propagation
- `compute_cost()` - Calculate cross-entropy loss
- `L_model_backward()` - Complete backward propagation
- `update_parameters()` - Gradient descent parameter updates
- `predict()` - Generate predictions from trained model

## 📈 Performance

The implemented model achieves impressive results:
- **Training Accuracy**: 94.28%
- **Test Accuracy**: 92.78%
- **Training Time**: 2,500 iterations with decreasing cost function

## 🔬 Mathematical Foundation

### Forward Propagation
For each layer l: `Z[l] = W[l] × A[l-1] + b[l]` and `A[l] = activation(Z[l])`

### Backward Propagation
- **Cost Gradient**: `dA[L] = -Y/A[L] + (1-Y)/(1-A[L])`
- **Parameter Gradients**: `dW[l] = (1/m) × dZ[l] × A[l-1]^T`
- **Bias Gradients**: `db[l] = (1/m) × sum(dZ[l])`

### Parameter Updates
`W[l] = W[l] - learning_rate × dW[l]`
`b[l] = b[l] - learning_rate × db[l]`

## 🛠️ Technology Stack

- **Python 3.x**
- **NumPy**: Core mathematical operations and array handling
- **Matplotlib**: Visualization of training progress
- **TensorFlow/Keras**: Dataset loading only (not for neural network implementation)
- **Pandas & Seaborn**: Data analysis and visualization support

## 📋 Requirements

```bash
numpy>=1.19.0
matplotlib>=3.3.0
tensorflow>=2.0.0  # For MNIST dataset only
pandas>=1.1.0
seaborn>=0.11.0
```

## 🚀 Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Abhishek-1804/Deep-Neural-Network-From-Scatch.git
   cd Deep-Neural-Network-From-Scatch
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**:
   Open `Deep_neural_network_from_scratch.ipynb` in Jupyter Notebook or Google Colab

4. **Customize the network**:
   ```python
   # Modify architecture
   layers_dims = [784, 128, 64, 10]  # Add more hidden layers
   
   # Adjust hyperparameters
   learning_rate = 0.01
   num_iterations = 5000
   ```

## 🎯 Key Features

- **Scalable Architecture**: Easily modify the number of layers and neurons
- **Multiple Activation Functions**: Switch between ReLU, Sigmoid, and Softmax
- **Comprehensive Implementation**: Every component built from mathematical fundamentals
- **Educational Value**: Clear code structure for learning neural network concepts
- **Visualization**: Training progress plotting and cost function monitoring

## 🔮 Future Enhancements

- **Regularization**: Add L1/L2 regularization to prevent overfitting
- **Optimization Algorithms**: Implement Adam, RMSprop, and momentum
- **Batch Processing**: Mini-batch gradient descent implementation
- **Different Datasets**: Extend to other classification problems
- **Advanced Initializations**: Xavier/He initialization methods

## 📖 Learning Outcomes

This project demonstrates:
- Deep understanding of neural network mathematics
- Implementation of gradient descent optimization
- Knowledge of activation functions and their derivatives
- Experience with multi-class classification problems
- Ability to build ML models without high-level frameworks

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements. Some areas for contribution:
- Performance optimizations
- Additional activation functions
- Better visualization features
- Code documentation improvements

## 📧 Contact

**Abhishek Deshpande**
- GitHub: [@Abhishek-1804](https://github.com/Abhishek-1804)
- Website: [abhishekdp.com](https://abhishekdp.com)

---

*This project showcases the implementation of fundamental deep learning concepts from scratch, providing a solid foundation for understanding how neural networks work under the hood.*