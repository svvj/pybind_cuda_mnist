# MNIST Classifier with CUDA and PyBind11

This project implements a neural network classifier for the MNIST dataset using CUDA acceleration and C++/Python integration via PyBind11.

## Project Structure

- **Python Scripts**
  - [`main.py`](main.py) - Main execution script
  - [`train.py`](train.py) - Model training implementation
  - [`test.py`](test.py) - Model evaluation

- **Model Files**
  - Epoch 10 weights: model_epoch_10_*.npy
  - Epoch 20 weights: model_epoch_20_*.npy
  - Final weights: model_final_*.npy

- **Visualizations**
  - [`mnist_samples.png`](mnist_samples.png) - Example MNIST digits
  - [`mnist_training.png`](mnist_training.png) - Training visualization
  - [`confusion_matrix.png`](confusion_matrix.png) - Model evaluation results
  - [`training_history.png`](training_history.png) - Training metrics over time

- **C++/CUDA Integration**
  - [`src`](src) - C++ source files with CUDA acceleration
  - [`extern/pybind11`](extern/pybind11) - PyBind11 submodule for C++/Python binding
  - [`CMakeLists.txt`](CMakeLists.txt) - CMake build configuration

## Setup and Execution

### Prerequisites
- Python with numpy, matplotlib
- CUDA toolkit (tested with CUDA 12.4)
- CMake
- C++ compiler

### Adding PyBind11 as a Git Submodule

If you're cloning this repository for the first time:
```bash
git clone --recursive https://github.com/svvj/pybind_cuda_mnist.git
```

If you've already cloned the repository without the submodule:
```bash
git submodule init
git submodule add https://github.com/pybind/pybind11.git extern/pybind11
git submodule update --init --recursive
```

To update the submodule to the latest version:
```bash
git submodule update --remote
```

### Building
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Running
```bash
# Using the batch script
run_cuda_main.bat

# Or directly with Python
python main.py
```

## Model Training and Evaluation
- Train a new model: `python train.py`
- Evaluate model performance: `python test.py`

## C++/CUDA Integration
The project uses PyBind11 to integrate C++ CUDA-accelerated code with Python, offering high-performance neural network operations while maintaining a convenient Python interface. All CUDA operations have been tested and verified with CUDA 12.4.