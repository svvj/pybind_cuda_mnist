import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'build', 'Release'))

import numpy as np
import matplotlib.pyplot as plt
import my_cuda_module as cu
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

def load_model(filename_prefix='model_final'):
    """Load the saved model parameters"""
    try:
        W1 = np.load(f'{filename_prefix}_W1.npy')
        b1 = np.load(f'{filename_prefix}_b1.npy')
        W2 = np.load(f'{filename_prefix}_W2.npy')
        b2 = np.load(f'{filename_prefix}_b2.npy')
        print(f"Model loaded from '{filename_prefix}_*.npy' files.")
        return W1, b1, W2, b2
    except FileNotFoundError as e:
        print(f"Error: Model files not found. {e}")
        sys.exit(1)

def main():
    # Load MNIST dataset (for testing)
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    
    # Data normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    
    # Use only test data
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Model parameters
    batch_size = 128
    input_dim = 784
    hidden_dim = 128
    output_dim = 10
    threads = 256
    
    # Load saved model parameters
    model_path = input("Model file path (default: model_final): ") or "model_final"
    W1, b1, W2, b2 = load_model(model_path)
    
    print("Evaluating model on test set...")
    start_time = time.time()
    
    # Evaluate on full test set
    test_correct = 0
    test_total = 0
    test_batches = len(X_test) // batch_size
    confusion_matrix = np.zeros((output_dim, output_dim), dtype=int)
    
    for i in range(test_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        test_x = X_test[start_idx:end_idx].astype(np.float32)
        test_labels = y_test[start_idx:end_idx].astype(np.int32)
        
        # Forward pass
        test_out, _ = cu.mlp_forward(test_x, W1, b1, W2, b2, batch_size, input_dim, hidden_dim, output_dim, threads)
        test_probs = cu.softmax_batch(test_out, batch_size, output_dim, threads)
        
        # Calculate predictions and accuracy
        test_preds = np.argmax(test_probs, axis=1)
        test_correct += np.sum(test_preds == test_labels)
        test_total += batch_size
        
        # Update confusion matrix
        for j in range(len(test_labels)):
            confusion_matrix[test_labels[j]][test_preds[j]] += 1
    
    # Calculate final accuracy
    final_accuracy = test_correct / test_total
    evaluation_time = time.time() - start_time
    
    print(f"Test Accuracy: {final_accuracy:.4f}")
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(range(output_dim))
    plt.yticks(range(output_dim))
    
    # Add text annotations to the confusion matrix
    for i in range(output_dim):
        for j in range(output_dim):
            plt.text(j, i, confusion_matrix[i, j],
                     ha="center", va="center", color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black")
    
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Visualize some test examples
    plt.figure(figsize=(12, 8))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        
        # Select random sample from test set
        sample_idx = np.random.randint(0, len(X_test))
        img = X_test[sample_idx].reshape(28, 28)
        
        # Make prediction
        sample_x = X_test[sample_idx:sample_idx+1].astype(np.float32)
        out, _ = cu.mlp_forward(sample_x, W1, b1, W2, b2, 1, input_dim, hidden_dim, output_dim, threads)
        probs = cu.softmax_batch(out, 1, output_dim, threads)
        pred = np.argmax(probs)
        
        plt.imshow(img, cmap='gray')
        plt.title(f'Pred: {pred}, True: {y_test[sample_idx]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_samples.png')
    plt.show()

if __name__ == "__main__":
    main()