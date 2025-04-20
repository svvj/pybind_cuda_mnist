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

def save_model(W1, b1, W2, b2, filename_prefix='model_final'):
    """Save the model parameters to disk"""
    np.save(f'{filename_prefix}_W1.npy', W1)
    np.save(f'{filename_prefix}_b1.npy', b1)
    np.save(f'{filename_prefix}_W2.npy', W2)
    np.save(f'{filename_prefix}_b2.npy', b2)
    print(f"Model saved to '{filename_prefix}_*.npy' files.")

def main():
    # Load MNIST dataset
    print("Loading the MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    
    # Data preprocessing
    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    
    # Split the dataset into training and testing sets
    # 90% training, 10% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Set Model parameters
    batch_size = 128
    input_dim = 784  # 28x28 Image
    hidden_dim = 128
    output_dim = 10  # 0-9 classes
    threads = 256
    
    # Hyperparameters
    num_epochs = 20
    learning_rate = 0.01
    
    # Initialize weights and biases
    W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)  # He initialization
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(output_dim, dtype=np.float32)
    
    # Records for loss and accuracy
    losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Indexes for mini batch
    num_samples = X_train.shape[0]
    num_batches = num_samples // batch_size
    
    print(f"Training Start: {num_samples} samples, {num_batches} batches")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        
        # Shuffle data at each epoch
        indices = np.random.permutation(num_samples)
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_indices = indices[batch_start:batch_start + batch_size]
            
            # Current batch data
            x = X_train[batch_indices].astype(np.float32)
            labels = y_train[batch_indices].astype(np.int32)
            
            # Forward pass
            hidden = np.zeros((batch_size, hidden_dim), dtype=np.float32)
            out = np.zeros((batch_size, output_dim), dtype=np.float32)
            out, hidden = cu.mlp_forward(x, W1, b1, W2, b2, batch_size, input_dim, hidden_dim, output_dim, threads)
            
            # Softmax and loss
            probs = cu.softmax_batch(out, batch_size, output_dim, threads)
            loss = cu.cross_entropy_loss(probs, labels, batch_size, output_dim, threads)
            epoch_loss += float(loss)
            
            # Calculate accuracy (during training)
            predictions = np.argmax(probs, axis=1)
            correct_predictions += np.sum(predictions == labels)
            
            # Backward pass
            d_out = cu.softmax_cross_entropy_backward(probs, labels, batch_size, output_dim, threads)
            
            # Gradients buffer
            dW1 = np.zeros((input_dim, hidden_dim), dtype=np.float32)
            db1 = np.zeros((hidden_dim,), dtype=np.float32)
            dW2 = np.zeros((hidden_dim, output_dim), dtype=np.float32)
            db2 = np.zeros((output_dim,), dtype=np.float32)
            
            # Backward
            cu.mlp_backward(x, hidden, d_out, W2, dW1, db1, dW2, db2, batch_size, input_dim, hidden_dim, output_dim, threads)
            
            # SGD Update with CUDA
            cu.sgd(W1, dW1, learning_rate, input_dim * hidden_dim, threads)
            cu.sgd(b1, db1, learning_rate, hidden_dim, threads)
            cu.sgd(W2, dW2, learning_rate, hidden_dim * output_dim, threads)
            cu.sgd(b2, db2, learning_rate, output_dim, threads)
        
        # Average epoch loss
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Training accuracy
        train_accuracy = correct_predictions / (num_batches * batch_size)
        train_accuracies.append(train_accuracy)
        
        # Evaluate on test set
        if (epoch + 1) % 2 == 0:  # Test every 2 epochs
            test_correct = 0
            test_total = 0
            
            # Process test data in batches
            test_batches = len(X_test) // batch_size
            
            for i in range(test_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                test_x = X_test[start_idx:end_idx].astype(np.float32)
                test_labels = y_test[start_idx:end_idx].astype(np.int32)
                
                # Forward pass
                test_out, _ = cu.mlp_forward(test_x, W1, b1, W2, b2, batch_size, input_dim, hidden_dim, output_dim, threads)
                test_probs = cu.softmax_batch(test_out, batch_size, output_dim, threads)
                
                # Calculate accuracy
                test_preds = np.argmax(test_probs, axis=1)
                test_correct += np.sum(test_preds == test_labels)
                test_total += batch_size
            
            test_accuracy = test_correct / test_total
            test_accuracies.append(test_accuracy)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                save_model(W1, b1, W2, b2, f"model_epoch_{epoch+1}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Train Acc: {train_accuracy:.4f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed! Total time: {total_time:.2f} seconds")
    
    # Save final model
    save_model(W1, b1, W2, b2, "model_final")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train')
    x_vals = np.arange(0, num_epochs, 2)
    plt.plot(x_vals, test_accuracies, label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    main()