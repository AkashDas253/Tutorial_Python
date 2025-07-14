# PyTorch Cheatsheet

## 1. Installing PyTorch
- pip install torch torchvision torchaudio  # Install PyTorch and related libraries

## 2. Importing Libraries
- import torch  # Import PyTorch
- import torch.nn as nn  # Import neural network module
- import torch.optim as optim  # Import optimization module
- import torchvision.transforms as transforms  # Import transforms

## 3. Checking Device
- device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check for GPU availability

## 4. Creating Tensors
- tensor = torch.tensor(data)  # Create a tensor from data
- tensor = torch.zeros(size)  # Create a tensor filled with zeros
- tensor = torch.ones(size)  # Create a tensor filled with ones
- tensor = torch.randn(size)  # Create a tensor with random normal values
- tensor = torch.arange(start, end, step)  # Create a tensor with a range of values

## 5. Basic Tensor Operations
- tensor1 + tensor2  # Element-wise addition
- tensor1 * tensor2  # Element-wise multiplication
- torch.matmul(tensor1, tensor2)  # Matrix multiplication
- torch.sum(tensor)  # Sum of elements
- torch.mean(tensor)  # Mean of elements

## 6. Reshaping Tensors
- tensor.view(new_shape)  # Reshape tensor
- tensor.transpose(dim0, dim1)  # Transpose tensor
- tensor.unsqueeze(dim)  # Add dimension
- tensor.squeeze(dim)  # Remove dimension

## 7. Defining a Neural Network
- class SimpleNN(nn.Module):  # Define a simple neural network
  - def __init__(self):
    - super(SimpleNN, self).__init__()  # Initialize the base class
    - self.fc1 = nn.Linear(2, 2)  # First layer
    - self.fc2 = nn.Linear(2, 1)  # Second layer
  - def forward(self, x):  # Forward pass
    - x = torch.relu(self.fc1(x))  # Apply ReLU activation
    - x = self.fc2(x)  # Output layer
    - return x  # Return output

## 8. Creating a DataLoader
- from torch.utils.data import DataLoader, Dataset, TensorDataset  # Import DataLoader
- dataset = TensorDataset(X, y)  # Create a dataset
- dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Create a DataLoader

## 9. Training the Model
- model = SimpleNN().to(device)  # Instantiate and move model to device
- criterion = nn.MSELoss()  # Define loss function
- optimizer = optim.SGD(model.parameters(), lr=0.01)  # Define optimizer

- for epoch in range(num_epochs):  # Loop over epochs
  - optimizer.zero_grad()  # Zero gradients
  - outputs = model(inputs)  # Forward pass
  - loss = criterion(outputs, labels)  # Compute loss
  - loss.backward()  # Backward pass
  - optimizer.step()  # Update weights

## 10. Evaluating the Model
- with torch.no_grad():  # Disable gradient tracking
  - outputs = model(inputs)  # Forward pass
  - loss = criterion(outputs, labels)  # Compute loss
  - _, predicted = torch.max(outputs.data, 1)  # Get predicted labels

## 11. Making Predictions
- model.eval()  # Set model to evaluation mode
- with torch.no_grad():  # Disable gradient calculation
  - predictions = model(X_test.to(device))  # Make predictions

## 12. Saving and Loading Models
- torch.save(model.state_dict(), 'model.pth')  # Save model
- model.load_state_dict(torch.load('model.pth'))  # Load model
- model.eval()  # Set model to evaluation mode

## 13. Using Pre-trained Models
- from torchvision import models  # Import models
- model = models.resnet18(pretrained=True)  # Load pre-trained ResNet model
- model.fc = nn.Linear(in_features, num_classes)  # Modify last layer for new task
- model.eval()  # Set model to evaluation mode

## 14. Transfer Learning
- model = torchvision.models.resnet18(pretrained=True)  # Load pre-trained model
- for param in model.parameters():
  - param.requires_grad = False  # Freeze parameters
- model.fc = nn.Linear(in_features, num_classes)  # Modify last layer for new task

## 15. Using CUDA for GPU
- model.to(device)  # Move model to GPU
- tensor.to(device)  # Move tensor to GPU

## 16. Plotting with Matplotlib
- plt.plot(losses)  # Plot loss over epochs

## 17. Custom Dataset
- class MyDataset(Dataset):  # Define custom dataset
  - def __init__(self, data, labels):
    - self.data = data
    - self.labels = labels
  - def __len__(self):
    - return len(self.data)
  - def __getitem__(self, idx):
    - return self.data[idx], self.labels[idx]
