import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm import tqdm

from dataset import MultiCancerDataset # Import our custom dataset

# --- 1. Configuration ---
DATA_DIR = "data/multicancer"
MODEL_SAVE_PATH = "src/unirag_cnn.pth" # Where to save the fine-tuned model
NUM_CLASSES = 3      # brain_tumor, breast_malignant, kidney_tumor
BATCH_SIZE = 16      # How many images to train on at once
NUM_EPOCHS = 10      # How many times to loop over the data (10 is a good start)
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2     # Use 20% of data for validation

def train_model():
    """
    Main function to load data, fine-tune the model, and save the weights.
    """
    
    # --- 2. Setup (Device and Model) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Model Training on {device} ---")

    # Load the pre-trained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # --- The "Surgical" Part ---
    # Freeze all layers except the final one
    for param in model.parameters():
        param.requires_grad = False
        
    # Get the number of input features for the final layer
    num_ftrs = model.fc.in_features
    
    # Replace the final layer (the "head")
    # nn.Linear(num_ftrs, NUM_CLASSES) creates a new, untrained
    # fully-connected layer with the correct number of outputs (3).
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # Move the model to the GPU *before* constructing the optimizer
    model = model.to(device)

    # --- 3. Load and Split Data ---
    full_dataset = MultiCancerDataset(root_dir=DATA_DIR)
    
    # Calculate split sizes
    test_size = int(len(full_dataset) * TEST_SPLIT)
    train_size = len(full_dataset) - test_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Data Loaded: {len(train_dataset)} train / {len(val_dataset)} validation samples.")

    # --- 4. Define Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss() # Standard for multi-class classification
    
    # Only optimize the parameters of the new final layer
    # This is the essence of "fine-tuning"
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # --- 5. The Training Loop ---
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        
        # --- Training Phase ---
        model.train() # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass + optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad(): # No gradients needed for validation
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

    # --- 6. Save the Model ---
    print("\n--- Training Complete ---")
    print(f"Saving fine-tuned model to {MODEL_SAVE_PATH}")
    
    # We save the "state_dict", not the whole model. This is safer.
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved successfully.")


if __name__ == "__main__":
    train_model()