I'll help organize these detailed notes into a clear, comprehensive format.



# Project Overview
## Workspace Structure
- **configs/**: Configuration files for hyperparameters and settings
- **data/**: Scripts for processing datasets
- **factory/**: Factory functions for creating datasets, models, optimizers, and loss functions
- **URN/**: Core neural network models and utilities
- **utils/**: Utility functions
- **train.py**: Main script for training the neural network
- **eval.py**: Script for evaluating the trained model
- **step.py**: Functions for running predictions on data batches

## Key Components
### 1. Data Loading
- `data/my_bio_dataset.py`: Defines `BioBaseDataset` and `BioDataset` classes
- `factory/bio_data_factory.py`: Contains functions like `get_data`, `get_dataloader`, and `get_dataset_by_name`

### 2. Model Training
- `train.py`: Initializes dataset, model, optimizer, and starts training loop
- `step.py`: Contains `predict` function for running predictions during training

### 3. Model Evaluation
- `eval.py`: Evaluates the trained model on test data

## Running the Project
1. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**: 
   - Ensure correct format
   - Update configuration file paths if needed

3. **Train Model**:
   ```sh
   python train.py
   ```

4. **Evaluate Model**:
   ```sh
   python eval.py
   ```

5. **Testing**:
   ```sh
   pytest tests/
   ```

# Machine Learning Concepts
## Training Terminology
### Epochs and Iterations
- **Epoch**: Complete pass through entire training dataset
  - Model sees every training sample once
  - Higher epochs (e.g., 2000) may improve performance but risk overfitting
  
- **Iteration**: Processing of one batch
  - 'n' batches require 'n' iterations
  - Batch can be whole sample or divided parts

## Loss Functions
### Overview
- Mathematical functions measuring prediction accuracy
- Compare model predictions to actual data labels
- Quantify error/difference between predictions and true values
- Provide feedback for model parameter adjustment

### Training Process
1. **Forward Pass**: 
   - Model makes predictions on data batch
   - Calculates loss by comparing to true values

2. **Backpropagation**:
   - Uses loss value to compute gradients
   - Determines parameter adjustment amounts

3. **Parameter Update**:
   - Optimizer updates model parameters
   - Aims to minimize loss

### Types of Loss Functions
1. **Classification Loss**
   - Binary Cross-Entropy (BCE)
   - Used for binary classification tasks
   - Penalizes incorrect predictions more as model confidence increases

2. **Segmentation Loss**
   - Includes BCE for pixel classification
   - Used for object boundary prediction in images
   - **Dice Similarity Coefficient (DSC)**:
     - Measures overlap between predicted and ground truth segmentation
     - Range: 0 (no overlap) to 1 (perfect overlap)
     - Formula: `Dice = (2 × |A∩B|) / (|A|+|B|)`
     - Dice Loss = 1 - Dice
     - Particularly effective for imbalanced datasets

## Optimizers
### Purpose
- Adjust model parameters to minimize loss function
- Guide model learning process

### Types
1. **Stochastic Gradient Descent (SGD)**
   - Simple but potentially slow
   - Moves parameters in negative gradient direction

2. **Momentum**
   - Adds momentum term
   - Speeds up convergence

3. **Adam**
   - Combines Momentum and RMSProp
   - Adapts learning rates based on recent gradients
   - Uses first and second moments of gradients

4. **RMSProp**
   - Adjusts learning rate based on recent gradient magnitudes

### Key Hyperparameters
- **Learning Rate (lr)**: Controls parameter update size
- **b1 and b2** (Adam specific): Decay rates for gradient moments