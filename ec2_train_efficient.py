import numpy as np
import os
import logging
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_images_from_folder(folder, max_images=100, img_size=(64, 64)):
    """
    Load images and their labels from a folder with memory constraints
    """
    images = []
    labels = []
    image_count = 0

    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if image_count >= max_images:
                    break

                img_path = os.path.join(class_path, filename)
                try:
                    # Open image, convert to grayscale, and resize
                    img = Image.open(img_path).convert('L')  # Grayscale reduces memory
                    img = img.resize(img_size)
                    
                    # Convert image to numpy array and flatten
                    img_array = np.array(img).flatten() / 255.0  # Normalize
                    images.append(img_array)
                    labels.append(class_name)
                    
                    image_count += 1
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")

        if image_count >= max_images:
            break
    
    return np.array(images), np.array(labels)

def create_data_loaders(data_path):
    """
    Load and split data into train, validation sets
    """
    # Load images from train folder with reduced size and count
    train_folder = os.path.join(data_path, 'train')
    X_train, y_train = load_images_from_folder(train_folder)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train_encoded, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val, label_encoder

def train_model(X_train, y_train, X_val, y_val):
    """
    Train a smaller neural network classifier
    """
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),  # Smaller network
        max_iter=50,  # Reduced iterations
        learning_rate_init=0.001,
        random_state=42,
        verbose=False
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_predictions = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    
    logger.info(f"Validation Accuracy: {val_accuracy}")
    logger.info("\nClassification Report:\n" + 
                classification_report(y_val, val_predictions))
    
    return clf

def main():
    # Hyperparameters
    data_path = 'dogImages'
    
    # Load and prepare data
    X_train, X_val, y_train, y_val, label_encoder = create_data_loaders(data_path)
    
    # Train the model
    logger.info("Starting Model Training")
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Save the model
    from joblib import dump
    os.makedirs('TrainedModels', exist_ok=True)
    dump(model, 'TrainedModels/model.joblib')
    dump(label_encoder, 'TrainedModels/label_encoder.joblib')
    
    logger.info('Model and label encoder saved')

if __name__ == "__main__":
    main()
