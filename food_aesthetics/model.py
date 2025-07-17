import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from datetime import datetime
import json

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class FoodAestheticsTrainer:
    def __init__(self, image_size=(224, 224)):
        """
        Initialize the Food Aesthetics Trainer.
        
        Args:
            image_size: Tuple of (width, height) for input images
        """
        self.image_size = image_size
        self.model = None
        self.history = None
        self.training_info = {}
        
    def load_and_prepare_data(self, images_dir):
        """
        Load and prepare training data from images directory.
        Expects images with 'good-' prefix for high aesthetic score (1)
        and 'bad-' prefix for low aesthetic score (0).
        
        Args:
            images_dir: Path to directory containing labeled images
            
        Returns:
            Tuple of (images, labels, filenames)
        """
        print("Loading and preparing training data...")
        
        images_path = Path(images_dir)
        images = []
        labels = []
        filenames = []
        
        # Get all image files
        image_files = [f for f in os.listdir(images_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in image_files:
            if filename.startswith('good-'):
                label = 1  # High aesthetic score
            elif filename.startswith('bad-'):
                label = 0  # Low aesthetic score
            else:
                continue  # Skip unlabeled images
            
            try:
                # Load and preprocess image
                img_path = images_path / filename
                img = Image.open(img_path).convert('RGB')
                img = img.resize(self.image_size)
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                
                images.append(img_array)
                labels.append(label)
                filenames.append(filename)
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"Loaded {len(images)} images:")
        print(f"  - High aesthetic (good): {sum(labels)} images")
        print(f"  - Low aesthetic (bad): {len(labels) - sum(labels)} images")
        
        return images, labels, filenames
    
    def create_model(self):
        """
        Create a CNN model for food aesthetics classification.
        This is a beginner-friendly architecture.
        
        Returns:
            Compiled Keras model
        """
        model = tf.keras.Sequential([
            # First convolutional block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                                 input_shape=(*self.image_size, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.BatchNormalization(),
            
            # Second convolutional block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.BatchNormalization(),
            
            # Third convolutional block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.BatchNormalization(),
            
            # Fourth convolutional block
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.BatchNormalization(),
            
            # Flatten and dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            # Output layer (binary classification)
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_data_augmentation(self):
        """
        Create data augmentation pipeline to improve model generalization.
        
        Returns:
            tf.keras.Sequential data augmentation pipeline
        """
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ])
    
    def train_model(self, images_dir, epochs=50, validation_split=0.2, 
                   batch_size=32, use_augmentation=True):
        """
        Train the food aesthetics model.
        
        Args:
            images_dir: Path to directory containing labeled images
            epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            batch_size: Batch size for training
            use_augmentation: Whether to use data augmentation
        """
        print("Starting model training...")
        
        # Load data
        images, labels, filenames = self.load_and_prepare_data(images_dir)
        
        if len(images) == 0:
            raise ValueError("No labeled images found! Make sure images have 'good-' or 'bad-' prefixes.")
        
        # Create model
        self.model = self.create_model()
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
        # Create data augmentation if requested
        if use_augmentation:
            data_augmentation = self.create_data_augmentation()
            # Apply augmentation to training data
            augmented_images = data_augmentation(images)
            images = np.concatenate([images, augmented_images])
            labels = np.concatenate([labels, labels])
        
        # Split data
        total_samples = len(images)
        indices = np.random.permutation(total_samples)
        val_samples = int(total_samples * validation_split)
        
        val_indices = indices[:val_samples]
        train_indices = indices[val_samples:]
        
        X_train, X_val = images[train_indices], images[val_indices]
        y_train, y_val = labels[train_indices], labels[val_indices]
        
        print(f"\nDataset split:")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Validation samples: {len(X_val)}")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        print("\nStarting training...")
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Store training info
        self.training_info = {
            'total_samples': total_samples,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'epochs_trained': len(self.history.history['loss']),
            'final_train_accuracy': self.history.history['accuracy'][-1],
            'final_val_accuracy': self.history.history['val_accuracy'][-1],
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"\nTraining completed!")
        print(f"Final training accuracy: {self.training_info['final_train_accuracy']:.3f}")
        print(f"Final validation accuracy: {self.training_info['final_val_accuracy']:.3f}")
        
        # Evaluate on validation set
        self.evaluate_model(X_val, y_val)
        
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model and show detailed metrics.
        
        Args:
            X_test: Test images
            y_test: Test labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        print("\nEvaluating model...")
        
        # Make predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        
        print(f"Test Accuracy: {accuracy:.3f}")
        print(f"Test Loss: {self.model.evaluate(X_test, y_test, verbose=0)[0]:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Bad Aesthetic', 'Good Aesthetic']))
        
    def plot_training_history(self):
        """
        Plot training history to visualize model performance.
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy plot
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss plot
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision plot
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall plot
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_model(self, model_path='food_aesthetics_model.h5'):
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path where to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        self.model.save(model_path)
        
        # Save training info
        info_path = model_path.replace('.h5', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(self.training_info, f, indent=2)
        
        print(f"Model saved to: {model_path}")
        print(f"Training info saved to: {info_path}")
        
    def load_model(self, model_path='food_aesthetics_model.h5'):
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        self.model = tf.keras.models.load_model(model_path)
        
        # Load training info if available
        info_path = model_path.replace('.h5', '_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.training_info = json.load(f)
        
        print(f"Model loaded from: {model_path}")


class FoodAesthetics:
    """
    Main class for food aesthetics inference.
    This replaces the original complex model with a simpler, trained version.
    """
    def __init__(self, model_path='food_aesthetics_model.h5'):
        """
        Initialize the Food Aesthetics scorer.
        
        Args:
            model_path: Path to the trained model
        """
        self.image_size = (224, 224)
        self.model = None
        self.model_path = model_path
        
        # Try to load the model
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"Warning: Model file '{model_path}' not found.")
            print("Please train a model first using the FoodAestheticsTrainer class.")
            print("Example: trainer = FoodAestheticsTrainer()")
            print("         trainer.train_model('images/')")
            print("         trainer.save_model()")
    
    def load_model(self, model_path):
        """Load the trained model."""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def _preprocess_image(self, image_path):
        """
        Preprocess an image for prediction.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Preprocessed image array
        """
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.image_size)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    
    def aesthetic_score(self, image_path):
        """
        Compute aesthetic score of an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Aesthetic score between 0 and 1 (higher is better)
        """
        if self.model is None:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        # Preprocess image
        processed_image = self._preprocess_image(image_path)
        
        # Make prediction
        prediction = self.model.predict(processed_image, verbose=0)[0][0]
        
        return float(prediction)
    
    def predict_batch(self, image_paths):
        """
        Predict aesthetic scores for multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of aesthetic scores
        """
        if self.model is None:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        scores = []
        for image_path in image_paths:
            try:
                score = self.aesthetic_score(image_path)
                scores.append(score)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                scores.append(0.0)
        
        return scores
    
    # Additional feature extraction methods (simplified versions)
    def brightness(self, path):
        """Calculate average brightness of an image."""
        img = cv.imread(str(path))
        if img is None:
            return 0
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        return hsv[:, :, 2].mean()
    
    def saturation(self, path):
        """Calculate average saturation of an image."""
        img = cv.imread(str(path))
        if img is None:
            return 0
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        return hsv[:, :, 1].mean()
    
    def contrast(self, path):
        """Calculate contrast (standard deviation of brightness)."""
        img = cv.imread(str(path))
        if img is None:
            return 0
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        return hsv[:, :, 2].std()


# Example usage and training script
if __name__ == '__main__':
    print("Food Aesthetics Model Training Example")
    print("=" * 40)
    
    # Initialize trainer
    trainer = FoodAestheticsTrainer()
    
    # Train model (adjust parameters as needed)
    print("Training model on labeled images...")
    trainer.train_model(
        images_dir='../images/',  # Path to your labeled images
        epochs=50,                # Number of training epochs
        validation_split=0.2,     # 20% for validation
        batch_size=16,            # Batch size (reduce if memory issues)
        use_augmentation=True     # Use data augmentation
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save the model
    trainer.save_model('food_aesthetics_model.h5')
    
    # Test the trained model
    print("\nTesting trained model...")
    fa = FoodAesthetics('food_aesthetics_model.h5')
    
    # Test on a sample image
    sample_image = '../images/good-1.jpeg'
    if os.path.exists(sample_image):
        score = fa.aesthetic_score(sample_image)
        print(f"Aesthetic score for {sample_image}: {score:.3f}")
