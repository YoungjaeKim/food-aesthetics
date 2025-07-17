#!/usr/bin/env python3
"""
Food Aesthetics Model Training Script

This script trains a deep learning model to assess food image aesthetics.
The model learns to distinguish between good and bad food presentation.

Usage:
    python train.py

Make sure your images are organized as:
    images/good-1.jpeg, images/good-2.jpeg, ... (high aesthetic score)
    images/bad-1.jpeg, images/bad-2.jpeg, ... (low aesthetic score)
"""

import os
import sys
from pathlib import Path
import argparse

# Add the food_aesthetics module to the path
sys.path.append(str(Path(__file__).parent))

from food_aesthetics.model import FoodAestheticsTrainer, FoodAesthetics

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Food Aesthetics Model')
    parser.add_argument('--images-dir', default='images/', 
                       help='Directory containing labeled images (default: images/)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Fraction of data for validation (default: 0.2)')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--output-model', default='food_aesthetics_model.h5',
                       help='Output model filename (default: food_aesthetics_model.h5)')
    
    args = parser.parse_args()
    
    print("🍔 Food Aesthetics Model Training")
    print("=" * 50)
    
    # Check if images directory exists
    if not os.path.exists(args.images_dir):
        print(f"❌ Error: Images directory '{args.images_dir}' not found!")
        print("Please make sure your images are organized as:")
        print("  images/good-1.jpeg, images/good-2.jpeg, ... (high aesthetic)")
        print("  images/bad-1.jpeg, images/bad-2.jpeg, ... (low aesthetic)")
        return
    
    # Check for labeled images
    images_path = Path(args.images_dir)
    good_images = list(images_path.glob('good-*'))
    bad_images = list(images_path.glob('bad-*'))
    
    if len(good_images) == 0 or len(bad_images) == 0:
        print(f"❌ Error: Not enough labeled images found!")
        print(f"Found {len(good_images)} 'good-' images and {len(bad_images)} 'bad-' images")
        print("Please make sure your images have proper prefixes:")
        print("  good-1.jpeg, good-2.jpeg, ... for high aesthetic images")
        print("  bad-1.jpeg, bad-2.jpeg, ... for low aesthetic images")
        return
    
    print(f"📁 Found {len(good_images)} good images and {len(bad_images)} bad images")
    
    # Initialize trainer
    trainer = FoodAestheticsTrainer()
    
    # Train the model
    try:
        print("\n🚀 Starting training...")
        trainer.train_model(
            images_dir=args.images_dir,
            epochs=args.epochs,
            validation_split=args.validation_split,
            batch_size=args.batch_size,
            use_augmentation=not args.no_augmentation
        )
        
        # Save the model
        print(f"\n💾 Saving model to '{args.output_model}'...")
        trainer.save_model(args.output_model)
        
        # Plot training history
        print("\n📊 Generating training plots...")
        trainer.plot_training_history()
        
        # Test the trained model
        print("\n🧪 Testing trained model...")
        fa = FoodAesthetics(args.output_model)
        
        # Test on sample images
        test_images = good_images[:2] + bad_images[:2]
        for img_path in test_images:
            try:
                score = fa.aesthetic_score(img_path)
                print(f"  {img_path.name}: {score:.3f}")
            except Exception as e:
                print(f"  Error testing {img_path.name}: {e}")
        
        print("\n✅ Training completed successfully!")
        print(f"Model saved as: {args.output_model}")
        print(f"Training plots saved as: training_history.png")
        print("\nYou can now use the model with:")
        print(f"  python run.py path/to/your/images")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("Please check your data and try again.")

if __name__ == '__main__':
    main() 