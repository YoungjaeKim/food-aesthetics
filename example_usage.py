#!/usr/bin/env python3
"""
Example script showing how to use the trained food aesthetics model.

This script demonstrates:
1. Loading a trained model
2. Scoring individual images
3. Scoring multiple images
4. Extracting additional features
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Add the food_aesthetics module to the path
sys.path.append(str(Path(__file__).parent))

from food_aesthetics.model import FoodAesthetics

def main():
    print("🍽️ Food Aesthetics Model - Example Usage")
    print("=" * 50)
    
    # Check if model exists
    model_path = 'food_aesthetics_model.h5'
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found!")
        print("Please train a model first using: python train.py")
        return
    
    # Load the trained model
    print(f"📂 Loading model: {model_path}")
    fa = FoodAesthetics(model_path)
    
    # Example 1: Score a single image
    print("\n1️⃣ Scoring a single image:")
    sample_images = ['images/good-1.jpeg', 'images/bad-1.jpeg']
    
    for img_path in sample_images:
        if os.path.exists(img_path):
            try:
                score = fa.aesthetic_score(img_path)
                print(f"  📸 {img_path}: {score:.3f}")
            except Exception as e:
                print(f"  ❌ Error with {img_path}: {e}")
    
    # Example 2: Score multiple images
    print("\n2️⃣ Scoring multiple images:")
    images_dir = Path('images')
    if images_dir.exists():
        # Get all image files
        image_files = []
        for ext in ['*.jpeg', '*.jpg', '*.png']:
            image_files.extend(images_dir.glob(ext))
        
        if image_files:
            # Score first few images
            test_images = image_files[:5]  # First 5 images
            image_paths = [str(img) for img in test_images]
            
            try:
                scores = fa.predict_batch(image_paths)
                
                print("  Results:")
                for img_path, score in zip(image_paths, scores):
                    print(f"    {Path(img_path).name}: {score:.3f}")
                
                # Create a simple report
                df = pd.DataFrame({
                    'image': [Path(p).name for p in image_paths],
                    'aesthetic_score': scores
                })
                
                print(f"\n  📊 Summary:")
                print(f"    Average score: {df['aesthetic_score'].mean():.3f}")
                print(f"    Best image: {df.loc[df['aesthetic_score'].idxmax(), 'image']}")
                print(f"    Worst image: {df.loc[df['aesthetic_score'].idxmin(), 'image']}")
                
            except Exception as e:
                print(f"  ❌ Error scoring batch: {e}")
    
    # Example 3: Extract additional features
    print("\n3️⃣ Extracting additional features:")
    sample_image = 'images/good-1.jpeg'
    if os.path.exists(sample_image):
        try:
            print(f"  📸 Analyzing: {sample_image}")
            
            # Extract various features
            aesthetic_score = fa.aesthetic_score(sample_image)
            brightness = fa.brightness(sample_image)
            saturation = fa.saturation(sample_image)
            contrast = fa.contrast(sample_image)
            
            print(f"    Aesthetic Score: {aesthetic_score:.3f}")
            print(f"    Brightness: {brightness:.1f}")
            print(f"    Saturation: {saturation:.1f}")
            print(f"    Contrast: {contrast:.1f}")
            
        except Exception as e:
            print(f"  ❌ Error extracting features: {e}")
    
    # Example 4: Custom scoring function
    print("\n4️⃣ Custom scoring with interpretation:")
    
    def interpret_score(score):
        """Interpret aesthetic score with descriptive labels."""
        if score >= 0.8:
            return "Excellent presentation! 🌟"
        elif score >= 0.6:
            return "Good aesthetic quality 👍"
        elif score >= 0.4:
            return "Average presentation 😐"
        elif score >= 0.2:
            return "Below average 👎"
        else:
            return "Poor presentation 💔"
    
    # Test with sample images
    test_images = ['images/good-1.jpeg', 'images/bad-1.jpeg']
    
    for img_path in test_images:
        if os.path.exists(img_path):
            try:
                score = fa.aesthetic_score(img_path)
                interpretation = interpret_score(score)
                print(f"  📸 {Path(img_path).name}: {score:.3f} - {interpretation}")
            except Exception as e:
                print(f"  ❌ Error with {img_path}: {e}")
    
    print("\n✅ Example usage completed!")
    print("\nNext steps:")
    print("  • Train your own model: python train.py")
    print("  • Score your images: python run.py /path/to/images")
    print("  • Check the training guide: TRAINING_GUIDE.md")

if __name__ == '__main__':
    main() 