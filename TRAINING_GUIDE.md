# Food Aesthetics Model Training Guide

This guide will help you train your own food aesthetics model from scratch using deep learning. The model learns to distinguish between good and bad food presentation based on your labeled image examples.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
Your images should be organized with prefixes indicating their aesthetic quality:
- `good-1.jpeg`, `good-2.jpeg`, ... for high aesthetic score images
- `bad-1.jpeg`, `bad-2.jpeg`, ... for low aesthetic score images

```
images/
├── good-1.jpeg    # High aesthetic score
├── good-2.jpeg    # High aesthetic score
├── good-3.jpeg    # High aesthetic score
├── ...
├── bad-1.jpeg     # Low aesthetic score
├── bad-2.jpeg     # Low aesthetic score
├── bad-3.jpeg     # Low aesthetic score
└── ...
```

### 3. Train the Model
```bash
python train.py
```

This will:
- Load your labeled images
- Train a neural network model
- Save the trained model as `food_aesthetics_model.h5`
- Generate training plots in `training_history.png`

### 4. Use the Trained Model
```bash
python run.py /path/to/your/images
```

## 📊 Understanding the Model

### Architecture
The model uses a Convolutional Neural Network (CNN) with:
- 4 convolutional layers for feature extraction
- Batch normalization for stable training
- Dropout layers to prevent overfitting
- Dense layers for final classification

### Training Process
1. **Data Loading**: Images are loaded and resized to 224x224 pixels
2. **Data Augmentation**: Random transformations (rotation, flip, zoom) to increase diversity
3. **Training**: The model learns to distinguish good vs bad aesthetics
4. **Validation**: Performance is monitored on unseen data
5. **Evaluation**: Final metrics are computed and displayed

## 🎯 Advanced Training Options

### Custom Training Parameters
```bash
python train.py --help
```

Options:
- `--epochs 100`: Train for more epochs (default: 50)
- `--batch-size 32`: Increase batch size if you have more GPU memory
- `--validation-split 0.3`: Use 30% of data for validation
- `--no-augmentation`: Disable data augmentation
- `--output-model my_model.h5`: Custom model filename

### Example: Extended Training
```bash
python train.py --epochs 100 --batch-size 32 --validation-split 0.25
```

## 📈 Monitoring Training Progress

The training script will show:
- **Training/Validation Loss**: Should decrease over time
- **Training/Validation Accuracy**: Should increase over time
- **Precision/Recall**: Quality metrics for classification

Example output:
```
Epoch 45/50
4/4 [==============================] - 2s 486ms/step - loss: 0.3421 - accuracy: 0.8750 - precision: 0.8889 - recall: 0.8571 - val_loss: 0.4123 - val_accuracy: 0.8333 - val_precision: 0.8000 - val_recall: 0.8889
```

## 🔍 Evaluating Your Model

### Understanding Metrics
- **Accuracy**: Overall correctness (higher is better)
- **Precision**: Of predicted "good" images, how many were actually good
- **Recall**: Of actual "good" images, how many were correctly identified
- **Loss**: Training error (lower is better)

### Training Plots
After training, check `training_history.png` for:
- **Accuracy Plot**: Should show increasing accuracy over epochs
- **Loss Plot**: Should show decreasing loss over epochs
- **Overfitting Signs**: Large gap between training and validation curves

## 🛠️ Troubleshooting

### Common Issues

**1. "No labeled images found"**
- Ensure images have correct prefixes: `good-` and `bad-`
- Check image formats (JPEG, PNG supported)

**2. "Memory error" during training**
- Reduce batch size: `--batch-size 8`
- Reduce image size (edit `image_size` in trainer)

**3. "Model accuracy is low"**
- Collect more training data
- Ensure good variety in your examples
- Check if your labels are consistent

**4. "Validation loss increases"**
- Model might be overfitting
- Try `--no-augmentation` or reduce epochs
- Add more validation data

### Performance Tips

**For Better Results:**
1. **More Data**: Aim for 50+ examples of each class
2. **Balanced Dataset**: Equal numbers of good/bad images
3. **Diverse Examples**: Various lighting, angles, food types
4. **Consistent Labeling**: Be consistent in what you consider "good" vs "bad"

**For Faster Training:**
1. Use GPU if available (automatically detected)
2. Increase batch size if you have sufficient memory
3. Reduce image size for faster processing

## 📋 Example Workflow

### Complete Training Session
```bash
# 1. Prepare data (manually organize images)
ls images/
# good-1.jpeg  good-2.jpeg  ...  bad-1.jpeg  bad-2.jpeg  ...

# 2. Train model
python train.py --epochs 50 --batch-size 16

# 3. Test on new images
python run.py /path/to/test/images

# 4. Review results
cat output/aesthetic_scores-*.csv
```

### Iterative Improvement
```bash
# Train initial model
python train.py --epochs 30

# Test and review results
python run.py test_images/

# If unsatisfied, add more training data and retrain
python train.py --epochs 50 --output-model improved_model.h5

# Use improved model
python run.py test_images/ --model improved_model.h5
```

## 🧠 Understanding Machine Learning Concepts

### For Beginners

**What is the model learning?**
- The model learns patterns in images that distinguish good from bad food presentation
- It analyzes colors, shapes, composition, and other visual features
- Each "epoch" is one complete pass through all training data

**How does validation work?**
- 20% of your data is held back for testing
- The model never sees this data during training
- Validation performance tells you how well the model generalizes

**What is overfitting?**
- When the model memorizes training data instead of learning patterns
- Signs: Training accuracy high, validation accuracy low
- Solutions: More data, data augmentation, early stopping

### Key Terms
- **Epoch**: One complete pass through the training data
- **Batch Size**: Number of images processed at once
- **Learning Rate**: How fast the model learns (automatically managed)
- **Dropout**: Randomly ignore some neurons to prevent overfitting
- **Augmentation**: Create variations of images (rotation, zoom, etc.)

## 🔄 Model Maintenance

### Retraining Your Model
As you collect more data:
```bash
# Add new images to your dataset
cp new_good_images/* images/
cp new_bad_images/* images/

# Retrain with more data
python train.py --epochs 60 --output-model updated_model.h5
```

### Model Versioning
Keep track of different model versions:
```bash
# Save models with descriptive names
python train.py --output-model model_v1_baseline.h5
python train.py --output-model model_v2_more_data.h5

# Test different models
python run.py test_images/ --model model_v1_baseline.h5
python run.py test_images/ --model model_v2_more_data.h5
```

## 📞 Getting Help

If you encounter issues:
1. Check the error message carefully
2. Verify your data organization
3. Try with a smaller dataset first
4. Reduce batch size if memory issues
5. Check the troubleshooting section above

Remember: Machine learning is iterative. Don't expect perfect results on the first try!

## 🎉 Success Tips

1. **Start Small**: Begin with 20-30 images per class
2. **Be Patient**: Training takes time, especially on CPU
3. **Monitor Progress**: Watch the training plots and metrics
4. **Iterate**: Improve your dataset based on results
5. **Document**: Keep notes on what works and what doesn't

Good luck with your food aesthetics model training! 🍽️✨ 