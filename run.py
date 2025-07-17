# import dependencies
from food_aesthetics.model import FoodAesthetics
import argparse, os, time
import pandas as pd
from pathlib import Path

# 1. create a path function 
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

# 2. parametrize script 
parser = argparse.ArgumentParser(description='Score food images for aesthetic quality')
parser.add_argument('path',
    help='Path to images folder',
    type=dir_path
)
parser.add_argument('--model', default='food_aesthetics_model.h5',
    help='Path to trained model file (default: food_aesthetics_model.h5)'
)
parser.add_argument('--extensions', default='jpeg,jpg,png',
    help='Image extensions to process (default: jpeg,jpg,png)'
)
args = parser.parse_args()

path_dir = Path(args.path)
extensions = args.extensions.split(',')

# 3. Check if model exists
if not os.path.exists(args.model):
    print(f"❌ Error: Model file '{args.model}' not found!")
    print("Please train a model first using:")
    print("  python train.py")
    print("Or specify a different model path with --model")
    exit(1)

# 4. init model 
print(f"🤖 Loading model from: {args.model}")
try:
    fa = FoodAesthetics(args.model)
    print('✅ Model loaded successfully')
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

print('🖼️  Scoring the pictures...')

# 5. score images
images = os.listdir(path_dir) # load images
# Filter for specified extensions
valid_images = []
for image in images:
    for ext in extensions:
        if image.lower().endswith(f'.{ext.lower()}'):
            valid_images.append(image)
            break

if not valid_images:
    print(f"❌ No images found with extensions: {extensions}")
    exit(1)

print(f"📁 Found {len(valid_images)} images to process")

out = []
failed_images = []

for i, image in enumerate(valid_images, 1):
    try:
        image_path = path_dir / image
        aes = fa.aesthetic_score(image_path)
        out.append(aes)
        print(f"  [{i}/{len(valid_images)}] {image}: {aes:.3f}")
    except Exception as e:
        print(f"  [{i}/{len(valid_images)}] ❌ Error processing {image}: {e}")
        out.append(0.0)  # Default score for failed images
        failed_images.append(image)

print(f'✅ Scored {len(valid_images)} pictures.')

if failed_images:
    print(f"⚠️  Failed to process {len(failed_images)} images:")
    for img in failed_images:
        print(f"    - {img}")

# 6. export results
df = pd.DataFrame({
    "image": valid_images,
    "aesthetic_score": out
})

# Add statistics
avg_score = df['aesthetic_score'].mean()
max_score = df['aesthetic_score'].max()
min_score = df['aesthetic_score'].min()

print(f"\n📊 Results Summary:")
print(f"  Average score: {avg_score:.3f}")
print(f"  Highest score: {max_score:.3f}")
print(f"  Lowest score: {min_score:.3f}")

# Find best and worst images
best_image = df.loc[df['aesthetic_score'].idxmax()]
worst_image = df.loc[df['aesthetic_score'].idxmin()]

print(f"  Best image: {best_image['image']} ({best_image['aesthetic_score']:.3f})")
print(f"  Worst image: {worst_image['image']} ({worst_image['aesthetic_score']:.3f})")

# Save results
timestr = time.strftime("%Y%m%d-%H%M%S")
output_file = f'./output/aesthetic_scores-{timestr}.csv'
df.to_csv(output_file, index=False)

print(f'\n💾 Results exported to: {output_file}')
print('🎉 Processing completed successfully!')
