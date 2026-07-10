import os
import numpy as np
import keras
import matplotlib.pyplot as plt
from PIL import Image

# Path Configuration
MODELS_DIR = "models"
DATASET_DIR = r"dataset/processed/train/Wild_Fire"
OUTPUT_DIR = "static" # Save visualizations to static folder for web app access as well
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_FILES = {
    "S1_LR_Besar": "S1_LR_Besar.keras",
    "S2_LR_Kecil": "S2_LR_Kecil.keras",
    "S5_Epoch_Tinggi": "S5_Epoch_Tinggi.keras"
}

def load_and_preprocess_images(directory, target_size=(224, 224)):
    print(f"Loading images from {directory}...")
    images = []
    filenames = []
    
    # List and sort filenames to keep order
    files = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for filename in files:
        img_path = os.path.join(directory, filename)
        try:
            # Load and convert to RGB
            img = Image.open(img_path).convert('RGB')
            # Resize
            img_resized = img.resize(target_size)
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            
            images.append(img_array)
            filenames.append(filename)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            
    return np.array(images), filenames

def evaluate_models():
    # 1. Load images
    x_test, filenames = load_and_preprocess_images(DATASET_DIR)
    print(f"Loaded {len(x_test)} images with shape {x_test.shape}")
    
    if len(x_test) == 0:
        print("No images found for evaluation.")
        return
        
    results = {}
    
    # 2. Load models and predict
    for name, filename in MODEL_FILES.items():
        model_path = os.path.join(MODELS_DIR, filename)
        print(f"\n--- Evaluating Model: {name} ({filename}) ---")
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue
            
        try:
            model = keras.models.load_model(model_path)
            # Run prediction
            preds = model.predict(x_test, batch_size=8)
            # Sigmoid outputs are shape (N, 1), flatten to (N,)
            scores = preds.flatten()
            
            # Since all images in the folder are Wild_Fire, target label is 1
            # Recall = percentage of predictions >= 0.5
            correct_classifications = np.sum(scores >= 0.5)
            recall = (correct_classifications / len(scores)) * 100
            avg_confidence = np.mean(scores) * 100
            
            results[name] = {
                "scores": scores,
                "recall": recall,
                "avg_confidence": avg_confidence,
                "correct": correct_classifications,
                "total": len(scores)
            }
            
            print(f"Average Wildfire Confidence: {avg_confidence:.2f}%")
            print(f"Correctly Classified (Confidence >= 50%): {correct_classifications}/{len(scores)} ({recall:.2f}%)")
            
        except Exception as e:
            print(f"Error loading/evaluating model {name}: {e}")
            
    if not results:
        print("No models were successfully evaluated.")
        return
        
    # 3. Create Evaluation Comparison Plot
    print("\nGenerating evaluation comparison plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    model_names = list(results.keys())
    avg_confidences = [results[m]["avg_confidence"] for m in model_names]
    recalls = [results[m]["recall"] for m in model_names]
    
    # Plot 1: Average Confidence
    bars1 = ax1.bar(model_names, avg_confidences, color=['#FF5E36', '#FFA259', '#FFC107'], edgecolor='black', width=0.5)
    ax1.set_ylabel("Average Wildfire Confidence (%)")
    ax1.set_title("Average Prediction Confidence")
    ax1.set_ylim(0, 105)
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    # Plot 2: Recall / Sensitivity (Detected Wildfires)
    bars2 = ax2.bar(model_names, recalls, color=['#10B981', '#34D399', '#6EE7B7'], edgecolor='black', width=0.5)
    ax2.set_ylabel("Recall / Detected Wildfires (%)")
    ax2.set_title("Recall / Sensitivity (Detected as Fire >= 50%)")
    ax2.set_ylim(0, 105)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    plt.suptitle("Wildfire CNN Models Performance Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    eval_chart_path = os.path.join(OUTPUT_DIR, "evaluation_results.png")
    plt.savefig(eval_chart_path, dpi=150)
    plt.close()
    print(f"Saved evaluation results chart to: {eval_chart_path}")
    
    # 4. Create Predictions Grid (Sample Images)
    print("Generating predictions grid for samples...")
    # Select up to 16 sample images
    num_samples = min(16, len(filenames))
    sample_indices = np.random.choice(len(filenames), num_samples, replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    for i, idx in enumerate(sample_indices):
        img_array = x_test[idx]
        filename = filenames[idx]
        
        # Display image
        axes[i].imshow(img_array)
        axes[i].axis('off')
        
        title_text = f"File: {filename}\n"
        for name in model_names:
            score = results[name]["scores"][idx] * 100
            title_text += f"{name}: {score:.1f}%\n"
            
        axes[i].set_title(title_text, fontsize=9, ha='center')
        
    plt.tight_layout()
    grid_chart_path = os.path.join(OUTPUT_DIR, "predictions_grid.png")
    plt.savefig(grid_chart_path, dpi=120)
    plt.close()
    print(f"Saved prediction grid to: {grid_chart_path}")
    
    print("\nEvaluation successfully completed!")

if __name__ == "__main__":
    evaluate_models()
