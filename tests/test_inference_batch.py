from pathlib import Path
from src.inference.predict import predict_image

def run_batch_inference(test_dir: str = "sample_tests"):
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"Test directory {test_dir} does not exist.")
        return
    
    supported_extensions = ['.jpg', '.jpeg', '.png', '.pdf']

    files = sorted(
        f for f in test_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions 
    )

    if not files:
        print(f"No supported files found in {test_dir}.")
        return
    print(f"Running batch inference on {len(files)} files in {test_dir}...\n")

    for file_path in files:
        try:
            predicted_class, confidence = predict_image(str(file_path))
            print("-" * 80)
            print(f"File: {file_path.name}")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
        except Exception as e:
            print(f"File: {file_path.name}")
            print(f"Error occurred while processing the file: {e}")
            print("-" * 80)


if __name__ == "__main__":
    run_batch_inference()