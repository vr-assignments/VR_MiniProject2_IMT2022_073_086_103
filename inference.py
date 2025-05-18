import argparse
import csv
import os
from PIL import Image
import torch
from transformers import BlipForQuestionAnswering, BlipProcessor
from peft import PeftModel

BASE_MODEL = "Salesforce/blip-vqa-base"
ADAPTER_REPO = "ananthakk/blip_fine_tuned"  
OUTPUT_PATH = "results.csv"


def load_model(device):
    """
    Load the BLIP VQA base model and PEFT adapter from Hugging Face.
    """
    processor = BlipProcessor.from_pretrained(BASE_MODEL)
    base_model = BlipForQuestionAnswering.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
    model.to(device)
    model.eval()
    return processor, model


def predict(processor, model, image_path, question, device, max_length=16, num_beams=5):
    """
    Run VQA inference on a single image/question pair.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=max_length, num_beams=num_beams)
    answer = processor.decode(output_ids[0], skip_special_tokens=True)
    return answer


def main():
    parser = argparse.ArgumentParser(description="BLIP VQA inference with LoRA adapter from HF Hub")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV file with columns: image_name,question,answer")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_model(device)

    # Read input CSV and collect predictions
    results = []
    with open(args.csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row.get("image_name")
            question = row.get("question")
            ground_truth = row.get("answer", "")
            image_path = os.path.join(args.image_dir, filename)
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue

            prediction = predict(processor, model, image_path, question, device)
            results.append({
                "image_name": filename,
                "question": question,
                "answer": ground_truth,
                "generated_answer": prediction
            })

    # Save predictions to fixed output path
    fieldnames = ["image_name", "question", "answer", "generated_answer"]
    with open(OUTPUT_PATH, "w", newline='', encoding='utf-8') as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(item)

    print(f"Inference completed. Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
