"""Test the v2 fine-tuned model on BOTH failure types."""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "models/halluc_model_v2"

print("Loading v2 fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Loaded.\n")

def predict(context, sentence):
    inputs = tokenizer(context, sentence, truncation=True,
                       max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    return {"clean": round(float(probs[0]), 3),
            "hallucinated": round(float(probs[1]), 3)}

print("=== NUMERIC CONTRADICTION (the thing we just trained for) ===")
ctx = "Refunds are processed within 30 days of purchase."
print("Faithful (30 days):    ", predict(ctx, "Customers can get a refund within 30 days."))
print("Contradiction (90 days):", predict(ctx, "Refunds are available for up to 90 days."))

print("\n=== BASELESS INFO (its original strength) ===")
ctx2 = "The museum is open from 9 AM to 5 PM on weekdays."
print("Faithful:              ", predict(ctx2, "The museum opens at 9 AM on weekdays."))
print("Baseless add-on:       ", predict(ctx2, "The museum offers free guided tours every hour."))

print("\n=== UNRELATED ===")
ctx3 = "Refunds are processed within 30 days of purchase."
print("Unrelated:             ", predict(ctx3, "The company ships internationally via FedEx."))