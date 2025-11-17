import gradio as gr
import numpy as np
import spaces
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import io

# Global variables to store model and tokenizer
model = None
tokenizer = None

def load_model(model_path):
    """Load the fine-tuned model and tokenizer from Hugging Face"""
    global model, tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Model loaded from {model_path}")

def draw_molecule(smiles):
    """Draw molecule structure from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=(400, 400))
        return img
    except Exception as e:
        print(f"Error drawing molecule: {e}")
        return None

@spaces.GPU
def predict(input_text):
        """Make prediction on the input text directly without creating a dataset"""
        if model is None or tokenizer is None:
            return "Error: Model not loaded"
        
        # Draw molecule structure
        mol_image = draw_molecule(input_text)

        model.to('cuda')
        # Tokenize input directly
        inputs = tokenizer(input_text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        
        # Move input tensors to GPU
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

        # Get model predictions
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()
        
        # Stable softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Get predicted label
        pred_label = np.argmax(probs, axis=1)[0]
        # Map prediction to label
        label_map = {0: "Unnatural", 1: "Natural"}
        pred_label_text = label_map[pred_label]
        
        # Format output
        result = f"Type: {pred_label_text}\n"
        natural_prob = probs[0][1] if pred_label == 1 else 1 - probs[0][0]
        result += f"Natural Product Probability: {natural_prob:.4f}\n"
        
        return mol_image, result



# Load model on initialization
load_model("shulik7/NP_SMILES_tokenized_PubChem_shard00_160k")  
# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, placeholder="Enter the SMILES here...", label="SMILES Input"),
    outputs=[
        gr.Image(label="Molecule Structure", type="pil"),
        gr.Textbox(lines=5, placeholder="Prediction results will appear here...", label="Prediction")
    ],
    title="Naturalness Prediction",
    description="Enter SMILES string to get the prediction from the fine-tuned ChemBERTa model.",
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()