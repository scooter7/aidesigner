import os
import streamlit as st
import torch
import requests
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from collections import Counter

# Get your OpenAI API key from Streamlit secrets.
API_KEY = st.secrets["openai"]["api_key"]

# Set up device for inference.
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Load CLIP Model & Processor
# ---------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define a list of adjectives that might describe a design style.
ADJECTIVES = [
    "modern", "minimalist", "vibrant", "elegant", "bold",
    "retro", "futuristic", "organic", "playful", "geometric",
    "sophisticated", "abstract", "colorful", "subtle", "dynamic",
    "classic", "innovative", "textured", "sleek", "artistic"
]

def get_adjective_embeddings():
    """
    Pre-compute CLIP text embeddings for the adjectives.
    """
    inputs = clip_processor(text=ADJECTIVES, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeds = clip_model.get_text_features(**inputs)
    # Normalize the embeddings.
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return text_embeds

# Pre-compute the adjective embeddings.
text_embeddings = get_adjective_embeddings()

def image_to_adjectives(image_path, top_k=3):
    """
    Given an image path, use CLIP to compute its embedding and return the top_k adjectives
    (from our predefined list) that best match the image.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embedding = clip_model.get_image_features(**inputs)
    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
    
    # Compute cosine similarity between the image embedding and each adjective embedding.
    similarity = (image_embedding @ text_embeddings.T).squeeze(0)
    _, indices = similarity.topk(top_k)
    selected = [ADJECTIVES[idx] for idx in indices.cpu().numpy()]
    return selected

def load_training_images():
    """
    Loads all images from the local 'images' folder.
    Assumes the folder is part of your repository.
    """
    images_dir = "images"
    image_files = []
    if os.path.exists(images_dir):
        for filename in os.listdir(images_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                image_files.append(os.path.join(images_dir, filename))
    return image_files

def generate_style_context_from_images(image_files):
    """
    For each image, extract descriptive adjectives using CLIP and then
    aggregate the results into a combined style context string.
    """
    all_adjectives = []
    for image_path in image_files:
        try:
            adjectives_list = image_to_adjectives(image_path, top_k=3)
            all_adjectives.extend(adjectives_list)
        except Exception as e:
            st.write(f"Error processing {image_path}: {e}")
    if not all_adjectives:
        return ""
    # Count frequency of each adjective.
    counter = Counter(all_adjectives)
    # Select the 5 most common adjectives.
    most_common = counter.most_common(5)
    style_context = ", ".join([adj for adj, _ in most_common])
    return style_context

def display_training_images(image_files):
    """
    Displays the training images in the sidebar.
    """
    st.sidebar.header("Training Images (Design Portfolio)")
    for image_path in image_files:
        try:
            img = Image.open(image_path)
            st.sidebar.image(img, caption=os.path.basename(image_path), use_container_width=True)
        except Exception as e:
            st.sidebar.write(f"Error loading {image_path}: {e}")

def generate_design_image_openai(user_prompt, style_context):
    """
    Combines the user's design prompt with the extracted style context and
    generates a new design image using OpenAI's image generation endpoint.
    This function calls the API directly via requests.
    """
    combined_prompt = f"Design an image with the following style characteristics: {style_context}. {user_prompt}"
    
    headers = {
         "Authorization": f"Bearer {API_KEY}",
         "Content-Type": "application/json"
    }
    data = {
         "prompt": combined_prompt,
         "n": 1,
         "size": "512x512"
    }
    try:
         response = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=data)
         response.raise_for_status()
         result = response.json()
         image_url = result["data"][0]["url"]
         return image_url
    except Exception as e:
         st.error(f"Error during image generation: {e}")
         return None

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.title("AI Design Assistant with CLIP-based Style Extraction & OpenAI Image Generation")
st.write("This app extracts design style characteristics from my portfolio using CLIP and uses OpenAI's image generation API to create new images that blend the extracted style with your design prompt.")

# Load and display training images.
training_images = load_training_images()
if training_images:
    display_training_images(training_images)
    with st.spinner("Extracting style adjectives from training images..."):
        style_context = generate_style_context_from_images(training_images)
    st.write("**Extracted Style Context:**", style_context)
else:
    st.write("No training images found in the 'images' folder.")

# User input for the design prompt.
user_prompt = st.text_input(
    "Enter your design prompt:",
    "A modern minimalist living room with vibrant colors."
)

if st.button("Generate Design Image"):
    if user_prompt and training_images:
        with st.spinner("Generating design image..."):
            image_url = generate_design_image_openai(user_prompt, style_context)
        if image_url:
            st.subheader("Generated Design Image:")
            st.image(image_url, use_container_width=True)
    else:
        st.warning("Please enter a design prompt and ensure training images are available.")
