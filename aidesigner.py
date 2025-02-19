import os
import streamlit as st
import openai
import torch
import clip
from PIL import Image
from collections import Counter

# Set up device for CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and preprocessing function.
model, preprocess = clip.load("ViT-B/32", device=device)

# Define a list of adjectives that might describe a design style.
ADJECTIVES = [
    "modern", "minimalist", "vibrant", "elegant", "bold",
    "retro", "futuristic", "organic", "playful", "geometric",
    "sophisticated", "abstract", "colorful", "subtle", "dynamic",
    "classic", "innovative", "textured", "sleek", "artistic"
]

# Pre-compute CLIP text embeddings for the adjectives.
text_inputs = torch.cat([clip.tokenize(adj) for adj in ADJECTIVES]).to(device)
with torch.no_grad():
    text_embeddings = model.encode_text(text_inputs)
text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

def image_to_adjectives(image_path, top_k=3):
    """
    Given an image path, use CLIP to compute its embedding and return the top_k adjectives
    (from our predefined list) that best match the image.
    """
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image_input)
    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
    
    # Compute cosine similarity between the image embedding and each adjective embedding.
    similarity = (image_embedding @ text_embeddings.T).squeeze(0)
    values, indices = similarity.topk(top_k)
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
            image = Image.open(image_path)
            st.sidebar.image(image, caption=os.path.basename(image_path), use_column_width=True)
        except Exception as e:
            st.sidebar.write(f"Error loading {image_path}: {e}")

# Securely load your OpenAI API key from Streamlit secrets.
openai.api_key = st.secrets["openai"]["api_key"]

def generate_design_description(user_prompt, style_context):
    """
    Combines the user's design prompt with the extracted style context and
    generates a design description using GPT-4.
    """
    full_prompt = (
        f"Based on my design portfolio characterized by these style adjectives: {style_context}. "
        f"{user_prompt}"
    )
    response = openai.chat.completions.create(
        model="gpt-4",  # Change model if necessary.
        messages=[
            {
                "role": "system", 
                "content": "You are a creative design assistant who understands a unique design style derived from a portfolio of images."
            },
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# --- Streamlit App Layout ---

st.title("AI Design Assistant with CLIP-based Style Extraction")
st.write("This app extracts style adjectives from my design portfolio using CLIP and uses GPT-4 to generate creative design descriptions.")

# Load training images.
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
    "Describe a modern minimalist design with vibrant colors."
)

if st.button("Generate Design Description"):
    if user_prompt and training_images:
        with st.spinner("Generating design description..."):
            description = generate_design_description(user_prompt, style_context)
        st.subheader("Generated Design Description:")
        st.write(description)
    else:
        st.warning("Please enter a design prompt and ensure training images are available.")
