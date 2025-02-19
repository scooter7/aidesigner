import streamlit as st
import openai

# Set your OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]

def generate_design_description(prompt):
    """
    Generates a design description using GPT-4 based on the given prompt.
    """
    response = openai.chat.completions.create(
        model="gpt-4",  # You can specify a particular version if needed (e.g., "gpt-4-0314")
        messages=[
            {"role": "system", "content": "You are a creative design assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=100
    )
    # Extract the generated content from the response
    return response.choices[0].message.content.strip()

# Streamlit app UI
st.title("Design Assistant Powered by GPT-4")
st.write("Generate creative design descriptions based on your prompt.")

# User input field for design prompt
user_prompt = st.text_input(
    "Enter your design prompt:",
    "Describe a modern minimalist design with vibrant colors."
)

# When the button is clicked, generate and display the description
if st.button("Generate Description"):
    if user_prompt:
        description = generate_design_description(user_prompt)
        st.subheader("Generated Description:")
        st.write(description)
    else:
        st.warning("Please enter a design prompt to generate a description.")
