import google.generativeai as genai
import os

# Set up Gemini API Key
genai.configure(api_key="AIzaSyDX2SQ9-1fi2ignqyn_oerIVPPsf4SfkOE")

def get_disease_solution(disease_name):
    model = genai.GenerativeModel("gemini-1.5-pro")  # Use the correct model
    prompt = f"What are the symptoms and treatments for {disease_name} in plants? Provide a short 3-4 line summary."
    
    response = model.generate_content(prompt)
    
    return response.text  # Return AI-generated text

# Example usage
print(get_disease_solution("Powdery Mildew"))
