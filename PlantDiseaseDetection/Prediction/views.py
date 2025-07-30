import os
import requests
import numpy as np
import cv2
import google.generativeai as genai
from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from ultralytics import YOLO
from django.contrib import messages
from .forms import ContactForm


# Load YOLO model
model_path = os.path.join(os.path.dirname(__file__), 'model_files/best.pt')
model = YOLO(model_path)

# Print the API key to verify it is set correctly

api_key = "AIzaSyBTGd9p0vGYXCCEtVqNm58GIlIDMapgEzw"
print(f"Using API Key: {api_key}")  # Debugging

genai.configure(api_key=api_key)


def get_disease_solution(disease_name):
    """ Calls Gemini AI to get disease info and solution, unless the leaf is healthy """

    # Normalize disease name for check
    if "healthy" in disease_name.lower():
        return {
            "disease_info": "The leaf is healthy. No disease detected.",
            "disease_solution": "No treatment required."
        }

    model = genai.GenerativeModel("gemini-2.0-flash")

    disease_info_prompt = f"List the key symptoms of {disease_name} in plants first, then mention its causes on a new line. Keep it under 30 words."
    disease_solution_prompt = f"Provide a **short and direct** treatment method for {disease_name} in plants in **under 30 words**."

    try:
        disease_info = model.generate_content(disease_info_prompt).text
        disease_solution = model.generate_content(disease_solution_prompt).text

        return {
            "disease_info": disease_info,
            "disease_solution": disease_solution
        }

    except Exception as e:
        print("Gemini API Error:", e)
        return {
            "disease_info": "No disease info available (Gemini API error).",
            "disease_solution": "No solution available (Gemini API error)."
        }


def predict(request):
    file_url = None  
    disease_solution = None  

    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        file_name = uploaded_file.name  

        # Save file to MEDIA_ROOT/uploads/
        save_path = os.path.join("uploads", file_name)
        file_path = default_storage.save(save_path, ContentFile(uploaded_file.read()))

        # Get full path for YOLO model
        full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)

        # Ensure file URL is correct (for displaying in HTML)
        file_url = f"{settings.MEDIA_URL}{file_path}"

        try:
            # Run model prediction
            results = model(full_file_path, conf=0.6)

            # Debug model output
            print("YOLO Results:", results)
            print("RESULTS[0] DIR:", dir(results[0]))

            # Use classification prediction if available
            if hasattr(results[0], 'probs') and results[0].probs:
                print("Using .probs for classification")
                top_prediction = results[0].probs.top1
                class_name = results[0].names[top_prediction]
                confidence = round(results[0].probs.data[top_prediction].item(), 2)

            # Fallback to object detection style if .probs is missing
            else:
                print("Falling back to .boxes for detection")
                boxes = getattr(results[0], 'boxes', None)
                if boxes is None or len(boxes) == 0:
                    raise ValueError("No disease detected in the image.")
                top_prediction = int(boxes.cls[0])
                class_name = results[0].names[top_prediction]
                confidence = round(boxes.conf[0].item(), 2)

            print(f"Predicted Class: {class_name} | Confidence: {confidence}")

            # Split class name if format is plant___disease
            if "___" in class_name:
                plant_name, disease_name = class_name.split("___")
                plant_name = plant_name.replace("_", " ").title()
                disease_name = disease_name.replace("_", " ").title()
            else:
                plant_name = "Unknown"
                disease_name = class_name.replace("_", " ").title()

            # Get info from Gemini
            disease_data = get_disease_solution(class_name)
            print("Disease Data from Gemini:", disease_data)

            # Safe unpack
            if isinstance(disease_data, dict):
                disease_info = disease_data.get("disease_info", "No info found.")
                disease_solution = disease_data.get("disease_solution", "No solution found.")
            else:
                disease_info = "No disease info available."
                disease_solution = "No solution available."

        except Exception as e:
            print("Prediction Error:", str(e))  # <-- SEE THIS IN TERMINAL
            class_name = "Prediction Error"
            plant_name = "Error"
            disease_name = "Prediction Failed"
            confidence = 0.0
            disease_info = "There was an error processing the image."
            disease_solution = "There was an error processing the image."

        context = {
            'model_prediction': class_name,
            'plant_name': plant_name,
            'disease_name': disease_name,
            'confidence': confidence,
            'file_url': file_url,
            'disease_info': disease_info,
            'disease_solution': disease_solution,  
        }
        return render(request, 'predict.html', context)
        

    return render(request, 'predict.html', {'file_url': file_url})
    



def about(request):
    return render(request, 'about.html')



def contact(request):
    if request.method == "POST":
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()  # Save the form data to the database
            messages.success(request, "Your message has been sent successfully!")
            return redirect("contact")  # Redirect to the same page after submission
    else:
        form = ContactForm()

    return render(request, "contact.html", {"form": form})




def news(request):
    API_KEY = "36368c8c79fb41f8ae028660bbc06fc1"  # Replace with actual API key

    # URL for English news related to Indian agriculture
    url_english = f"https://newsapi.org/v2/everything?q=indian+agriculture&language=en&apiKey={API_KEY}"
    response_english = requests.get(url_english)
    data_english = response_english.json()
    articles_english = data_english.get("articles", [])

    # Define the keywords to filter English news
    filter_keywords_english = ['plant disease', 'crop disease', 'farmer']

    # Filter English articles based on the defined keywords
    filtered_english_articles = [
        article for article in articles_english 
        if any(keyword.lower() in (article['title'] + article['description']).lower() for keyword in filter_keywords_english)
    ]
    return render(request, 'news.html', {'news': filtered_english_articles})



def hindi_news(request):
    API_KEY = "36368c8c79fb41f8ae028660bbc06fc1"  # Replace with actual API key

    # URL for Hindi news related to Indian agriculture
    url_hindi = f"https://newsapi.org/v2/everything?q=कृषि OR खेती OR किसान&language=hi&apiKey={API_KEY}"
    response_hindi = requests.get(url_hindi)
    data_hindi = response_hindi.json()
    articles_hindi = data_hindi.get("articles", [])

    # Define the keywords to filter Hindi news
    filter_keywords_hindi = ['फसल', 'पौधे की बीमारी', 'फसल रोग']

    # Filter Hindi articles based on the defined keywords
    filtered_hindi_articles = [
        article for article in articles_hindi
        if any(keyword in (article['title'] + article['description']) for keyword in filter_keywords_hindi)
    ]
    return render(request, 'hindi_news.html', {'news': filtered_hindi_articles})

def plant_info(request):
    return render(request, 'plant_info.html')

def quick_links(request):
    return render(request, 'quick_links.html')


def is_leaf_image(image_path):
    """Check if an image has leaf-like characteristics using edge detection"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size  
    return edge_ratio > 0.05  
