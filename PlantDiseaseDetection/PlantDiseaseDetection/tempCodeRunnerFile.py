import google.generativeai as genai

genai.configure(api_key="AIzaSyDX2SQ9-1fi2ignqyn_oerIVPPsf4SfkOE")

models = genai.list_models()

for model in models:
    print(model.name)
