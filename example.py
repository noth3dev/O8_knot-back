import google.generativeai as genai

genai.configure(api_key="AIzaSyDPfSFRRJ5WNrW36X6Gc8EvMTNnu8Tbt18")

model = genai.GenerativeModel("models/gemini-2.0-flash")
response = model.generate_content("hey")
print(response.text)
