from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate caption
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

# --- MAIN TEST ---
if __name__ == "__main__":
    img = Image.open("test.jpg")  # Make sure test.jpg is in the same folder
    caption = generate_caption(img)
    print("Generated caption:", caption)  
    
    