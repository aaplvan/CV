# gradio version 3.5.0 is required to run this code. higher versions may not work
import numpy as np
import gradio as gr
from sky import load_image_from_disk, get_sky_region, params
import cv2
from gradio.components import Image as ImageInput


def detect_sky(image_path):
    # Load image
    cv_image = load_image_from_disk(image_path)
    # Process the image
    sky_region = get_sky_region(cv_image, **params)
    # Convert the OpenCV image format (BGR) to the format Gradio uses (RGB)
    sky_region = cv2.cvtColor(sky_region, cv2.COLOR_BGR2RGB)

    return sky_region

# Define the Gradio interface
iface = gr.Interface(
    fn=detect_sky,  
    inputs=ImageInput(type="filepath"),  
    outputs="image",  
    interpretation="default"
)

# Launch the interface
iface.launch(share=True)