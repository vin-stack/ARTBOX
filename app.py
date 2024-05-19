import streamlit as st
import requests
import time
import random
from PIL import Image
import io
from rembg import remove
import cv2
import cvzone
import numpy as np
from streamlit_extras.stodo import to_do
from io import BytesIO

# Page navigation
def navigate_to(page):
    st.session_state.page = page

# Page 1: Artbox
def artbox():
    st.title("Artbox")

    # Your existing Artbox code here
    API_URL = "https://api-inference.huggingface.co/models/chaitanyakusumanchi/chaitanya_photorealism"
    headers = {"Authorization": "Bearer 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"}

    # Membership Levels (price and request limits)
    MEMBERSHIP_LEVELS = {
        "Basic": (9.99, 10),
        "Premium": (39.99, 50),
        "Pro": (69.99, 100)
    }

    if "membership" not in st.session_state:
        st.session_state["membership"] = None
    if "saved_images" not in st.session_state:
        st.session_state["saved_images"] = []
    if "request_count" not in st.session_state:
        st.session_state["request_count"] = 0
    if "art_generated" not in st.session_state:
        st.session_state["art_generated"] = False

    def generate_unique_image(prompt):
        image_bytes = None
        while image_bytes is None:
            noise = str(random.random())
            modified_prompt = f"{prompt} {noise}"

            try:
                response = requests.post(API_URL, headers=headers, json={"inputs": modified_prompt})
                image_bytes = response.content
            except requests.exceptions.RequestException as e:
                print(f"Error generating image: {e}")
                time.sleep(5)

        try:
            image = Image.open(io.BytesIO(image_bytes))
            with open('image.jpg', 'wb') as f:
                f.write(image_bytes)
            imag = Image.open('image.jpg')
            image = remove(imag)
            return image
        except:
            with open('image.jpg', 'wb') as f:
                f.write(image_bytes)
            image = Image.open('image.jpg')
        return image

    def show_membership_info():
        selected_membership = st.selectbox("Select Membership", list(MEMBERSHIP_LEVELS.keys()))
        price, request_limit = MEMBERSHIP_LEVELS[selected_membership]
        st.write(f"**Price:** ${price}/month")
        st.write(f"**Request Limit:** {request_limit} images")

        if st.button("Subscribe Now"):
            st.session_state["membership"] = selected_membership
            st.session_state["request_count"] = 0
            st.success(f"Subscribed to {selected_membership} membership!")
            if st.button("Continue"):
                print("hi")

    if st.session_state["membership"] is None:
        st.header("Welcome to Artbox!")
        show_membership_info()
    else:
        st.header(f"Artbox - {st.session_state['membership']} Member")
        remaining_requests = MEMBERSHIP_LEVELS[st.session_state["membership"]][1] - st.session_state["request_count"]
        st.write(f"Remaining Requests: {remaining_requests}")

        prompt = st.text_input("Enter your tattoo design prompt:")
        styles = st.multiselect("Select desired styles (optional)", ["realistic", "sketchy", "abstract"])

        if st.button("Generate Art"):
            if st.session_state["request_count"] < MEMBERSHIP_LEVELS[st.session_state["membership"]][1]:
                generated_image = generate_unique_image(f"{prompt} {' '.join(styles)}")
                st.image(generated_image, width=500)
                st.session_state["request_count"] += 1
                st.session_state["art_generated"] = True
            else:
                st.warning(f"You have reached your request limit for the {st.session_state['membership']} membership.")

        st.markdown("_____")

        if st.session_state["art_generated"]:
            if st.button("Call AI"):
                navigate_to("live_face_detection")
                to_do(
                    [(st.write, "Yes Proceed")],
                    "Yes",
                )

# Page 2: Live Face Detection
def live_face_detection():
    st.title("Live Face Detection with Overlay")

    enable_detection = st.checkbox("Enable hand Detection", value=False)
    cascade = cv2.CascadeClassifier('hand.xml')
    
    enable_detection1 = st.checkbox("Enable Face Detection", value=False)
    cascade1 = cv2.CascadeClassifier('face.xml')

    overlay = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    overlay_applied = False
    frame = st.camera_input("Live Webcam Feed")

    if frame is not None:
        frame_bytes = frame.read()  # Read bytes from UploadedFile
        frame = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)  # Decode from bytes
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Function to process a frame and apply face detection and overlay
        def process_frame(frame):
            if enable_detection:
                hands = cascade.detectMultiScale(gray_scale)
                for (x, y, w, h) in hands:
                    overlay_resize = cv2.resize(overlay, (int(w * overlay_scale), int(h * overlay_scale)))
                    frame = cvzone.overlayPNG(frame, overlay_resize, [x + overlay_x_offset, y + overlay_y_offset])

            if enable_detection1:
                faces = cascade1.detectMultiScale(gray_scale)
                for (x, y, w, h) in faces:
                    overlay_resize = cv2.resize(overlay, (int(w * overlay_scale), int(h * overlay_scale)))
                    frame = cvzone.overlayPNG(frame, overlay_resize, [x + overlay_x_offset, y + overlay_y_offset])

            return frame

        # Sliders for overlay size and position adjustments
        overlay_scale = st.slider("Overlay Scale", 0.1, 2.0, 1.0, step=0.1)
        overlay_x_offset = st.slider("Overlay X Offset", -200, 200, 0, step=10)
        overlay_y_offset = st.slider("Overlay Y Offset", -200, 200, 0, step=10)

        # Process and display the frame
        processed_frame = process_frame(frame)
        st.image(processed_frame, channels="BGR")  # Display the image
        

        # Convert the processed frame to an image for downloading
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_image = Image.open(BytesIO(buffer))

        # Download button
        buffered = BytesIO()
        processed_image.save(buffered, format="JPEG")
        st.download_button(
            label="Download Image",
            data=buffered.getvalue(),
            file_name="processed_image.jpg",
            mime="image/jpeg"
        )


    if st.button("Back to Artbox"):
        navigate_to("artbox")
        to_do(
            [(st.write, "Yes Proceed to Artbox")],
            "Yes",
        )

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "artbox"

# Page routing
if st.session_state.page == "artbox":
    artbox()
elif st.session_state.page == "live_face_detection":
    live_face_detection()
