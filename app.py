# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper




#Stages of detection process added to session state
if 'detect' not in st.session_state:
    st.session_state['detect'] = False
if 'download' not in st.session_state:
    st.session_state['download'] = False
if 'predicted' not in st.session_state:
    st.session_state['predicted'] = False
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = False



# Setting page layout
st.set_page_config(
    page_title="ECE 499 Marine Species Detection",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("🪸 ECE 499 Marine Species Detection")
# Sidebar
st.sidebar.header("Detection Configuration")
# Model Options
detect_type = st.sidebar.radio("Choose Detection Type", ["Objects Only", "Objects + Segmentation"])
model_type = st.sidebar.radio("Select Model", ["Built-in", "Upload"])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Built-in':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Upload':
    model_file = st.sidebar.file_uploader("Upload a model...", type=("pt"))
    try:
        model_path = Path(settings.MODEL_DIR, model_file.name)
    except:
        st.sidebar.write("No Model Uploaded Yet...")

# Load ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.sidebar.write("Unable to load model...")
    # st.error(f"Unable to load model. Check the specified path: {model_path}")
    # st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST, help="Choose if a single image or video will be used for detection")


if st.session_state["initialized"] == False:
    with st.spinner('Initializing...'):
        helper.init_func()


source_img = None
tab1, tab2 = st.tabs(["Detection", "About"])

#Main Detection Tab
with tab1:
    # If image is selected
    if source_radio == settings.IMAGE:
        source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
        if source_img is not None:
                helper.change_image(source_img)
        
        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    file_name = ""
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image",
                            use_column_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image",
                            use_column_width=True)
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                default_detected_image = PIL.Image.open(
                    default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image',
                        use_column_width=True)
            else:
                #Uploaded image
                st.sidebar.button('Detect Objects', on_click=helper.click_detect)
                
                #If Detection is clicked
                if st.session_state['detect']:
                    #Perform the prediction
                    boxes, labels = helper.predict(model, uploaded_image, confidence, detect_type)
                    #Show the detection results              
                    selected_boxes = helper.show_detection_results(boxes, labels)

                    #Download Button
                    try:
                        csv = helper.download_boxes(selected_boxes)
                    except:
                        st.write("No results yet...")
                    st.sidebar.download_button( label = "Download Results", 
                                        data=csv, 
                                        file_name="Detection_Results.csv", 
                                        mime='text/csv')

                        

    elif source_radio == settings.VIDEO:
        st.write("Under Construction...")
        # helper.play_stored_video(confidence, model)

    else:
        st.error("Please select a valid source type!")


with tab2:
    st.header("About the App")
    st.write("Have a question or comment? Let us Know!")
    st.write("Email me jakefriesen@uvic.ca")
    st.write("Visit the GitHub for this project: https://github.com/JakeFriesen/Spectral_Detection")

    st.header("How to Use")
    st.write("Source: Image will detect species in a single image, video will detect for the whole image")
