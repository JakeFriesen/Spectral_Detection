# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st
import supervision as sv

# Local Modules
import settings
import helper

#Stages of detection process added to session state
if 'detect' not in st.session_state:
    st.session_state['detect'] = False
if 'predicted' not in st.session_state:
    st.session_state['predicted'] = False
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'image_name' not in st.session_state:
    st.session_state.image_name = None
if 'list' not in st.session_state:
    st.session_state.list = None
if 'add_to_list' not in st.session_state:
    st.session_state.add_to_list = False
if 'img_num' not in st.session_state:
    st.session_state.img_num = 0
if 'next_img' not in st.session_state:
    st.session_state.next_img = False

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
    "Select Model Confidence", 25, 100, 40
    # , on_change = helper.repredict()
    )) / 100

# Selecting The Model to use
if model_type == 'Built-in':
    #Built in model - Will be the best we currently have
    model_path = Path(settings.DETECTION_MODEL)
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.sidebar.write("Unable to load model...")
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        # st.error(ex)
elif model_type == 'Upload':
    #Uploaded Model - Whatever you want to try out
    model_file = st.sidebar.file_uploader("Upload a model...", type=("pt"))
    try:
        model_path = Path(settings.MODEL_DIR, model_file.name)
        with open(model_path, 'wb') as file:
            file.write(model_file.getbuffer())

        model = helper.load_model(model_path)
    except Exception as ex:
        st.sidebar.write("No Model Uploaded Yet...")
        # st.error(ex)


st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST, help="Choose if a single image or video will be used for detection")

# Initializing Functions
# Put here so that the sidebars and title show up while it loads
if st.session_state["initialized"] == False:
    with st.spinner('Initializing...'):
        helper.init_func()

source_img = None
tab1, tab2 = st.tabs(["Detection", "About"])

#Main Detection Tab
with tab1:
    # If image is selected
    if source_radio == settings.IMAGE:
        source_img_list = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'), key = "src_img", accept_multiple_files= True)
        if source_img_list:
            helper.change_image(source_img_list)
            # st.write(st.session_state.img_num)
            source_img = source_img_list[st.session_state.img_num]
        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    # file_name = ""
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
                st.sidebar.button('Detect', on_click=helper.click_detect)

            #If Detection is clicked
            if st.session_state['detect']:
                #Perform the prediction
                helper.predict(model, uploaded_image, confidence, detect_type)
        #If Detection is clicked
        if st.session_state['detect']:
            #Show the detection results
            with st.spinner("Calculating Stats..."):
                # if detect_type == "Objects + Segmentation":
                selected_df = helper.results_math(uploaded_image, detect_type)
                # else:
                #     selected_df = helper.show_detection_results()
                
            #Download Button
            list_btn = st.button('Add to List')
            if list_btn:
                helper.add_to_list(selected_df)
                st.session_state.next_img = True
    
        #Always showing list if something is in it
        if st.session_state.add_to_list:
            st.write("Image List:")
            st.dataframe(st.session_state.list)
            try:
                st.download_button( label = "Download Results", 
                                data=st.session_state.list.to_csv().encode('utf-8'), 
                                file_name="Detection_Results.csv", 
                                mime='text/csv')
            except:
                st.write("Add items to the list to download them")

                        

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
    st.write(":blue[**Choose Detection Type:**] Choose to detect species only, or also use segmentation to get coverage results.")
    st.write(":blue[**Select Model:**] Choose between the built in model, or use your own (supports .pt model files).")
    st.write(":blue[**Select Model Confidence:**] Choose the confidence threshold cutoff for object detection. Useful for fine tuning detections on particular images.")
    st.write(":blue[**Select Source:**] Choose to upload an image or video")

    st.header("Image Detection")
    st.write("After an image is uploaded, the :blue[**Detect**] button will display, and run the detection or segmentation based on the :blue[**Detection Type**] chosen above.")
    st.write("The detected and segmented images will be displayed, along with the image detection results.")
    st.header("Results")
    st.write("A list of results will be displayed, with an index corresponding to the numbered box in the detected image.")
    st.write("Select which results to keep, manually select the substrate, and press :blue[**Add To List**], which will show the current list of images saved")
    st.write("When complete, press :blue[**Download Results**] to download the csv file with the resulting data")
    st.header("Batch Images")
    st.write("If multiple files are uploaded, after pressing :blue[**Add To List**], pressing :blue[**Detect**] again will load the next image.")
    st.header("Video Detection")
    #TODO: VIDEO STUFF
    st.write("Under Construction...")