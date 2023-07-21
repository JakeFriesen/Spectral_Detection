# Python In-built packages
from pathlib import Path
import PIL
import os
import numpy as np

# External packages
import streamlit as st
from ffmpy import FFmpeg

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
if 'video_data' not in st.session_state:
    st.session_state.results = []
if 'image_name' not in st.session_state:
    st.session_state.image_name = None
if 'list' not in st.session_state:
    st.session_state.list = None
if 'img_list' not in st.session_state:
    st.session_state.img_list = None
if 'add_to_list' not in st.session_state:
    st.session_state.add_to_list = False
if 'img_num' not in st.session_state:
    st.session_state.img_num = 0
if 'next_img' not in st.session_state:
    st.session_state.next_img = False
if 'segmented' not in st.session_state:
    st.session_state.segmented = False
if 'side_length' not in st.session_state:
    st.session_state.side_length = 0
if "drop_quadrat" not in st.session_state:
    st.session_state.drop_quadrat = "Percentage"
if 'manual_class' not in st.session_state:
    st.session_state.manual_class = ""
if 'class_list' not in st.session_state:
    st.session_state.class_list = []
if 'kelp_conf' not in st.session_state:
    st.session_state.kelp_conf = 0.04
if 'model_type' not in st.session_state:
    st.session_state.model_type = 'Built-in'

# Setting page layout
st.set_page_config(
    page_title="ECE 499 Marine Species Detection",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("🪸 ECE 499 Marine Species Detection")

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST, help="Choose if a single image or video will be used for detection")

# Sidebar
st.sidebar.header("Detection Configuration")
# Model Options
detect_type = st.sidebar.radio("Choose Detection Type", ["Objects Only", "Objects + Segmentation"])
model_type = st.sidebar.radio("Select Model", ["Built-in", "Upload"])
st.session_state.model_type = model_type

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 0, 100, 40,
    on_change = helper.repredict(),
    )) / 100
if model_type == 'Built-in':
    kelp_c =st.sidebar.slider(
        "Select Kelp Confidence", 0, 100, 10,
        on_change = helper.repredict(),
        )
    st.session_state.kelp_conf = float(kelp_c)/100

# Selecting The Model to use
if model_type == 'Built-in':
    #Built in model - Will be the best we currently have
    if detect_type == "Objects Only":
        model_path = Path(settings.DETECTION_MODEL)
    else:
        model_path = Path(settings.SEGMENTATION_MODEL)
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
# Option for Drop Quadrat selection
st.sidebar.radio("Choose Results Formatting", ["Percentage", "Area (Drop Quadrat)"], key = "drop_quadrat")
if st.session_state.drop_quadrat == "Area (Drop Quadrat)":
    st.sidebar.number_input("Side Length of Drop Quadrat (cm)", value = 0, key= 'side_length')

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
            "Choose an image...", 
            type=("jpg", "jpeg", "png", 'bmp', 'webp'), 
            key = "src_img", 
            accept_multiple_files= True)
        if source_img_list:
            try:
                for img in source_img_list:
                    if not os.path.exists(settings.IMAGES_DIR):
                        os.mkdir(settings.IMAGES_DIR)
                    img_path = Path(settings.IMAGES_DIR, img.name)
                    with open(img_path, 'wb') as file:
                        file.write(img.getbuffer())
            except:
                st.sidebar.write("There is an issue with writing image files")
            helper.change_image(source_img_list)
            # st.write(st.session_state.img_num)
            source_img = source_img_list[st.session_state.img_num]
        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image",
                            use_column_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    if not st.session_state['detect']:
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
        if st.session_state['detect'] and source_img is not None:
            #Perform the prediction
            try:
                helper.predict(model, uploaded_image, confidence, detect_type)
            except Exception as ex:
                st.write("Upload an image or select a model to run detection")
                st.write(ex)
        #If Detection is clicked
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.session_state['detect']:
                #Show the detection results
                with st.spinner("Calculating Stats..."):
                    selected_df = None
                    try:
                        selected_df = helper.results_math(uploaded_image, detect_type)
                    except Exception as ex:
                        st.write("Upload an image first")
                        # st.write(ex)
                    
                #Download Button
                list_btn = st.button('Add to List')
                if list_btn and (selected_df is not None):
                    helper.add_to_list(selected_df, uploaded_image)
                    st.session_state.next_img = True
                    #This gets the update to be forced, removing the double detect issue.
                    #It does look a bit weird though, consider removing
                    st.experimental_rerun()
        with bcol2:
            if st.session_state['detect']:
                st.text_input("Enter New Manual Classes", 
                              value="", 
                              help="You can enter more classes here which can be used with the manual annotator. They will not be automatically detected.", 
                              key= 'manual_class')


        #Always showing list if something is in it
        if st.session_state.add_to_list:
            st.write("Image List:")
            st.dataframe(st.session_state.list)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                try:
                    st.download_button( label = "Download Results", 
                                    help = "Download a csv with the saved image results",
                                    data=st.session_state.list.to_csv().encode('utf-8'), 
                                    file_name="Detection_Results.csv", 
                                    mime='text/csv')
                except:
                    st.write("Add items to the list to download them")
            with col2:
                helper.zip_images()
            with col3:
                if st.button("Clear Image List", help="Clear the saved image data"):
                    helper.clear_image_list()
            with col4:
                helper.dump_data_button()
                        

    elif source_radio == settings.VIDEO:
        source_vid = st.sidebar.file_uploader(
            "Upload a Video...", type=("mp4"), key = "src_vid")
        interval = st.sidebar.slider("Select Capture Rate:", 0.25, 4.00, 1.00, 0.25)
        if source_vid is not None:
            source_name = str(Path(source_vid.name).stem)
            vid_path = 'preprocess_' + source_name + '.mp4'
            des_path = 'process_'+ source_name + '.mp4'
            h264_path = Path(settings.VIDEO_RES, source_name + '_h264.mp4')
            bytes_data = source_vid.getvalue()
            video_path = helper.preview_video_upload(vid_path, bytes_data)
            if not st.session_state['detect']:
                Done = helper.capture_uploaded_video(confidence, model, interval, vid_path, des_path)
                if (True == Done):
                    st.session_state['detect'] = True
            else:
                if not os.path.exists(h264_path):
                    import subprocess
                    subprocess.call(args=f"ffmpeg -y -i {des_path} -c:v libx264 {h264_path}".split(" "))
                helper.preview_finished_capture(h264_path)
                video_df = helper.format_video_results(model, h264_path)
                list_btn = st.button('Add to List')
                if list_btn and (video_df is not None):
                    helper.add_to_listv(video_df)
                st.session_state.next_img = False
        else:
            st.session_state['detect'] = False            
        
        if st.session_state.add_to_list:
            st.write("Video List:")
            st.dataframe(st.session_state.list)
            col1, col2, col3 = st.columns(3)
            with col1:
                try:
                    st.download_button( label = "Download Results", 
                                    help = "Download a csv with the saved Video results",
                                    data=st.session_state.list.to_csv().encode('utf-8'), 
                                    file_name="Detection_Results.csv", 
                                    mime='text/csv')
                except:
                    st.write("Add items to the list to download them")
            with col2:
                helper.zip_video()
            with col3:
                if st.button("Clear List", help="Clear the saved data"):
                    helper.clear_image_list()

    else:
        st.error("Please select a valid source type!")


with tab2:
    st.header("About the App")
    st.write("Have a question or comment? Let us Know!")
    st.write("Email me jakefriesen@uvic.ca")
    st.write("Visit the GitHub for this project: https://github.com/JakeFriesen/Spectral_Detection")

    st.header("How to Use")
    st.write(":blue[**Select Source:**] Choose to upload an image or video")
    st.write(":blue[**Choose Detection Type:**] Choose to detect species only, or also use segmentation to get coverage results.")
    st.write(":blue[**Select Model:**] Choose between the built in model, or use your own (supports .pt model files).")
    st.write(":blue[**Select Model Confidence:**] Choose the confidence threshold cutoff for object detection. Useful for fine tuning detections on particular images.")
    st.write(":blue[**Select Kelp Confidence:**] Choose the confidence threshold for the Kelp class only. This was added to allow greater detection flexibility with kelp.")
    st.write(":blue[**Choose Results Formatting:**] Choose the formatting for the segmentation results. \
             Choosing :blue[Area (Drop Quadrat)] will allow you to enter the size of the PVC surrounding the image, and the results will be in square centimeters. \
             Choosing :blue[Percentage] will output the results as a percentage of the image area.")
    st.write(":blue[**Choose an Image:**] Upload the images that will be used for the detection and segmentation.")
    

    st.header("Image Detection")
    st.write("After an image is uploaded, the :blue[**Detect**] button will display, and run the detection or segmentation based on the :blue[**Detection Type**] chosen above.")
    st.write("The detected and segmented images will be displayed, along with a manual annotator and the image detection results.  \
            The Manual Annotator can be used to delete bounding boxes using :blue[del], or to add and move boxes using :blue[transform] \
            Pressing :blue[Complete] will update the results below, and also remove any bounding boxes that were deleted from the detection and segmentation results.")
    st.write("New classes can be added to the manual annotator dropdown by entering the name in the :blue[**Enter New Manual Classes**] box.")
    st.write("Note: The manual annotator will not update coverage results if a new bounding box is created, or an original bounding box is resized.")
    st.write("Note: The manual annotator does not support reclassifying existing bounding boxes at this time. Please delete the orignal and create a new bounding box with the new class type ")
    
    st.header("Results")
    st.write("A list of results will be displayed, showing the number of detections, and coverage if segmentation is selected. Coverage will be in cm^2 or %% based on the :blue[**Results Formatting**] chosen.")
    st.write("Press :blue[**Download Results**] to download the csv file with the resulting data")
    st.write("Press :blue[**Download Image**] to download the image files with the bounding boxes")
    st.write("Press :blue[**Clear Image List**] to clear all images from the saved list")
    st.write("Press :blue[**Download Data Dump**] to download the detection data in YOLO format for use with future training")
    st.header("Batch Images")
    st.write("If multiple files are uploaded, after pressing :blue[**Add To List**], pressing :blue[**Detect**] again will load the next image and start the next detection.")

    st.header("Video Detection")
    #TODO: VIDEO STUFF
    st.write("Under Construction...")