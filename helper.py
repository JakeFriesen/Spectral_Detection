import math
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import streamlit as st
import pandas as pd
import base64
import cv2
import numpy as np
import supervision as sv
import torch
from supervision.draw.color import Color, ColorPalette

import settings

@st.cache_data
def init_models():
    SAM_ENCODER_VERSION = "vit_l"
    SAM_CHECKPOINT_PATH = "weights/sam_vit_l_0b3195.pth"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
    global sam_predictor 
    sam_predictor = SamPredictor(sam)

def init_func():
    init_models()
    global file_name 
    file_name = ""
    st.session_state['initialized'] = True


# Segment Anything Model
# Runs the segmenter, with proper data formatting
def segment(image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    global sam_predictor
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


# Checks that a new image is loaded
# Changes the session state accordingly
def change_image(img):
    global file_name
    if img.name != file_name:
        file_name = img.name
        st.session_state['detect'] = False
        st.session_state['predict'] = False

# Use this to repridict IMMEDIATELY, 
# Detect Does not have to be pressed again
def repredict():
    st.session_state['predict'] = False

# Use this to repredict AFTER pressing detect
def redetect():
    st.session_state['predict'] = False
    st.session_state['detect'] = False

# Detect Button 
def click_detect():
    st.session_state['detect'] = True


# Creates the CSV file with the result data
# TODO: This needs a lot of formatting
def download_boxes(selected_boxes):
    # Create a DataFrame to hold the selected bounding box data
    df = pd.DataFrame(selected_boxes, columns=["Data"])

    # Convert the DataFrame to a CSV string
    csv_string = df.to_csv().encode('utf-8')
    return csv_string


# Predict Function
# Performs the object detection and image segmentation
# Also runs the results statistics 
def predict(_model, _uploaded_image, confidence, detect_type):
    boxes = []
    labels = []
    if st.session_state['predict'] == False:
        res = _model.predict(_uploaded_image, conf=confidence)
        boxes = res[0].boxes
        masks = res[0].masks
        detections = sv.Detections.from_yolov8(res[0])
        classes = res[0].names
        if(detections is not None):
            labels = [
                f"{idx} {classes[class_id]} {confidence:0.2f}"
                for idx, [_, _, confidence, class_id, _] in enumerate(detections)
                ]
        box_annotator = sv.BoxAnnotator(text_scale=3, text_thickness=4, thickness=4, text_color=Color.white())
        annotated_image = box_annotator.annotate(scene=np.array(_uploaded_image), detections=detections, labels=labels)
        st.image(annotated_image, caption='Detected Image', use_column_width=True)
        st.session_state['predict'] = True
    #Segmentation
    if detect_type == "Objects + Segmentation":
        with st.spinner('Running Segmenter...'):
            #Do the Segmentation
            detections.mask = segment(
                image=cv2.cvtColor(np.array(_uploaded_image), cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )
            # annotate image with detections
            box_annotator = sv.BoxAnnotator()
            mask_annotator = sv.MaskAnnotator()
            annotated_image = mask_annotator.annotate(scene=np.array(_uploaded_image), detections=detections)
            # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            st.image(annotated_image, caption='Segmented Image', use_column_width=True)
            results_math(detections, _uploaded_image)
    return boxes, labels

#TODO: Class differentiation!!!
def results_math(detections, image):
    grid_size_dimension = math.ceil(math.sqrt(len(detections.mask)))

    segmentation_mask = detections.mask
    binary_mask = np.where(segmentation_mask > 0.5, 1, 0)
    white_background = np.ones_like(image) * 255
    new_images = white_background * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]

    total_percentage = np.sum(new_images != 255) / (new_images[0].size) *100
    percentage = [np.sum(x != 255)/(x.size)*100 for x in new_images]
    titles = ["Coverage:" + str(np.around(x,2)) + "%" for x in percentage]

    st.write("The Total Percentage of Sea Urchins in this image is:", total_percentage, "%")
    st.write("Total Number of Sea Urchins is: ", len(detections))
    st.write("I know this is only ever sea urchins... I'll fix it")



def show_detection_results(boxes, labels):
    selected_boxes = []
    try:
        with st.expander("Detection Results"):
            for idx, box in enumerate(boxes):
                checkbox_label = f"{labels[idx]}"
                checkbox_state = st.checkbox(checkbox_label, value=True)
                if checkbox_state:
                    #TODO: Format this result for csv!
                    selected_boxes.append(f"{box.xyxy}")
    except Exception as ex:
        st.write("No image is uploaded yet!")
    return selected_boxes









def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))