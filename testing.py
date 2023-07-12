import streamlit as st
from glob import glob
from streamlit_image_annotation import detection
import settings
from pathlib import Path


def detection_test():
    #Grab the list of classes for this detection
    #Limited to the classes in the current model, can't add more
    #TODO: This could be a hardcoded, or user updateable cached list
    label_list = list(st.session_state.results[2].values())

    bboxes = []
    labels = []
    for box in st.session_state.results[0].xywh.numpy():
        top_coord = [box[0] - (box[2]/2), box[1] - (box[3]/2)]
        bboxes.append([top_coord[0], top_coord[1], box[2], box[3]])
    for detections in st.session_state.results[1]:
        labels.append(int(detections[3]))

    if 'result_dict' not in st.session_state:
        result_dict = {}
        result_dict[st.session_state.image_name] = {'bboxes': bboxes,'labels':labels}
        st.session_state['result_dict'] = result_dict.copy()

    target_image_path = Path(settings.IMAGES_DIR , st.session_state.image_name)
    new_labels = detection(image_path=target_image_path, 
                        bboxes=st.session_state['result_dict'][st.session_state.image_name]['bboxes'], 
                        labels=st.session_state['result_dict'][st.session_state.image_name]['labels'], 
                        label_list=label_list, 
                        key=st.session_state.image_name,
                        height = 1080,
                        width = 1920)
    if new_labels is not None:
        st.session_state['result_dict'][st.session_state.image_name]['bboxes'] = [v['bbox'] for v in new_labels]
        st.session_state['result_dict'][st.session_state.image_name]['labels'] = [v['label_id'] for v in new_labels]
    st.json(st.session_state['result_dict'])