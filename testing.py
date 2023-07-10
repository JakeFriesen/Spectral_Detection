import streamlit as st
from glob import glob
from streamlit_image_annotation import detection


def detection_test():
    label_list = ['sea star', 'sea urchin', 'sea cucumber', 'kelp']
    image_path_list = glob('images/*.jpg')
    if 'result_dict' not in st.session_state:
        result_dict = {}
        result_dict[st.session_state.image_name] = {'bboxes': st.session_state.results[0].xyxyn.numpy(),'labels':[0,3]}
        st.session_state['result_dict'] = result_dict.copy()

    # num_page = st.slider('page', 0, len(image_path_list)-1, 0, key='slider')
    target_image_path = image_path_list[0]

    st.write(st.session_state.results[0].xyxyn.numpy())
    st.write(st.session_state.results[3])
    new_labels = detection(image_path=st.session_state.image_name, 
                        bboxes=st.session_state['result_dict'][st.session_state.image_name]['bboxes'], 
                        # bboxes = st.session_state.results[0],
                        # labels = st.session_state.results[3],
                        labels=st.session_state['result_dict'][st.session_state.image_name]['labels'], 
                        label_list=label_list, key=st.session_state.image_name)
    if new_labels is not None:
        st.session_state['result_dict'][st.session_state.image_name]['bboxes'] = [v['bbox'] for v in new_labels]
        st.session_state['result_dict'][st.session_state.image_name]['labels'] = [v['label_id'] for v in new_labels]
    st.json(st.session_state['result_dict'])