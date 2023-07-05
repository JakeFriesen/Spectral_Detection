# ECE 499 Species Detection for Spectral Labs
## About
This needs a write up about the project


## TODO:
- [ ] User Interface (Jake)
    - [x] Upload your own model
    - [x] Choose image/video, transect/drop quadrat, confidence
    - [x] Manual selection of detections (check how intuitive this is)
    - [ ] Slider for video slicing
    - [x] Batch detection workflow
    - [x] Manual substrate classifier
- [ ] Functional UI Changes
    - [ ] Use form for sidebar toggles
    - [x] Add Substrate to table
    - [x] Add detection results path (no coverage)
- [x] SAM Integration into website (Jake)
- [ ] Video parsing function
    - [ ] Selectable frame sequence (transects)
    - [ ] Individual tracking after detection (transects)
    - [ ] Finding optimal frame (drop quadrat)
- [ ] Results
    - [ ] Area calculation inside pvc pipe
    - [ ] Diameter of urchins, stars?
    - [x] Spreadsheet formatting
- [ ] Model (Matt)
    - [x] Better kelp model
    - [ ] More data from Romina (?) to make models better
    - [ ] Subclass model for sea stars
- [ ] Manual addition of detections
    - [ ] Using a python chart? Need to convert detection results 
    - [ ] https://discuss.streamlit.io/t/new-component-streamlit-image-annotation-a-easy-way-to-annotate-images-using-streamlit/41146
- [ ] Documentation
    - [ ] A lot of this, for everything
    - [ ] Add it to the GitHub: https://github.com/JakeFriesen/Spectral_Detection
    - [ ] Collect all data, and give it to Navneet after project is over


#### References
User Interface inspiration: https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/tree/master
Streamlit: https://github.com/streamlit/streamlit
Ultralytics: https://github.com/ultralytics/ultralytics
Meta SAM: https://github.com/facebookresearch/segment-anything
