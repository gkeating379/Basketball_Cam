# Table of Contents
- [Visualization of Results](#visualization-of-results)
- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Future Improvements](#future-improvments)

# Visualization of Results

<p style="text-align: center">
    <img src="Input_Videos\knicks_trim.gif" width="400"/>
    <img src="Output_Videos\annotated_input_trim.gif" width="400"/>
</p>

<p style="text-align: center"> 
    <img src="Output_Videos\court_diagram_r_trim.gif" width="800" /> 
</p>

Top left: Input Video

Top right: Marked up output.  Red and yellow lines indicate court boundaries. 
Players are marked with the number that is used in the court diagram. 
Blue lines indicate features tracked with optical flows

Bottom: Court diagram used to show player motion.


# Description

Takes an input video of a basketball game and creates a top down view of the players movement across the court.
This resulting view can be used to better analyze the spacing of players, distance traveled at different
times and can even be generalized to produce deeper analysis of shot location and defensive preformance.
Expects the camera to be the common broadcast angle, eg. fixed at the halfcourt line with a view of most of the half.  


# Features

- Display player locations on a top-down view of the court
- Identify player locations with YOLOv5
- Automatically detect court boundaries by Hough Lines
- Use Optical Flow of court feaures to update court boundaries where
Hough Lines fails

# Installation

```
pip install -r requirements.txt
```

# Usage

```
python main.py
```
Will compute the Court Diagram with the sample knicks.mp4.  To use a different 
video, change input_video_path in main.py.  To use a different YOLOv5 model,
change model in main.py

Output is found in Output_Videos folder by default.

# Future Improvments
- Identify players by jersey color and number.  
    - Using KMeans for jersey color fails because the court is often very similar 
    to the team in white.  Training a model for jersey number was challenging because
    most datasets zoom into the jersey instead of viewing the full player
- Identify ball possesion.  
    - The models I worked with could not identify the ball for more than a few frames
    at best.
- Shot detection
    - Detecting ball possesion could help with shots.  Some data exists to detect a 
    field goal attempt from the player stance.  Could also identify the score on the screen.
    - Doing this would also open up the door to assist and rebound detection
- Transition the Homography to ML model.
    - Modern field recognition uses deep learning to compute the homography end-to-end.
    - My CPU is too slow for this so I used Hough Lines which is less robust