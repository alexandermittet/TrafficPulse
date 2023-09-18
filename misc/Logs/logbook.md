#### Date: 11. sept 2023  
  - IRL:  
    - Meeting with Bjorn:  
      - Introduced him to the project  
      - Discussed details and scope. 'trappemodel'  
    - Made plan, visual and sent to all superiors  
  - Technical:  
    - Sniffing position in py  
    - Plotting positions on img in py
   
#### 13. sept 2023
  - Technical:
    - Implemented a version that could take video as input
    - Created the first draft of a id system using sort, this needs to be optimised in the future as sort makes the program run very slowly (around 11 fps, 80ms processing pr. frame).

#### 14. sept 2023
- Called Aase i KK, handling manual traffic counting
- Contacted 4 stakeholders on linkedin:
  - Maja Sig BEstergaard
  - Steffen Bank
  - Rebekka Overgaard Birkov
  - PeterAndreas Rosbak Juhl
- Technical:
  - Tried to optimize the program where we can run the model on video.
    - We tried downscaling the video
    - We tried to only segment one channel of the RGBA channels (we couldn't get this working currently)
  - Another problem we ran into using sort was that we expirenced alot of flickering when running the program which caused objects to be assinged new ids when being 'redetected'.
    - We tried fixing this using and combination of sort and some calculations of the IOU but this only somewhat helped on the problem as the program still ran slow.
    - By doing research and discovering that IOU actually could be used to track an object over multiple frames, we chose to ditch sort and implement our own detection relying only on IOU. We then implemented our own system for assignming ids. This ran much smoother with around 28fps and 30ms processing pr. frame. Without testing but just by observation we saw much better results than using sort as we now expirenced alot less flickering when using the smallest model from YOLOv5 (nano). We also kept the implementation of downscaling as we do not need 4k video to do detection.
  
#### 15. sept 2023
  - Technical:
      - We discovered yolov7 and yolov8.
      - We successfully implmented yolov7 into our current code but with yolov7 we saw a decrease in performance when running our program.
      - We are still tring to make yolov8 work with our code as yolov8 works using the ultralytics library

#### 18. sept 2023
- IRL:
  - We have called Steffen from Movia about getting in contact with relevant people to do the mom test. We gave us an email to a person called Jakob Bommersholdt which we could call and talk to.
- Technical:
  - We have begun working on a test set such that when we change code to optimzie it we know if it will have an effect on the over all accuracy. This will also help us eliminate overfitting to a single video. The way we have done this is by manually annotating a video using RoboFlow. 
