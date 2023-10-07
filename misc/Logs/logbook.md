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
 
#### 19. sept 2023
- Technical:
    - Current got yolov8 working with id assignment using iou, but currently we still expirence id switching, so currently we are trying to see if we expirence a decreses in id switches.
    - Tried to make yolov8 work with deepsort for tracking. We made it work with displaying the ids but the bounding boxes where not in the correct sizes and we didnt really see any improvement.
    - We then discovered a repo with a library called ByteTrack. We made version that uses yolov8 with bytetracker to track objects across frames. We saw a huge improvement in the flicker and tracking of objects. One thing to mention is that we need to find a way to reset the ids after each run and make it work with an external webcam. Later on we need to add our allready developed line counter to this program.
- IRL:
  - Meeting with Jakub, giving status and discussing our how, pitch. Jakub mentioned some other ideas that our technology could be used for, e.g. animal tracking (birds, wila boars, even farm animals in cages).
  - Discussed possibility for exam date postponing. JR said sure, but that the other people on the contract needed to confirm aswell.
  - We contacted; Sidsel (KK)
  - We recived a 35 page manual of trafficcounting in KBH from Åse.

#### 21. sept 2023
- Technical:
  - Implemented line counter to the yolov8 model with ByteTracker
  - Working on tracking cars based on color and using area for determening if the cars is moving closer or away from the camera

#### 21-26. sept 2023
- Technical:
  - Converted the jupyter notebook to a .py file, further more structed everything into YOLOv8BTSP.py, functions.py and constant.py for easier management
  - We now display the area of the bounding boxes and can switch the color based on how close/far away an object is to the camera.
  - We now store the amount of cars pr frame in a csv file such that with plot.py we can plot how many cars over time is counted, we later want to convert this to maybe pr minute or second such that we dont have for each frame.
  - We have implemented the use of webcam, we can now toggle the webcam on and off in constants.py.
  - restructured our repo
  - We also now save the original video from the webcam with out boundingboxes such that we can compare on differnet models.

#### 27-29. sept 2023
- Technical:
  - We can now draw the counting line interactively using the coordinates from mouse clicking on the frame
  - We now keep the count of each frame in a csv file for a whole run which we now plot when a run is done. 
  - We have made our own version of line_counter where we now get a unique in/out-count for each class detected.
  - We have tried to retrain a yolov8 model just for testing

- IRL:
  - We went out to dr. louises bro and did a real life test on bikes againt manual counting and sensor counting (results in run 1-4 on Marucs' mac)
  - We read and discussed the bytetrack paper as we promised Bjørn

#### 30sept-5okt. sept 2023
- IRL:
  - We finish our last workshop
  - Anton has joined or business case
  - We pitched infront of a board of judges

- Technical:
  - We have written a demo.py such that people can get a quick demo
  - We wrote a benchmark program with motmetrics and got a hold of the MOT datasets which we can benchmark against
  - We have been cleaning up our code/our github
  - We will soon create a prototype and conduct more tests in the field