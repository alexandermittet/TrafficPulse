# TRAFFIC PULSE
TrafficPulse is a project focused on automating traffic data collection using Machine Learning techniques. It leverages computer vision and neural networks to accurately count and categorize different types of vehicles in urban settings.

## Abstract:

In the face of rapid urbanization, traditional traffic management methods struggle to provide real-time, accurate data essential for effective urban planning. This paper introduces TrafficPulse, a software system leveraging computer vision and the YOLOv8 model to revolutionize traffic counting in urban environments. By integrating real-time video analysis and machine learning, TrafficPulse accurately identifies and categorizes various vehicle types, offering a significant improvement over manual and sensor-based methods. The system's development, from conception to implementation, highlights the technical feasibility and economic viability of such an approach. TrafficPulse's ability to provide continuous and comprehensive data collection makes it a valuable tool for dynamic traffic management and smart city infrastructure. The project demonstrates the potential of machine learning in addressing complex urban challenges, paving the way for future innovations in intelligent transportation systems.

Screenshot of model in action:
![trafficpulse examples](https://github.com/user-attachments/assets/67f3cd33-86a1-48ed-a6db-d9062ad0863c)
Statistics and findings for Copenhagen traffic, field trip:
![trafficpulse stats findings](https://github.com/user-attachments/assets/a9df487a-fdcc-4703-b2c0-7ead96d7a9a7)



## Demo

Follow these steps to run a demo of TrafficPulse:

1. **Install Requirements**

   You can set up the environment using either pip or conda. Run one of the following commands:

   Using pip:
   ```
   pip install -r requirements.txt
   ```

   Using conda:
   ```
   conda create --name TPenv python=3.10 --file requirements.txt
   ```

2. **Run the Demo**

   In the root folder, execute the demo script:
   ```
   python3 demo.py
   ```

3. **Line Counter Setup**

   Once the demo starts, you must select two points on the screen. These points will define the line used for counting vehicles.

4. **Demo Execution**

   If the installation is correct, the tracker will run on a 30-second video demonstrating the traffic counting capabilities of TrafficPulse.

## Standards

To maintain code quality and consistency, we adhere to the following standards:

- **Code Style:** Follow the PEP8 standard.
- **Documentation:** Every function must include a docstring explaining its purpose and usage.
- **Git Practices:** Commit messages and code reviews should be concise and informative.
- **Pre-commit Hook:** We use pre-commit hooks to enforce PEP8 standards, primarily through the 'black' code formatter.

## Requirements

Ensure that only necessary packages are included before freezing the requirements. To manage the dependencies, use:

To install dependencies from the requirements file:
```
pip install -r requirements.txt
```

If you end up installing more/less packages, update the requirements.txt using:
```
pip freeze > requirements.txt
```
