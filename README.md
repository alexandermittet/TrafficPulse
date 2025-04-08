# TRAFFIC PULSE
TrafficPulse is a project focused on automating traffic data collection using Machine Learning techniques. It leverages computer vision and neural networks to accurately count and categorize different types of vehicles in urban settings.

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
