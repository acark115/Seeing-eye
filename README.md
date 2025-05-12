# Gören Göz: Object Recognition and Voice Assistance System

![IMG-20250412-WA0002](https://github.com/user-attachments/assets/038d5ff5-0a1c-41d4-8a03-b4cf94df6c25)


## Project Aim

This project aims to enable visually impaired individuals to navigate their daily lives more independently and safely by detecting objects around them, especially traffic lights. The system seeks to increase users' awareness of their surroundings and prevent potential hazards by providing voice alerts about nearby obstacles.

## Project Method

The "Gören Göz" system operates by integrating various sensors and artificial intelligence technologies:

* **Object Recognition:** YOLO (You Only Look Once) algorithm is used for real-time detection of objects in the environment.
* **Traffic Light Detection:** The Raspberry Pi camera module and image processing techniques identify the colors of traffic lights (green, red).
* **Voice Alert:** Detected objects, their distances, and traffic light statuses are conveyed to the user via voice prompts using the espeak library. Warnings such as "very close," "close," or "be careful" help the user move safely.
* **Data Processing:** The system, running on a Raspberry Pi 4, processes data and camera images to generate meaningful information.
* **Image Processing:** The OpenCV library is used to process camera images and visualize the results of the object recognition model (an example output is shown in Image 3).
* **Time Management:** The `time` library manages the timing of different processes and optimizes real-time performance.
* **Data Logging:** All voice alerts are recorded via a microSD card, contributing to retrospective analysis and system development processes.

## Technologies Used

* **Artificial Intelligence:** YOLO (You Only Look Once)
* **Image Processing:** OpenCV
* **Text-to-Speech:** espeak
* **Microprocessor:** Raspberry Pi 4
* **Camera:** Web Camera
* **Programming Language:** Python (assumed if not specified)

## Setup and Usage

(This section will be detailed when the project codes are shared.)

1.  **Install Required Libraries:**
    ```bash
    pip install opencv-python
    # Other libraries can be added here if Python is used.
    ```
2.  **Hardware Connections:** The Raspberry Pi 4, sensors, and other components must be connected correctly.
3.  **Run the Code:** The main Python script is executed to start the system.
    ```bash
    python object ident.py # An example execution command
    ```

## Images

![WhatsApp Image 2025-05-12 at 14 56 10](https://github.com/user-attachments/assets/bf486c22-23d7-4318-b254-b22481be1bf8)
*The system being tested by a user.*

![DSC_0270](https://github.com/user-attachments/assets/f0581dbe-e11f-4ac4-8a3f-50a66827b872)
*The image on the screen shows how the object recognition algorithm frames objects.*

## Contributions

This project is open for contributions. Any kind of contribution (code development, bug reporting, documentation improvement, etc.) is welcome.


## Contact

**acar.keremarda115@gmail.com


