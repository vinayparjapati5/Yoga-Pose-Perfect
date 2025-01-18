
import cv2
import mediapipe as mp
import numpy as np
import time
import csv

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate angle between three points
def calculate_angle(landmark1, landmark2, landmark3, select=''):
    if select == '1':
        x1, y1, _ = landmark1.x, landmark1.y, landmark1.z
        x2, y2, _ = landmark2.x, landmark2.y, landmark2.z
        x3, y3, _ = landmark3.x, landmark3.y, landmark3.z

        angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
    else:
        radians = np.arctan2(landmark3[1] - landmark2[1], landmark3[0] - landmark2[0]) - \
                  np.arctan2(landmark1[1] - landmark2[1], landmark1[0] - landmark2[0])
        angle = np.abs(np.degrees(radians))

    angle_calc = angle + 360 if angle < 0 else angle
    return angle_calc

def correct_feedback(model, csv_path):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Load angle data for all poses from the CSV
    pose_angle_data = {}
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                angles = list(map(float, row[:12]))
                pose_name = row[12]
                pose_angle_data[pose_name] = angles
            except ValueError:
                continue  # Skip rows with non-numerical data

    # Define the names and coordinate indices for each body part angle
    angle_name_list = [
        "Left Wrist", "Right Wrist", "Left Elbow", "Right Elbow", 
        "Left Shoulder", "Right Shoulder", "Left Knee", "Right Knee", 
        "Left Ankle", "Right Ankle", "Left Hip", "Right Hip"
    ]
    angle_coordinates = [
        [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX],
        [mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_INDEX],
        [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
        [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST],
        [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP],
        [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW],
        [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE],
        [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE],
        [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX],
        [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX],
        [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
        [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE]
    ]

    # Define thresholds for corrections
    minor_correction_threshold = 10  # Minor adjustments
    major_correction_threshold = 30  # Major adjustments

    # Status messages for corrections
    status_messages = [
        "Straighten your arm more",
        "Bend your elbow slightly",
        "Adjust your shoulder position",
        "Lift your knee higher",
        "Lower your hip a bit",
        "Open your chest more",
        "Keep your wrist stable",
        "Balance your ankle better",
        "Engage your core muscles",
        "Relax your shoulders slightly"
    ]

    while cap.isOpened():
        ret_val, image = cap.read()
        if not ret_val:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        angles = []

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Calculate angles for each body part
            for part in angle_coordinates:
                angle = calculate_angle(landmarks[part[0].value], landmarks[part[1].value], landmarks[part[2].value], '1')
                angles.append(angle)

            # Predict the pose using the deep learning model
            angles_array = np.array([angles])  # Convert to array
            predictions = model.predict(angles_array)
            predicted_class_index = predictions.argmax(axis=1)[0]
            class_name = list(pose_angle_data.keys())[predicted_class_index]
            print(f"Predicted Pose: {class_name}")

            if class_name in pose_angle_data:
                expected_angles = pose_angle_data[class_name]
            else:
                continue

            # Display feedback on each body part
            correct_angle_count = 0
            for i, part in enumerate(angle_coordinates):
                point_b = (int(landmarks[part[1].value].x * image.shape[1]), 
                           int(landmarks[part[1].value].y * image.shape[0]))
                
                deviation = abs(angles[i] - expected_angles[i])
                if deviation < minor_correction_threshold:
                    status = "OK"
                    correct_angle_count += 1
                elif deviation < major_correction_threshold:
                    status = status_messages[i % len(status_messages)]
                else:
                    if angles[i] < expected_angles[i]:
                        status = f"{status_messages[i % len(status_messages)]} - move up"
                    else:
                        status = f"{status_messages[i % len(status_messages)]} - move down"

                # Display the status at each body part
                cv2.putText(image, f"{angle_name_list[i]}: {status}", (point_b[0], point_b[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Overall posture assessment
            #posture_status = "CORRECT" if correct_angle_count > 5 else "IMPROVE"
            total_angles = len(angle_coordinates)
            correct_percentage = (correct_angle_count / total_angles) * 100

# Determine overall posture status
            posture_status = f"{correct_percentage:.1f}% CORRECT" if correct_percentage >= 50 else f"{correct_percentage:.1f}% IMPROVE"

            posture_color = (0, 255, 0) if posture_status == "CORRECT" else (0, 0, 255)
            # Display posture name and status
            cv2.putText(image, f"Pose: {class_name}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"Posture: {posture_status}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, posture_color, 2)


            # Draw pose landmarks
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Pose Estimation', image)

        if cv2.waitKey(1) == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
# Replace `your_model` with your trained model
# correct_feedback(your_model, "path_to_video.mp4", "path_to_angles.csv")
