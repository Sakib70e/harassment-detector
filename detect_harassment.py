from ultralytics import YOLO
import mediapipe as mp
import cv2
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# YOLO
model = YOLO("yolov8n.pt")

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

PROXIMITY_THRESHOLD = 100
HAND_SPEED_THRESHOLD = 40
RISK_ALERT_THRESHOLD = 90

prev_left_wrist, prev_right_wrist = None, None

def calc_speed(prev, current):
    if prev is None or current is None:
        return 0
    return np.linalg.norm(np.array(prev) - np.array(current))

def send_email(subject, body):
    sender_email = "pandarivu2004@gmail.com"  # Replace with your Gmail address
    receiver_email = "sakibmukhtar7044@gmail.com"  # Replace with recipient's email
    app_password = "vxrg muvj frxb bxhz"  # Use the generated App Password, not your normal password

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to Gmail's SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Start encryption
        server.login(sender_email, app_password)  # Login with App Password
        server.sendmail(sender_email, receiver_email, msg.as_string())  # Send email
        server.quit()  # Close connection
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame, verbose=False)[0]
    pose_results = pose.process(frame_rgb)

    risk_score = 0
    persons = []

    # Detect people
    for box in results.boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            persons.append(((x1 + x2)//2, (y1 + y2)//2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Check proximity
    if len(persons) >= 2:
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                distance = np.linalg.norm(np.array(persons[i]) - np.array(persons[j]))
                if distance < PROXIMITY_THRESHOLD:
                    risk_score += 40
                    cv2.line(frame, persons[i], persons[j], (0, 0, 255), 2)
                    cv2.putText(frame, "TOO CLOSE!", persons[i],
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Aggressive hand movement
    if pose_results.pose_landmarks:
        h, w, _ = frame.shape
        left_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        lw_pos = (int(left_wrist.x * w), int(left_wrist.y * h))
        rw_pos = (int(right_wrist.x * w), int(right_wrist.y * h))

        lw_speed = calc_speed(prev_left_wrist, lw_pos)
        rw_speed = calc_speed(prev_right_wrist, rw_pos)

        if lw_speed > HAND_SPEED_THRESHOLD or rw_speed > HAND_SPEED_THRESHOLD:
            risk_score += 30
            cv2.putText(frame, "AGGRESSIVE HAND MOVEMENT", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        prev_left_wrist, prev_right_wrist = lw_pos, rw_pos

    # Display score
    cv2.putText(frame, f"RISK SCORE: {risk_score}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Trigger alert
    if risk_score >= RISK_ALERT_THRESHOLD:
        cv2.putText(frame, "!!! ALERT !!!", (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        send_email(subject="Harassment Alert", body=f"Risk Score: {risk_score}\nAlert: The detected actions have exceeded the risk threshold.")

    cv2.imshow("Harassment Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
