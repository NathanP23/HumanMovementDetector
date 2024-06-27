import cv2
from PoseModule import PoseDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        # Detect humans using OpenCV's pre-trained model
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        humans = human_cascade.detectMultiScale(gray, 1.1, 4)

        for i, (x, y, w, h) in enumerate(humans):
            human_img = img[y:y+h, x:x+w]
            human_img = detector.find_pose(human_img)
            landmarks = detector.get_landmarks(human_img)

            # Draw bounding box around the person and label
            if landmarks:
                img = detector.draw_bounding_box(img, landmarks, (x, y, w, h), label_index=i+1)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
