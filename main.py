import cv2
from PoseModule import PoseDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.find_pose(img)
        landmarks = detector.get_landmarks(img)

        # Draw bounding box around the person
        img = detector.draw_bounding_box(img, landmarks)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
