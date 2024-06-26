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

        # Display landmarks on image if needed
        for lm in landmarks:
            cv2.circle(img, (lm[1], lm[2]), 5, (255, 0, 0), cv2.FILLED)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
