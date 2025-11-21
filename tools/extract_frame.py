"""Extract frame from MP4 video."""
import cv2

# Open video
video = cv2.VideoCapture('/tmp/rl_rollout.mp4')

# Read first frame
ret, frame = video.read()
if ret:
    # Save frame
    cv2.imwrite('/home/gong-zerui/code/ResNav/map_frame0.png', frame)
    print(f"Saved frame: {frame.shape}")
else:
    print("Failed to read frame")

video.release()
