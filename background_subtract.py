import numpy as np
import cv2
import os

# 建立資料夾
if not os.path.exists("./frames"):
    os.mkdir("./frames")

if not os.path.exists("./videos"):
    os.mkdir("./videos")

# 輸入影片名稱
video_name = input("請輸入影片名稱：")

# 讀取影片
capture = cv2.VideoCapture(video_name)

# 影片寬高
FRAME_WIDTH = int(capture.get(3))
FRAME_HEIGHT = int(capture.get(4))
# FPS
FPS = 30

# 寫出影片
writer = cv2.VideoWriter(
    "./videos/result.avi",
    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
    FPS,
    (FRAME_WIDTH, FRAME_HEIGHT),
)

# 紀錄目前為第幾個frame
count = 1

# 隨機選取100個frame
rng = np.random.default_rng(seed=42)
frame_counts = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
random_frame_ids = rng.choice(frame_counts, size=frame_counts // 3, replace=False)

frames = []
for random_frame_id in random_frame_ids:
    # 跳到指定的frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(random_frame_id))
    # 讀取該frame
    _, frame = capture.read()
    if frame is None:
        break
    frames.append(frame)

# 找出中位數(當作背景)
median_frame = np.median(frames, axis=0).astype(np.uint8)
cv2.imwrite("median.png", median_frame)

# 跳回第0個frame
capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
# 灰階化
gray_median_frame = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)


# 迴圈是否繼續
while True:
    # 讀取frame
    success, frame = capture.read()

    # 如果frame不是None
    if success:
        # 寫出
        cv2.imwrite(f"./frames/frame_{count}.jpg", frame)
        count += 1

        # 灰階化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 取目前的frame與中位數(背景)的差的絕對值
        difference = cv2.absdiff(gray, gray_median_frame)

        # 二值化
        _, difference = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)

        # 先侵蝕(黑的部分放大，亮的部分縮小)再膨脹(黑的部分縮小，亮的部分放大)
        kernel = np.ones((2, 2), dtype=np.uint8)
        opening = cv2.morphologyEx(difference, cv2.MORPH_OPEN, kernel)

        # 將相近的白點連接(才不會一個輪廓內有很多個小輪廓)
        kernel = np.ones((6, 6), np.uint8)
        dilate = cv2.dilate(opening, kernel, iterations=3)

        # 透過模糊讓圖片更平滑
        blur = cv2.blur(dilate, (5, 5))

        # 透過OTSU找尋閾值
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 找出前景輪廓
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 畫出輪廓
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)

        # 寫出frame
        writer.write(frame)

        # 按q退出(沒有用waitKey的話imshow不會顯示東西)
        if cv2.waitKey(20) == ord("q"):
            break

        # 顯示frame
        cv2.imshow("foreground", opening)
        cv2.imshow("result", frame)
    else:
        break

capture.release()
writer.release()
cv2.destroyAllWindows()
