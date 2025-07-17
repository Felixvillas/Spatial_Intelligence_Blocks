import cv2

img = cv2.imread('./n-0-1-mrtview_45_image.png')
clone = img.copy()

def click(event, x, y, flags, param):
    # 当鼠标左键按下时
    if event == cv2.EVENT_LBUTTONDOWN:
        # 终端打印鼠标点击的位置
        print('(x, y) =', x, y)          # 终端打印
        # 在鼠标点击的位置画一个半径为3的红色圆
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        # 显示图片
        cv2.imshow('image', img)

cv2.namedWindow('image')
cv2.setMouseCallback('image', click)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
