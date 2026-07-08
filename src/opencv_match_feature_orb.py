import cv2

imgs = ['src/data/nailong.jpg', 'src/data/nailong2.jpg']
# imgs = ['src/data/nailong2.jpg', 'src/data/nailong3.jpg']
# imgs = ['src/data/lulumi.jpg', 'src/data/nailong2.jpg']
# imgs = ['src/data/duck.jpg', 'src/data/chair.jpg']
# imgs = ['src/data/zebra.jpg', 'src/data/zebra2.jpg']

img1 = cv2.imread(imgs[0])
img2 = cv2.imread(imgs[1])

orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x:x.distance)
print('Matching points :',len(matches))

img3 = cv2.drawMatches(
    img1, kp1, img2, kp2,
    matches[:10],
    outImg=None,
    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
)
# 計算 matches[:10] 的平均距離
avg_distance = sum(match.distance for match in matches[:10]) / 10 
print(f"Average distance: {avg_distance}")

width, height, channel = img3.shape
ratio = float(width) / float(height)
img3 = cv2.resize(img3, (1024, int(1024 * ratio)))
cv2.imshow('frame', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
