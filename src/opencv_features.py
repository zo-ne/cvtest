import cv2

image = cv2.imread('src/data/nailong.jpg', 0)
feature_sift = cv2.xfeatures2d.SIFT_create()
# feature_surf = cv2.xfeatures2d.SURF_create()
feature_orb = cv2.ORB_create(50)

image_sift = cv2.drawKeypoints(
    image, feature_sift.detect(image), None, 
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
# image_surf = cv2.drawKeypoints(
#     image, feature_surf.detect(image), None, 
#     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )
image_orb = cv2.drawKeypoints(
    image, feature_orb.detect(image), None, 
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
image = cv2.hconcat([image_sift, image_orb])
cv2.imshow('image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
