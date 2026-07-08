from deepface import DeepFace
import numpy as np
import faiss

imgs = [
    "src/face/database/iu/image1.jpg",
    "src/face/database/tomcruise/image1.jpg",
    "src/face/database/eun-bin/image1.jpg",
]
names = [
    "iu",
    "tomcruise",
    "eun-bin",
]
source_img = "src/face/find_test2.jpg"

emb = []
for img in imgs:
    result = DeepFace.represent(img, model_name="Facenet512")
    emb.append(result[0]["embedding"])

dimension = 512  # Facenet512 embedding 向量維度
index = faiss.IndexFlatL2(dimension)
index.add(np.array(emb).astype("float32"))

print("索引庫準備完成！")
print(f"索引庫大小: {index.ntotal}")
print(f"索引庫維度: {index.d}")

source_emb = DeepFace.represent(source_img, model_name="Facenet512")
D, I = index.search(np.array([source_emb[0]["embedding"]]).astype("float32"), k=1)  # 取最相近的1筆
name = names[I[0][0]]

# 新增：顯示圖片並標註名稱
import cv2

img = cv2.imread(source_img)
cv2.putText(img, name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

