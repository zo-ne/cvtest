from deepface import DeepFace
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
import cv2

# PostgreSQL 連線設定
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "089453",
    "dbname": "opencv",
}

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

conn = psycopg2.connect(**DB_CONFIG)
register_vector(conn)
cur = conn.cursor()

# 建立資料表（如果不存在）
cur.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        embedding vector(512)
    );
    TRUNCATE face_embeddings;
""")
conn.commit()

# 產生並寫入 embedding
for img_path, name in zip(imgs, names):
    result = DeepFace.represent(img_path, model_name="Facenet512")
    emb = np.array(result[0]["embedding"], dtype="float32")
    cur.execute("INSERT INTO face_embeddings (name, embedding) VALUES (%s, %s)", (name, emb))

conn.commit()
print(f"已寫入 {len(imgs)} 筆 embedding")

# 查詢最相近的人臉（L2 距離）
source_emb = DeepFace.represent(source_img, model_name="Facenet512")
query_vec = np.array(source_emb[0]["embedding"], dtype="float32")

cur.execute("""
    SELECT name, embedding <-> %s AS distance
    FROM face_embeddings
    ORDER BY distance
    LIMIT 1
""", (query_vec,))

row = cur.fetchone()
name, distance = row
print(f"最相近: {name}，距離: {distance:.4f}")

cur.close()
conn.close()

# 顯示結果
img = cv2.imread(source_img)
cv2.putText(img, name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

