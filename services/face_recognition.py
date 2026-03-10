"""
人脸识别模块
"""
import os
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from deepface import DeepFace
from models import Photo, Person
from config import (
    FACE_THRESHOLD, FACE_MODEL, FACE_DETECTOR, FACE_ALIGN, FACE_MIN_SIZE,
    FACE_DB_PATH, CACHE_DIR
)
from utils import cosine_similarity, save_json, load_json


class FaceRecognition:
    """人脸识别器"""

    def __init__(self):
        self.threshold = FACE_THRESHOLD
        self.face_db = {}  # {person_id: Person对象}

        # 尝试加载已有的人脸库
        self._load_face_db()

    def _load_face_db(self):
        """从文件加载人脸库"""
        data = load_json(FACE_DB_PATH)

        if data:
            for person_id, person_data in data.items():
                # 恢复特征向量
                features = [np.array(f) for f in person_data.get("features", [])]

                self.face_db[person_id] = Person(
                    person_id=person_id,
                    name=person_data.get("name", "未知"),
                    features=features,
                    photo_count=person_data.get("photo_count", 0),
                    first_seen=datetime.fromisoformat(person_data["first_seen"]) if person_data.get("first_seen") else None,
                    last_seen=datetime.fromisoformat(person_data["last_seen"]) if person_data.get("last_seen") else None,
                    avg_confidence=person_data.get("avg_confidence", 0.0)
                )

            print(f"加载人脸库：{len(self.face_db)} 个人物")

    def _save_face_db(self):
        """保存人脸库到文件"""
        data = {}

        for person_id, person in self.face_db.items():
            # 转换特征向量为列表（便于JSON序列化）
            features = []
            for f in person.features:
                if isinstance(f, np.ndarray):
                    features.append(f.tolist())
                else:
                    features.append(f)  # 已经是list

            data[person_id] = {
                "name": person.name,
                "features": features,
                "photo_count": person.photo_count,
                "first_seen": person.first_seen.isoformat() if person.first_seen else None,
                "last_seen": person.last_seen.isoformat() if person.last_seen else None,
                "avg_confidence": person.avg_confidence
            }

        save_json(data, FACE_DB_PATH)

    def process_photo(self, photo: Photo) -> List[Dict]:
        """
        处理单张照片，识别人脸

        Args:
            photo: 照片对象

        Returns:
            人脸列表：[{"person_id": "person_0", "confidence": 0.95, "bbox": [...]}]
        """
        # 优先使用原图，如果没有压缩图则用原图
        image_path = photo.compressed_path if photo.compressed_path else photo.path

        try:
            # Step 1: 检测人脸
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=FACE_DETECTOR,
                enforce_detection=False
            )

            # Step 2: 提取所有人脸特征（只调用一次）
            try:
                embedding_objs = DeepFace.represent(
                    img_path=image_path,
                    model_name=FACE_MODEL,
                    detector_backend=FACE_DETECTOR,
                    enforce_detection=False
                )
            except Exception as e:
                print(f"  调试：特征提取失败 - {e}")
                return []

            if not embedding_objs or len(embedding_objs) == 0:
                return []

            results = []

            # Step 3: 对每个检测到的人脸，找到对应的embedding并匹配
            for i, face in enumerate(faces):
                # 过滤低置信度的人脸（假人脸）
                face_confidence = face.get("confidence", 0)
                if face_confidence < 0.5:
                    continue  # 跳过低置信度的人脸

                # 过滤小脸（人脸质量过滤）
                facial_area = face.get("facial_area", {})
                face_width = facial_area.get("w", 0)
                face_height = facial_area.get("h", 0)
                min_dim = min(face_width, face_height)

                if min_dim < FACE_MIN_SIZE:
                    print(f"  调试：跳过小脸 ({min_dim}x{min_dim} < {FACE_MIN_SIZE}x{FACE_MIN_SIZE})")
                    continue

                # 使用对应索引的embedding（detector返回的顺序应该一致）
                if i >= len(embedding_objs):
                    print(f"  调试：embedding索引越界，跳过人脸 {i}")
                    continue

                embedding = embedding_objs[i].get("embedding")

                if embedding is None:
                    print(f"  调试：embedding为空")
                    continue

                # 匹配或创建人物
                person_id, match_confidence = self._match_or_create_person(embedding, photo.timestamp)

                results.append({
                    "person_id": person_id,
                    "confidence": match_confidence,
                    "bbox": face.get("facial_area", {})
                })

                print(f"  调试：识别人脸 -> {person_id} (匹配度: {match_confidence:.2f}, 大小: {min_dim}x{min_dim})")

            # 保存到照片对象
            photo.faces = results

            return results

        except Exception as e:
            print(f"警告：人脸识别失败 ({photo.filename}): {e}")
            return []

    def _match_or_create_person(self, embedding: np.ndarray, timestamp: datetime) -> Tuple[str, float]:
        """
        匹配或创建人物

        Args:
            embedding: 人脸特征向量
            timestamp: 照片时间戳

        Returns:
            (person_id, confidence)
        """
        # 尝试匹配已有人物
        best_match = None
        best_similarity = 0

        for person_id, person in self.face_db.items():
            # 计算和每个人物平均特征的相似度
            if person.features:
                avg_feature = np.mean(person.features, axis=0)
                similarity = cosine_similarity(embedding, avg_feature)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_id

        # 判断是否匹配成功
        if best_match and best_similarity >= self.threshold:
            # 匹配成功，更新特征
            person = self.face_db[best_match]
            person.features.append(embedding)

            # 更新统计信息
            person.photo_count += 1
            person.last_seen = timestamp
            person.avg_confidence = (person.avg_confidence * (person.photo_count - 1) + best_similarity) / person.photo_count

            # 如果特征太多，保留最近的
            if len(person.features) > 10:
                person.features = person.features[-10:]

            return best_match, best_similarity

        else:
            # 创建新人物（从person_1开始编号）
            person_id = f"person_{len(self.face_db) + 1}"

            # 判断是否是主角（出现次数最多的）
            name = "未知"
            if not self.face_db:
                name = "主角"  # 第一个人(person_0)是主角
            else:
                # 找出现次数最多的
                max_count = max(p.photo_count for p in self.face_db.values())
                if max_count < 10:
                    name = "未知"
                else:
                    name = "未知"

            new_person = Person(
                person_id=person_id,
                name=name,
                features=[embedding],
                photo_count=1,
                first_seen=timestamp,
                last_seen=timestamp,
                avg_confidence=best_similarity
            )

            self.face_db[person_id] = new_person

            return person_id, best_similarity

    def update_names(self):
        """
        更新人物名称：
        - 出现次数最多（>=2次）的是"主角"
        - 其他是"未知"
        """
        if not self.face_db:
            return

        # 找出现次数最多的
        max_person = max(self.face_db.values(), key=lambda p: p.photo_count)
        max_count = max_person.photo_count

        # 更新名称：只有出现>=2次的才标记为主角
        for person_id, person in self.face_db.items():
            if person.photo_count == max_count and max_count >= 2:
                person.name = "主角"
            else:
                person.name = "未知"

    def reorder_protagonist(self, photos: list) -> dict:
        """
        重排person_id：让主角变为person_0，其他人按出现次数排序为P1, P2, P3...

        Args:
            photos: 照片列表

        Returns:
            id_mapping: 旧ID → 新ID 的映射，例如 {"person_1": "person_0", "person_2": "person_1"}
        """
        if not self.face_db:
            return {}

        # 1. 所有人按出现次数排序
        sorted_persons = sorted(
            self.face_db.items(),
            key=lambda x: x[1].photo_count,
            reverse=True
        )

        # 2. 创建映射：最多次→P0，其次→P1，P2...
        id_mapping = {}
        for new_id, (old_id, _) in enumerate(sorted_persons):
            id_mapping[old_id] = f"person_{new_id}"

        # 3. 更新人脸库
        new_face_db = {}
        for old_id, person in self.face_db.items():
            new_id = id_mapping[old_id]
            person.person_id = new_id
            new_face_db[new_id] = person
        self.face_db = new_face_db

        # 4. 更新所有照片的faces字段
        for photo in photos:
            for face in photo.faces:
                old_id = face["person_id"]
                if old_id in id_mapping:
                    face["person_id"] = id_mapping[old_id]

        # 5. 更新名称（主角标记）
        self.update_names()

        # 6. 保存
        self._save_face_db()

        # 打印映射结果
        print(f"  ID重排映射: {id_mapping}")

        return id_mapping

    def save(self):
        """保存人脸库"""
        self._save_face_db()

    def get_person(self, person_id: str) -> Optional[Person]:
        """获取人物信息"""
        return self.face_db.get(person_id)

    def get_all_persons(self) -> Dict[str, Person]:
        """获取所有人物"""
        return self.face_db
