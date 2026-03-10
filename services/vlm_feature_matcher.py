"""
VLM特征匹配模块 - 补充人脸识别
基于外貌和穿着特征，跨照片匹配人物
"""
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
from models import Photo
from utils import save_json, load_json
from config import FEATURE_DB_PATH


class VLMFeatureMatcher:
    """VLM特征匹配器"""

    def __init__(self):
        # 特征库：{person_id: {features, photo_count, appearances, clothing_list}}
        self.feature_db = {}
        self._load_feature_db()

    def _load_feature_db(self):
        """加载特征库"""
        data = load_json(FEATURE_DB_PATH)

        if data:
            self.feature_db = data
            print(f"加载VLM特征库：{len(self.feature_db)} 个人物")

    def _save_feature_db(self):
        """保存特征库"""
        save_json(self.feature_db, FEATURE_DB_PATH)

    def extract_features(self, vlm_result: Dict) -> List[Dict]:
        """
        从VLM结果中提取人物特征（结构化）

        Args:
            vlm_result: VLM分析结果

        Returns:
            人物特征列表
        """
        if not vlm_result or not vlm_result.get('people_details'):
            return []

        features = []
        for person in vlm_result['people_details']:
            # 提取所有优先级的特征到统一的key_features中
            key_features = {}

            # 优先级1（最稳定）
            key_features['gender'] = person.get('gender', '')
            key_features['approximate_age'] = person.get('approximate_age', '')
            key_features['body_size'] = person.get('body_size', '')
            key_features['main_clothing_color'] = person.get('main_clothing_color', '')
            key_features['position_in_scene'] = person.get('position_in_scene', '')

            # 优先级2（相对稳定）
            key_features['hair_color'] = person.get('hair_color', '')
            key_features['hair_length'] = person.get('hair_length', '')
            key_features['distinctive_clothing'] = person.get('distinctive_clothing', '')

            # 优先级3（近距离可见）
            key_features['face_shape'] = person.get('face_shape', '')
            key_features['glasses'] = person.get('glasses', '')
            key_features['skin_tone'] = person.get('skin_tone', '')

            # 构建clothing描述：优先使用distinctive_clothing，否则使用main_clothing_color
            clothing_desc = person.get('distinctive_clothing', '')
            if not clothing_desc or clothing_desc == '未详述':
                clothing_desc = person.get('main_clothing_color', '')
                if clothing_desc and clothing_desc != '未详述':
                    clothing_desc += '上衣'

            features.append({
                'person_id': person.get('person_id', ''),
                'key_features': key_features,
                'clothing': clothing_desc,
                'activity': person.get('activity', '')
            })

        return features

    def calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        计算两个VLM特征的相似度（基于粗粒度特征）

        Args:
            features1: 人物特征1
            features2: 人物特征2

        Returns:
            相似度（0-1）
        """
        key1 = features1.get('key_features', {})
        key2 = features2.get('key_features', {})

        score = 0.0

        # === 优先级1: 性别（权重0.3，最重要） ===
        gender1 = key1.get('gender', '')
        gender2 = key2.get('gender', '')

        if gender1 and gender2:
            if gender1 == gender2:
                score += 0.3
            else:
                # 性别不匹配 → 直接返回0
                return 0.0

        # === 优先级2: 体型（权重0.25，很稳定） ===
        body1 = key1.get('body_size', '')
        body2 = key2.get('body_size', '')

        if body1 and body2:
            if body1 == body2:
                score += 0.25
            elif body1 in ['苗条', '瘦小'] and body2 in ['苗条', '瘦小']:
                score += 0.15  # 部分匹配
            elif body1 in ['偏胖', '壮实'] and body2 in ['偏胖', '壮实']:
                score += 0.15  # 部分匹配

        # === 优先级3: 发色（权重0.15，相对稳定） ===
        hair_color1 = key1.get('hair_color', '')
        hair_color2 = key2.get('hair_color', '')

        if hair_color1 and hair_color2:
            if hair_color1 == hair_color2:
                score += 0.15
            elif hair_color1 == '黑色' and hair_color2 == '黑色':
                score += 0.15

        # === 优先级4: 主要衣服颜色（权重0.15，可变但有区分度） ===
        clothing_color1 = key1.get('main_clothing_color', '')
        clothing_color2 = key2.get('main_clothing_color', '')

        if clothing_color1 and clothing_color2:
            if clothing_color1 == clothing_color2:
                score += 0.15

        # === 优先级5: 年龄段（权重0.15，相对稳定） ===
        age1 = key1.get('approximate_age', '')
        age2 = key2.get('approximate_age', '')

        if age1 and age2:
            if age1 == age2:
                score += 0.15
            else:
                # 年龄段不匹配，扣分（跨度越大扣分越多）
                age_diff = abs(self._age_to_number(age1) - self._age_to_number(age2))
                if age_diff >= 2:
                    # 年龄跨度大（如青年vs老年），不太可能是同一个人
                    return max(0.0, score - 0.3)
                elif age_diff == 1:
                    # 年龄跨度小（如青年vs中年），稍微扣分
                    score = max(0.0, score - 0.1)

        # === 优先级6: 发长（权重0.05，会变但相对稳定） ===
        hair_len1 = key1.get('hair_length', '')
        hair_len2 = key2.get('hair_length', '')

        if hair_len1 and hair_len2:
            if hair_len1 == hair_len2:
                score += 0.05

        return score

    def _age_to_number(self, age_str: str) -> int:
        """将年龄段转换为数字用于比较"""
        age_map = {
            '儿童': 0,
            '少年': 1,
            '青年': 2,
            '中年': 3,
            '老年': 4
        }
        return age_map.get(age_str, -1)

    def _is_valid_features(self, vlm_features: Dict) -> bool:
        """
        检查VLM特征是否有效（防止强行匹配无效特征）

        Args:
            vlm_features: VLM特征

        Returns:
            是否有效
        """
        key_features = vlm_features.get('key_features', {})

        # 检查优先级1的特征，至少要有2个有效值
        priority_1_fields = ['gender', 'approximate_age', 'body_size', 'main_clothing_color']
        valid_count = 0

        for field in priority_1_fields:
            value = key_features.get(field, '')
            if value and value not in ['未知', '', '未详述']:
                valid_count += 1

        # 至少2个优先级1特征有效才算有效
        return valid_count >= 2

    def match_person(self, vlm_features: Dict, face_db: Dict) -> Tuple[Optional[str], float]:
        """
        根据VLM特征匹配已有人物

        Args:
            vlm_features: VLM提取的人物特征
            face_db: 人脸库

        Returns:
            (匹配的person_id, 相似度)
        """
        if not self.feature_db:
            return None, 0.0

        best_match = None
        best_similarity = 0.0

        # 遍历特征库中的每个人物
        for person_id, person_features in self.feature_db.items():
            # 用累计的特征（多张照片聚合）进行匹配
            aggregated_features = self._aggregate_features(person_features)

            similarity = self.calculate_similarity(vlm_features, aggregated_features)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_id

        # 相似度阈值：0.5（需要高置信度才能匹配）
        if best_similarity >= 0.5:
            return best_match, best_similarity

        return None, 0.0

    def _aggregate_features(self, person_features: Dict) -> Dict:
        """
        聚合同一个人的多次VLM特征（使用新特征结构）

        Args:
            person_features: 人物的所有特征记录

        Returns:
            聚合后的特征
        """
        # 聚合key_features
        all_key_features = person_features.get('all_key_features', [])

        if not all_key_features:
            return {}

        aggregated = {}

        # 对于每个字段，取最常见的值
        fields_to_aggregate = [
            'gender', 'approximate_age', 'body_size', 'main_clothing_color',
            'position_in_scene', 'hair_color', 'hair_length', 'distinctive_clothing',
            'face_shape', 'glasses', 'skin_tone'
        ]

        for field in fields_to_aggregate:
            values = [kf.get(field) for kf in all_key_features if kf.get(field)]
            if values:
                # 取最常见的值
                most_common = max(set(values), key=values.count)
                aggregated[field] = most_common

        return {
            'key_features': aggregated
        }

    def add_face_recognition_person(self, person_id: str, vlm_features: Dict, photo_id: str):
        """
        添加人脸识别的人物（有ID的）

        Args:
            person_id: 人脸识别分配的ID
            vlm_features: VLM特征（结构化）
            photo_id: 照片ID
        """
        if person_id not in self.feature_db:
            self.feature_db[person_id] = {
                'photo_count': 0,
                'all_key_features': [],  # 所有关键特征列表
                'clothing_list': [],
                'source': 'face_recognition',  # 标记来源：人脸识别
                'photo_ids': []
            }

        # 更新特征
        person_data = self.feature_db[person_id]

        # 检查是否已经在这张照片中添加过
        if photo_id not in person_data['photo_ids']:
            person_data['photo_count'] += 1
            person_data['photo_ids'].append(photo_id)
            # 新照片，才添加特征
            if vlm_features.get('key_features'):
                person_data['all_key_features'].append(vlm_features['key_features'])
            if vlm_features.get('clothing'):
                person_data['clothing_list'].append(vlm_features['clothing'])
            self._save_feature_db()
        # 如果照片已存在，不重复添加特征（避免同一张照片的多个人物记录重复）

    def _find_next_available_id(self) -> str:
        """找到下一个可用的person_id"""
        # 解析所有已存在的ID数字
        existing_numbers = set()
        for person_id in self.feature_db.keys():
            if person_id.startswith('person_'):
                try:
                    num = int(person_id.split('_')[1])
                    existing_numbers.add(num)
                except (ValueError, IndexError):
                    pass

        # 找到最小的可用数字
        next_num = 0
        while next_num in existing_numbers:
            next_num += 1

        return f"person_{next_num}"

    def add_vlm_only_person(self, vlm_features: Dict, photo_id: str, existing_ids_in_photo: List[str] = None) -> str:
        """
        添加纯VLM识别的人物（无人脸识别ID）

        Args:
            vlm_features: VLM特征（结构化）
            photo_id: 照片ID
            existing_ids_in_photo: 当前照片中已经存在的人物ID列表

        Returns:
            分配的person_id
        """
        if existing_ids_in_photo is None:
            existing_ids_in_photo = []

        # 尝试匹配已有人物
        matched_id, similarity = self.match_person(vlm_features, {})

        if matched_id:
            # 检查这个ID是否已经在当前照片中出现了
            if matched_id in existing_ids_in_photo:
                # 同一张照片里不应该有两个人是同一个ID（除非照镜子）
                # 创建新人物
                new_id = self._find_next_available_id()
                self.feature_db[new_id] = {
                    'photo_count': 1,
                    'all_key_features': [vlm_features.get('key_features', {})],
                    'clothing_list': [vlm_features.get('clothing', '')],
                    'source': 'vlm_only',
                    'photo_ids': [photo_id]
                }
                self._save_feature_db()
                print(f"  VLM特征匹配：{matched_id}已在此照片中，创建新人物 {new_id}")
                return new_id
            else:
                # 匹配成功，归入已有人物
                self.add_face_recognition_person(matched_id, vlm_features, photo_id)
                print(f"  VLM特征匹配：归入 {matched_id} (相似度: {similarity:.2f})")
                return matched_id
        else:
            # 创建新人物 - 使用下一个可用ID
            new_id = self._find_next_available_id()
            self.feature_db[new_id] = {
                'photo_count': 1,
                'all_key_features': [vlm_features.get('key_features', {})],
                'clothing_list': [vlm_features.get('clothing', '')],
                'source': 'vlm_only',  # 标记来源：纯VLM识别
                'photo_ids': [photo_id]
            }
            self._save_feature_db()
            print(f"  VLM特征匹配：创建新人物 {new_id}")
            return new_id

    def process_photo_vlm_features(self, photo: Photo, face_db: Dict):
        """
        处理单张照片的VLM特征，匹配人物

        优先级：
        1. 人脸识别检测到的人 -> 使用人脸识别的ID，VLM只补充特征
        2. 人脸识别没检测到，但VLM识别到且特征清晰 -> VLM特征匹配
        3. 人脸识别没检测到，VLM特征不清晰 -> 跳过

        重要：同一张照片里的不同人物不能归入同一个ID（除非照镜子等特殊情况）

        Args:
            photo: 照片对象
            face_db: 人脸库
        """
        if not photo.vlm_analysis:
            return

        # 提取VLM识别的人物特征
        vlm_features_list = self.extract_features(photo.vlm_analysis)

        if not vlm_features_list:
            return

        print(f"  照片 {photo.photo_id}: VLM识别 {len(vlm_features_list)} 个人")

        # 获取当前照片已经存在的人物ID
        existing_ids = [f['person_id'] for f in photo.faces]

        # 情况1: 人脸识别检测到了人
        if len(photo.faces) > 0:
            print(f"  人脸识别检测到 {len(photo.faces)} 人，优先使用人脸识别ID")
            # 按顺序对应：VLM的第i个人对应人脸识别的第i个人
            for i, vlm_features in enumerate(vlm_features_list):
                if i < len(photo.faces):
                    face_person_id = photo.faces[i]['person_id']
                    print(f"    [{i}] VLM特征归入人脸识别的 {face_person_id}")
                    # 将VLM特征补充到这个人脸识别的人物
                    self.add_face_recognition_person(face_person_id, vlm_features, photo.photo_id)
                    # 加入已存在ID列表
                    existing_ids.append(face_person_id)
                else:
                    # VLM识别的人比人脸识别多，尝试特征匹配（但特征必须清晰）
                    if self._is_valid_features(vlm_features):
                        print(f"    [{i}] VLM识别到更多人，尝试特征匹配...")
                        matched_id = self.add_vlm_only_person(vlm_features, photo.photo_id, existing_ids)
                        if matched_id and matched_id not in [f['person_id'] for f in photo.faces]:
                            photo.faces.append({
                                'person_id': matched_id,
                                'confidence': 0.0,
                                'source': 'vlm_match',
                                'bbox': {}
                            })
                            existing_ids.append(matched_id)
                    else:
                        print(f"    [{i}] VLM特征不清晰，跳过")
        else:
            # 情况2: 人脸识别没有检测到人（都太小、被过滤或遮挡）
            # 策略：只有当VLM识别≤2人时才尝试跨照片匹配
            # 如果≥3人（聚会、大合影），人物太模糊不匹配，避免错误归并
            vlm_count = len(vlm_features_list)

            if vlm_count > 3:
                print(f"  VLM识别{vlm_count}人（聚会/大合影场景），人物较模糊，跳过跨照片匹配")
                # 不匹配，直接跳过
                return
            else:
                print(f"  VLM识别{vlm_count}人，尝试跨照片匹配")
                # VLM识别到的每个人都尝试匹配，但特征必须清晰
                for i, vlm_features in enumerate(vlm_features_list):
                    if not self._is_valid_features(vlm_features):
                        print(f"    [{i}] VLM特征不清晰，跳过")
                        continue

                    print(f"    [{i}] VLM特征清晰，尝试匹配...")
                    matched_id = self.add_vlm_only_person(vlm_features, photo.photo_id, existing_ids)

                    if matched_id and matched_id not in [f['person_id'] for f in photo.faces]:
                        photo.faces.append({
                            'person_id': matched_id,
                            'confidence': 0.0,
                            'source': 'vlm_match',
                            'bbox': {}
                        })
                        existing_ids.append(matched_id)

    def reconcile_protagonist(self, face_db: Dict) -> str:
        """
        识别主角：结合人脸识别和VLM特征库

        Args:
            face_db: 人脸库

        Returns:
            主角的person_id
        """
        if not self.feature_db:
            return None

        # 统计每个人物的出现次数（包括VLM补充的）
        person_counts = {}

        # 来自人脸识别
        for person_id, person in face_db.items():
            person_counts[person_id] = person.photo_count

        # 来自VLM特征库（可能更多）
        for person_id, features in self.feature_db.items():
            if person_id in person_counts:
                # 取最大值（避免重复计数）
                person_counts[person_id] = max(person_counts[person_id], features['photo_count'])
            else:
                person_counts[person_id] = features['photo_count']

        if not person_counts:
            return None

        # 找出现次数最多的
        protagonist = max(person_counts, key=person_counts.get)

        print(f"\n主角识别：{protagonist}（出现 {person_counts[protagonist]} 次）")

        return protagonist

    def get_feature_db(self) -> Dict:
        """获取特征库"""
        return self.feature_db

    def save(self):
        """保存特征库"""
        self._save_feature_db()
