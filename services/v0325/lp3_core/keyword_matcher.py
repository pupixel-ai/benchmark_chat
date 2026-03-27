"""
关键词匹配模块 - 统一的基于列表的候选提取逻辑

替代原来的正则表达式模式匹配，支持：
1. 已知品牌/地点/主题的直接匹配
2. 模糊匹配（支持中英文混合）
3. 黑名单过滤
"""
from typing import List, Set, Tuple, Dict, Any
import re


class BrandMatcher:
    """品牌识别器 - 基于已知品牌列表"""

    # 已知品牌列表（按类别组织）
    KNOWN_BRANDS = {
        '服装鞋帽': [
            'Zara', 'H&M', 'Uniqlo', 'Gap', 'Forever 21', 'ASOS',
            '森马', '美邦', '太平鸟', 'Pinko', 'KANS',
            'Adidas', 'Nike', '阿迪达斯', '耐克',
            'Puma', 'New Balance', 'Converse',
        ],
        '消费品': [
            'Hello Kitty', '蒙牛', '伊利', '花呗', '支付宝', '微信',
            '淘宝', '美团', '饿了么', '星巴克', 'Starbucks',
            '瑞幸', '喜茶', 'Costa', '肯德基', 'KFC',
        ],
        '电子产品': [
            'iPhone', 'iPad', 'MacBook', 'Apple', 'Samsung',
            '华为', 'OPPO', 'vivo', 'OnePlus', 'Google', 'Pixel',
            'PlayStation', 'Xbox', 'Switch',
        ],
        '家居日用': [
            'MUJI', '宜家', 'IKEA', '万利达', '美的', 'Philips',
            '飞利浦', 'Dyson', '戴森',
        ],
        '烟酒': [
            '黄金叶', '黄鹤楼', '南京', '中南海', '硬中华',
            '青岛', '燕京', '百威', 'Corona', 'Heineken',
        ],
        '化妆护肤': [
            '兰蔻', '迪奥', 'Dior', 'Chanel', '香奈儿',
            '雪花秀', '后', 'SK-II', '资生堂', '欧莱雅',
        ],
        '其他': [
            'Chrome Hearts', 'Chrom Hearts', 'Louis Vuitton', 'LV', 'Gucci', 'Prada',
        ]
    }

    # 品牌黑名单（常见的非品牌词）
    BRAND_BLACKLIST = {
        'hello', 'brand', 'logo', '品牌', '图案', '标识',
        '衣物', '物品', '东西', '东西', '材质', '配件',
    }

    # 品牌产品词汇（用于上下文验证）
    BRAND_CONTEXT_KEYWORDS = {
        '服装': ['卫衣', '衣服', '衣物', 'shirt', 't-shirt', 'hoodie', '裙子', '裤子', '外套', '夹克'],
        '鞋': ['鞋', 'shoe', '凉拖', '运动鞋', '靴子'],
        '包': ['包', 'bag', '手提', '背包', '钱包'],
        '配饰': ['帽子', '围巾', '手套', '耳机', '手表', '眼镜', '项链'],
        '食饮': ['咖啡', '奶茶', '饮料', '饼干', '巧克力', '葡萄酒'],
        '电子': ['手机', 'phone', '电脑', 'computer', '耳机', '平板'],
        '化妆': ['口红', 'lipstick', '眼影', '粉底', '护肤', 'skincare', '化妆', 'cosmetic'],
    }

    WATERMARK_HINTS = (
        'shot on',
        'watermark',
        'device watermark',
        'camera watermark',
        'phone watermark',
        '水印',
        '设备水印',
        '手机水印',
        '相机水印',
    )

    def __init__(self):
        # 构建平坦的品牌列表和小写索引
        self.all_brands = []
        self.brand_lookup_lower = {}  # 小写 → 原始名称映射

        for category, brands in self.KNOWN_BRANDS.items():
            for brand in brands:
                self.all_brands.append(brand)
                self.brand_lookup_lower[brand.lower()] = brand

        self.all_brands_lower = {b.lower() for b in self.all_brands}

    def extract_brands(self, text: str) -> Tuple[List[str], List[str]]:
        """
        从文本中提取品牌名

        Returns:
            (candidates, rejected) - 识别的品牌列表和被拒绝的候选词
        """
        if not text:
            return [], []

        candidates = []
        rejected = []

        # 方法 1: 直接匹配已知品牌（完全 + 模糊）
        found_brands = self._direct_brand_match(text)
        candidates.extend(found_brands)

        # 方法 2: 从正则表达式模式中提取（保留原来的逻辑作为补充）
        # 这样可以识别新的品牌名（不在列表中的）
        regex_brands, regex_rejected = self._regex_brand_extraction(text)
        for brand in regex_brands:
            if brand.lower() not in {c.lower() for c in candidates}:
                candidates.append(brand)
        rejected.extend(regex_rejected)

        # 方法 3: 从设备/相机水印里宽松提取品牌样式 token，交给下游 ownership + LLM 判断
        watermark_brands = self._watermark_brand_extraction(text)
        for brand in watermark_brands:
            if brand.lower() not in {c.lower() for c in candidates}:
                candidates.append(brand)

        # 方法 4: 过滤黑名单
        filtered = [b for b in candidates if b.lower() not in self.BRAND_BLACKLIST]
        filtered = list(dict.fromkeys(filtered))  # 去重并保留顺序

        return filtered, rejected

    def _direct_brand_match(self, text: str) -> List[str]:
        """直接从文本中匹配已知品牌列表"""
        matched = []
        text_lower = text.lower()

        # 按长度排序（长的优先，避免部分匹配问题）
        sorted_brands = sorted(self.all_brands, key=len, reverse=True)

        for brand in sorted_brands:
            brand_lower = brand.lower()
            # 简化匹配：直接子串搜索
            if brand_lower not in text_lower:
                continue

            if self._contains_cjk(brand):
                original = self.brand_lookup_lower.get(brand_lower, brand)
                matched.append(original)
                continue

            pattern = re.compile(rf"(?<![a-z0-9]){re.escape(brand_lower)}(?![a-z0-9])")
            if pattern.search(text_lower):
                original = self.brand_lookup_lower.get(brand_lower, brand)
                matched.append(original)

        return matched

    def _regex_brand_extraction(self, text: str) -> Tuple[List[str], List[str]]:
        """使用原来的正则表达式模式提取品牌（保留为兜底方案）"""
        # 这里可以调用原来的 _extract_brand_candidates_from_text 逻辑
        # 或者简化版本
        candidates = []
        rejected = []

        # 寻找 "品牌名 + 产品词汇" 的模式
        product_keywords = '|'.join(
            kw for keywords in self.BRAND_CONTEXT_KEYWORDS.values()
            for kw in keywords
        )

        # 匹配 "Brandname productword" 或 "品牌名 产品词"
        pattern = rf'([A-Z][A-Za-z0-9&\-]*(?:\s+[A-Z][A-Za-z0-9&\-]*)?)\s+(?:{product_keywords})'
        pattern_cjk = rf'([\u4e00-\u9fff]{{2,6}})\s+(?:{product_keywords})'

        for match in re.finditer(f'({pattern})|({pattern_cjk})', text):
            candidate = match.group(1) or match.group(2)
            if candidate:
                candidate = candidate.strip()
                if candidate.lower() not in self.BRAND_BLACKLIST:
                    candidates.append(candidate)

        return list(dict.fromkeys(candidates)), rejected

    def _watermark_brand_extraction(self, text: str) -> List[str]:
        normalized = str(text or "")
        lowered = normalized.lower()
        watermark_pattern = re.compile(
            r"(?<![A-Za-z0-9])([A-Z][A-Z0-9&+\-]{1,24})(?=\s+[A-Za-z0-9][A-Za-z0-9+&+\-]{1,24})"
        )
        model_pattern = re.compile(
            r"(?<![A-Za-z0-9])([A-Z][A-Z0-9&+\-]{1,24})(?=\s+[A-Za-z]*\d[A-Za-z0-9+&+\-]{0,24})"
        )
        matches: List[str] = []
        if any(hint in lowered for hint in self.WATERMARK_HINTS):
            matches.extend(match.group(1).strip() for match in watermark_pattern.finditer(normalized))
        matches.extend(match.group(1).strip() for match in model_pattern.finditer(normalized))
        return list(dict.fromkeys(matches))

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


class LocationMatcher:
    """地点识别器 - 基于已知地点列表"""

    KNOWN_LOCATIONS = {
        '城市': [
            '北京', '上海', '广州', '深圳', '杭州', '南京', '苏州',
            '成都', '武汉', '西安', '重庆', '天津',
            'Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Hangzhou',
            'New York', 'Los Angeles', 'London', 'Paris', 'Tokyo', 'Seoul',
        ],
        '地标': [
            '故宫', '长城', '天安门', '埃菲尔铁塔', '自由女神像',
            '商业街', '购物中心', '公园', '咖啡厅', '餐厅', '学校', '办公室',
        ],
        '场景': [
            '家', '宿舍', '教室', '办公室', '咖啡厅', '餐厅', '公园',
            '海滩', '山顶', '街道', '商场', '车站', '机场',
        ]
    }

    LOCATION_BLACKLIST = {
        '地点', '位置', '地方', '环境', '场景', '背景',
        'location', 'place', 'scene',
    }

    def __init__(self):
        self.all_locations = []
        for category, locations in self.KNOWN_LOCATIONS.items():
            self.all_locations.extend(locations)
        self.all_locations_lower = {l.lower() for l in self.all_locations}

    def extract_locations(self, text: str) -> List[str]:
        """从文本中提取已知地点"""
        if not text:
            return []

        matched = []
        text_lower = text.lower()

        sorted_locations = sorted(self.all_locations, key=len, reverse=True)

        for location in sorted_locations:
            loc_lower = location.lower()
            if re.search(rf'\b{re.escape(loc_lower)}\b', text_lower):
                if loc_lower not in self.LOCATION_BLACKLIST:
                    matched.append(location)

        return list(dict.fromkeys(matched))


class InterestMatcher:
    """兴趣识别器 - 基于已知兴趣/活动列表"""

    KNOWN_INTERESTS = {
        '运动': [
            '跑步', '健身', '游泳', '瑜伽', '篮球', '足球', '网球',
            'running', 'fitness', 'swimming', 'yoga', 'basketball',
        ],
        '娱乐': [
            '看电影', '追剧', '玩游戏', '看书', '听音乐', '唱歌',
            'watching movies', 'gaming', 'reading', 'music', 'singing',
        ],
        '社交': [
            '聚会', '旅游', '逛街', '吃饭', '喝茶', '聊天',
            'party', 'travel', 'shopping', 'dining', 'tea time',
        ],
        '创意': [
            '摄影', '绘画', '手工', '写作', '弹琴', '舞蹈',
            'photography', 'painting', 'crafting', 'writing', 'music', 'dancing',
        ],
    }

    INTEREST_BLACKLIST = {
        '活动', '兴趣', '爱好', '事件', '事情', '事',
        'activity', 'interest', 'hobby', 'event',
    }

    def __init__(self):
        self.all_interests = []
        for category, interests in self.KNOWN_INTERESTS.items():
            self.all_interests.extend(interests)
        self.all_interests_lower = {i.lower() for i in self.all_interests}

    def extract_interests(self, text: str) -> List[str]:
        """从文本中提取已知兴趣"""
        if not text:
            return []

        matched = []
        text_lower = text.lower()

        sorted_interests = sorted(self.all_interests, key=len, reverse=True)

        for interest in sorted_interests:
            int_lower = interest.lower()
            if re.search(rf'\b{re.escape(int_lower)}\b', text_lower):
                if int_lower not in self.INTEREST_BLACKLIST:
                    matched.append(interest)

        return list(dict.fromkeys(matched))


# 全局实例
_brand_matcher = None
_location_matcher = None
_interest_matcher = None


def get_brand_matcher() -> BrandMatcher:
    global _brand_matcher
    if _brand_matcher is None:
        _brand_matcher = BrandMatcher()
    return _brand_matcher


def get_location_matcher() -> LocationMatcher:
    global _location_matcher
    if _location_matcher is None:
        _location_matcher = LocationMatcher()
    return _location_matcher


def get_interest_matcher() -> InterestMatcher:
    global _interest_matcher
    if _interest_matcher is None:
        _interest_matcher = InterestMatcher()
    return _interest_matcher
