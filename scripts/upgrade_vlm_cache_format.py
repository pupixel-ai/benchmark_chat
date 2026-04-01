#!/usr/bin/env python3
"""
升级 VLM cache 格式脚本
将旧版本 VLM cache（缺少 details 字段）映射到最新格式
从现有数据（summary、people.appearance、clothing、scene等）反向提取硬核线索

使用方式:
  python3 scripts/upgrade_vlm_cache_format.py \
    --input /path/to/old_vlm_cache.json \
    --output /path/to/new_vlm_cache.json
"""
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any


class VLMCacheUpgrader:
    """升级 VLM cache 格式"""

    # 品牌关键词库
    BRAND_KEYWORDS = {
        '品牌': ['Hello Kitty', 'Pinko', 'KANS', '阿迪达斯', '耐克', 'Nike', 'Adidas', '蒙牛', '伊利', '花呗', '支付宝', '微信', '淘宝', '美团', '饿了么'],
        '科技产品': ['iPhone', 'iPad', 'MacBook', 'Apple', 'Samsung', '华为', 'OPPO', 'vivo', 'OnePlus', 'Google', 'Pixel', 'PlayStation', 'Xbox'],
        '服装鞋帽': ['Zara', 'H&M', 'Uniqlo', 'MUJI', 'Gap', 'Forever 21', 'ASOS', '森马', '美邦', '太平鸟'],
        '食饮': ['星巴克', 'Costa', '瑞幸', '喜茶', '奶茶', '咖啡', 'Starbucks', 'Coca-Cola', 'Pepsi'],
        '证件和账单': ['学号', '订单', '账单', '证件', '身份证', '学生卡', '会员卡', '驾驶证', '护照'],
    }

    def __init__(self):
        pass

    def upgrade(self, input_path: str, output_path: str):
        """升级 VLM cache 文件"""
        input_file = Path(input_path)
        output_file = Path(output_path)

        print(f"📖 读取旧版本 VLM cache: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        # 处理结构
        if 'photos' in cache_data and isinstance(cache_data['photos'], list):
            photos = cache_data['photos']
            print(f"✓ 找到 {len(photos)} 张照片")

            # 升级每张照片的数据
            upgraded_count = 0
            for photo in photos:
                if self._upgrade_photo(photo):
                    upgraded_count += 1

            print(f"✅ 升级了 {upgraded_count} 张照片")

        # 保存升级后的数据
        print(f"💾 保存升级后的 VLM cache: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 完成！升级后的文件已保存到 {output_file}")

    def _upgrade_photo(self, photo: Dict[str, Any]) -> bool:
        """升级单张照片的数据，返回是否成功升级"""
        if 'vlm_analysis' not in photo:
            return False

        analysis = photo['vlm_analysis']

        # 提取 details
        details = self._extract_details_from_analysis(analysis)

        # 强制覆盖或添加 details（即使已有空的 details 也要填充）
        old_details = analysis.get('details', [])
        analysis['details'] = details if details else old_details

        return len(details) > 0

    def _extract_details_from_analysis(self, analysis: Dict[str, Any]) -> List[str]:
        """从 vlm_analysis 中提取硬核线索"""
        details = set()

        # 1. 从 summary 中提取文本线索
        summary = analysis.get('summary', '')
        if summary:
            details.update(self._extract_from_text(summary))

        # 2. 从 people 数据中提取品牌和文字信息
        people = analysis.get('people', [])
        if isinstance(people, list):
            for person in people:
                if isinstance(person, dict):
                    # clothing 字段通常包含品牌信息
                    clothing = person.get('clothing', '')
                    if clothing:
                        details.update(self._extract_from_text(clothing))

                    # activity 中可能有品牌或物品信息
                    activity = person.get('activity', '')
                    if activity:
                        details.update(self._extract_from_text(activity))

        # 3. 从 scene 中提取环境细节和品牌
        scene = analysis.get('scene', {})
        if isinstance(scene, dict):
            env_details = scene.get('environment_details', [])
            if isinstance(env_details, list):
                for detail in env_details:
                    if detail:
                        details.update(self._extract_from_text(detail))

        # 4. 从 event 中提取活动相关的品牌和线索
        event = analysis.get('event', {})
        if isinstance(event, dict):
            activity = event.get('activity', '')
            if activity:
                details.update(self._extract_from_text(activity))

            objects = event.get('objects', [])
            if isinstance(objects, list):
                for obj in objects:
                    if obj:
                        details.update(self._extract_from_text(obj))

        # 5. 从 relations 中提取物品信息
        relations = analysis.get('relations', [])
        if isinstance(relations, list):
            for relation in relations:
                if isinstance(relation, dict):
                    obj = relation.get('object', '')
                    if obj:
                        details.update(self._extract_from_text(obj))

        # 返回列表形式，去除重复
        return list(details)

    def _extract_from_text(self, text: str) -> List[str]:
        """从文本中提取硬核线索（品牌、证件、账单等）"""
        extracted = []

        if not text:
            return extracted

        text_lower = text.lower()

        # 1. 匹配已知品牌
        for category, brands in self.BRAND_KEYWORDS.items():
            for brand in brands:
                if brand.lower() in text_lower:
                    extracted.append(brand)

        # 2. 提取证件/账单关键词
        cert_keywords = ['学号', '订单', '账单', '证件', '身份证', '学生卡', '会员卡', '驾驶证', '护照', '发票', '收据']
        for keyword in cert_keywords:
            if keyword in text:
                # 尝试提取这个关键词附近的内容（比如学号后面的数字）
                match = re.search(rf'{keyword}[：:]\s*([^\s,。，;；]*)', text)
                if match:
                    value = match.group(1).strip()
                    if value:
                        extracted.append(f"{keyword}：{value}")
                else:
                    extracted.append(keyword)

        # 3. 提取数字模式（可能是证件号、学号等）
        # 学号通常是 8-10 位数字，账单号也类似
        numbers = re.findall(r'\d{6,12}', text)
        for num in numbers[:2]:  # 只保留前两个数字
            extracted.append(f"ID/编号: {num}")

        # 4. 提取 @ 提及的名称（可能是用户名、账户）
        at_mentions = re.findall(r'@(\w+)', text)
        for mention in at_mentions:
            extracted.append(f"@{mention}")

        # 5. 提取 URL 或域名
        urls = re.findall(r'(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})+)', text)
        for url in urls[:2]:
            if url not in ['com', 'cn', 'org', 'net']:
                extracted.append(f"域名: {url}")

        # 去除重复并限制长度
        extracted = list(dict.fromkeys(extracted))  # 保留插入顺序去重

        return extracted[:10]  # 最多保留 10 个细节


def main():
    parser = argparse.ArgumentParser(
        description='升级 VLM cache 格式，补全 details 字段'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='输入的旧版本 VLM cache 文件路径'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='输出的升级后 VLM cache 文件路径'
    )

    args = parser.parse_args()

    upgrader = VLMCacheUpgrader()
    upgrader.upgrade(args.input, args.output)


if __name__ == '__main__':
    main()
