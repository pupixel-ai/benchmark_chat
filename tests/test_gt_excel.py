from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook


class ProfileGtExcelConversionTests(unittest.TestCase):
    def test_extract_bundle_album_id_reads_backflow_album_id(self) -> None:
        from services.reflection.gt_excel import extract_bundle_album_id

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir)
            (bundle_dir / "downstream_audit_report.json").write_text(
                json.dumps(
                    {
                        "metadata": {"album_id": None},
                        "backflow": {"album_id": "bundle_album_001"},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            self.assertEqual(extract_bundle_album_id(bundle_dir), "bundle_album_001")

    def test_convert_profile_gt_excel_builds_only_usable_non_null_records(self) -> None:
        from services.reflection.gt_excel import convert_profile_gt_excel

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            workbook_path = tmp_path / "annotated_profile.xlsx"
            bundle_dir = tmp_path / "bundle"
            bundle_dir.mkdir()
            (bundle_dir / "downstream_audit_report.json").write_text(
                json.dumps(
                    {
                        "backflow": {"album_id": "bundle_album_002"},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (bundle_dir / "structured_profile.json").write_text(
                json.dumps(
                    {
                        "long_term_facts": {
                            "social_identity": {
                                "education": {
                                    "value": "在读学生",
                                    "confidence": 0.45,
                                    "evidence": [],
                                    "reasoning": "校园券和校园活动",
                                }
                            }
                        },
                        "short_term_expression": {
                            "motivation_shift": {
                                "value": "由兴趣导向向高端体验消费偏移",
                                "confidence": 0.7,
                                "evidence": [],
                                "reasoning": "近期两次消费记录",
                            },
                            "mental_state": {
                                "value": "irregular",
                                "confidence": 0.4,
                                "evidence": [],
                                "reasoning": "英文枚举输出",
                            },
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "画像"
            sheet.append(
                [
                    "中文标签",
                    "JSON Path",
                    "结果",
                    "置信度",
                    "是否准确和补充（1代表准确）",
                    "修正",
                    "证据摘要",
                ]
            )
            sheet.append(
                [
                    "教育背景",
                    "long_term_facts.social_identity.education",
                    "在读学生",
                    0.45,
                    1,
                    "",
                    "校园券和校园活动",
                ]
            )
            sheet.append(
                [
                    "动机变化",
                    "short_term_expression.motivation_shift",
                    "由兴趣导向向高端体验消费偏移",
                    0.7,
                    0,
                    "同步性",
                    "近期两次消费记录",
                ]
            )
            sheet.append(
                [
                    "职业",
                    "long_term_facts.social_identity.career",
                    "",
                    0,
                    1,
                    "",
                    "没有直接证据",
                ]
            )
            sheet.append(
                [
                    "社交群体",
                    "long_term_facts.relationships.social_groups",
                    "社交群体, 社交群体, 社交群体",
                    0.2,
                    "",
                    "",
                    "待确认",
                ]
            )
            sheet.append(
                [
                    "短期心理状态",
                    "short_term_expression.mental_state",
                    "不规律",
                    0.4,
                    1,
                    "",
                    "英文枚举输出",
                ]
            )
            workbook.save(workbook_path)

            records = convert_profile_gt_excel(
                workbook_path=workbook_path,
                bundle_dir=bundle_dir,
                labeler="vigar_manual_annotation",
            )

            self.assertEqual(len(records), 3)
            self.assertEqual(
                [record["field_key"] for record in records],
                [
                    "long_term_facts.social_identity.education",
                    "short_term_expression.motivation_shift",
                    "short_term_expression.mental_state",
                ],
            )
            self.assertTrue(all(record["album_id"] == "bundle_album_002" for record in records))
            self.assertEqual(records[0]["gt_value"], "在读学生")
            self.assertEqual(records[1]["gt_value"], "同步性")
            self.assertEqual(records[2]["gt_value"], "irregular")
            self.assertEqual(records[1]["original_output"], "由兴趣导向向高端体验消费偏移")
            self.assertIn("近期两次消费记录", records[1]["notes"])

    def test_write_profile_gt_jsonl_persists_utf8_payloads(self) -> None:
        from services.reflection.gt_excel import write_profile_gt_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profile_field_gt_default.jsonl"
            records = [
                {
                    "album_id": "bundle_album_003",
                    "field_key": "long_term_facts.identity.nationality",
                    "gt_value": "中国",
                    "labeler": "vigar_manual_annotation",
                    "notes": "地理位置与中文环境",
                }
            ]

            write_profile_gt_jsonl(records, output_path)

            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["gt_value"], "中国")
