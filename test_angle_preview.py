"""Boundary checks for local view separation."""

import base64
import io
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app import studio
from app.sheet_panels import split_gutters
from app.view_detection import angle_of


class AnglePreviewTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(studio.app)

    def test_crop_uses_source_pixels_and_reports_missing_views(self):
        source = Image.new("RGB", (40, 20), "#010203")
        source.paste("#fafbfc", (5, 2, 31, 18))
        upload = io.BytesIO()
        source.save(upload, format="PNG")
        with patch("app.angle_preview.detect_views", return_value=[("front", (3, 1, 29, 17))]):
            response = self.client.post("/api/angle-preview", content=upload.getvalue())
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual((result["width"], result["height"]), source.size)
        self.assertEqual(result["missing"], ["rear", "left", "right", "top", "bottom"])
        view, = result["views"]
        self.assertEqual(view["box"], [3, 1, 29, 17])
        crop = base64.b64decode(view["image"].removeprefix("data:image/png;base64,"))
        with Image.open(io.BytesIO(crop)) as decoded:
            self.assertEqual(decoded.size, (26, 16))
            self.assertEqual(decoded.tobytes(), source.crop((3, 1, 29, 17)).tobytes())

    def test_rejects_bad_upload_and_foreign_origin(self):
        route = "/api/angle-preview"
        self.assertEqual(self.client.post(route, content=b"broken JPG").status_code, 400)
        self.assertEqual(self.client.post(route, headers={"Origin": "https://example.test"}).status_code, 403)

    def test_flat_image_does_not_produce_a_view(self):
        upload = io.BytesIO()
        Image.new("RGB", (300, 250), "#ebeae9").save(upload, format="JPEG")
        result = self.client.post("/api/angle-preview", content=upload.getvalue())
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["views"], [])

    def test_split_preserves_separate_panels(self):
        source = Image.new("RGB", (440, 180), "white")
        draw = ImageDraw.Draw(source)
        draw.ellipse((12, 12, 205, 168), fill="black")
        draw.rectangle((238, 18, 429, 167), fill="black")
        blank, panels = split_gutters(source)
        self.assertFalse(blank)
        left, right = panels
        self.assertLess(left[2], right[0])
        self.assertGreater(left[2], 205)
        self.assertLessEqual(right[0], 238)
        self.assertEqual(angle_of("TOP SPEED"), "")


if __name__ == "__main__":
    unittest.main()
