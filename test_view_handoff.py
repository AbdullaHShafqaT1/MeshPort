"""Optional Chromium check for both generation upload paths, using a simulated service."""

import base64
import io
import json
import mimetypes
import re
import unittest
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:
    sync_playwright = None


STATIC = Path(__file__).resolve().parent / "app" / "static"
ORIGIN = "http://127.0.0.1"


def png(color):
    with io.BytesIO() as output:
        Image.new("RGB", (16, 16), color).save(output, format="PNG")
        return output.getvalue()


@unittest.skipUnless(sync_playwright, "Playwright and a Chromium browser are required")
class ViewHandoffTests(unittest.TestCase):
    def test_selected_crops_and_direct_uploads_submit_the_same_views(self):
        angles = ["front", "rear", "left", "right", "top", "oblique"]
        image_bytes = [png(color) for color in ("red", "green", "blue", "orange", "yellow", "purple")]
        encoded = [base64.b64encode(image).decode("ascii") for image in image_bytes]
        names = [f"meshport-{angle}-{index}.png" for index, angle in enumerate(angles, 1)]
        original = png("white")
        seventh = base64.b64encode(png("black")).decode("ascii")
        crop_result = {
            "model": "test-vision",
            "width": 16,
            "height": 16,
            "missing": ["bottom"],
            "views": [
                {"angle": angle, "box": [1, 1, 15, 15], "image": "data:image/png;base64," + image}
                for angle, image in zip([*angles, "rear_three_quarter"], [*encoded, seventh])
            ],
        }
        sheet_uploads = []
        jobs = []
        job_id = "a" * 32

        def studio(route):
            request = route.request
            url = urlsplit(request.url)
            if f"{url.scheme}://{url.netloc}" != ORIGIN:
                route.abort()
                return
            path = url.path
            if path.startswith("/static/"):
                file = STATIC / path.removeprefix("/static/")
            elif path in ("/", "/angle-preview"):
                file = STATIC / ("studio.html" if path == "/" else "angle-preview.html")
            else:
                if path == "/api/angle-preview":
                    sheet_uploads.append(request.post_data_buffer)
                    reply = crop_result
                elif path == "/api/status":
                    reply = {"connected": True}
                elif path == "/api/jobs" and request.method == "POST":
                    jobs.append(json.loads(request.post_data))
                    reply = {"id": job_id}
                elif path == "/api/jobs/" + job_id:
                    reply = {"status": "complete", "stage": "Model ready"}
                else:
                    route.fulfill(status=404)
                    return
                route.fulfill(json=reply)
                return
            route.fulfill(path=file, content_type=mimetypes.guess_type(file)[0])

        def upload(name, data):
            return {"name": name, "mimeType": "image/png", "buffer": data}

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.route(re.compile(r"^https?://"), studio)
            page.goto(ORIGIN + "/angle-preview")
            page.locator("#file").set_input_files(upload("reference-sheet.png", original))
            page.locator("#analyze").click()
            expect(page.locator(".crop-card")).to_have_count(7)
            send = page.locator("#send-to-studio")
            expect(send).to_be_disabled()
            expect(page.locator("#handoff-error")).to_contain_text("up to six")
            page.locator(".crop-select").last.uncheck()
            expect(send).to_be_enabled()
            send.click()
            expect(page.locator("#thumbnails .reference-view span")).to_have_text(names)
            expect(page).to_have_url(ORIGIN + "/")
            expect(page.locator("#viewSummary")).to_contain_text("Multi-view")
            self.assertEqual(sheet_uploads, [original])
            self.assertEqual(jobs, [])

            page.locator("#generate").click()
            expect(page.locator("#message")).to_contain_text("Your model is ready")
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["images"], encoded)

            page.reload()
            expect(page.locator("#thumbnails .reference-view")).to_have_count(0)
            direct_uploads = [upload(name, data) for name, data in zip(names, image_bytes)]
            page.locator("#images").set_input_files(direct_uploads[:2])
            page.locator("#images").set_input_files(direct_uploads[2:])
            expect(page.locator("#thumbnails .reference-view span")).to_have_text(names)
            self.assertEqual(len(jobs), 1)

            page.locator("#generate").click()
            expect(page.locator("#message")).to_contain_text("Your model is ready")
            self.assertEqual(len(jobs), 2)
            self.assertEqual(jobs[0]['angles'], angles)
            self.assertEqual(jobs[0]['images'], jobs[1]['images'])
            browser.close()


if __name__ == "__main__":
    unittest.main()
