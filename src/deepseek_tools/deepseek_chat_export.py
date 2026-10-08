import re
from playwright.sync_api import Playwright, sync_playwright, expect

from playwright.sync_api import sync_playwright

def run(playwright: Playwright) -> None:
    with open("deepseek_export.html", "w", encoding="utf-8") as f:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()
        
        page.goto("https://chat.deepseek.com/share/a8trlygl4vehtcv9mc")
        inner_htmls = {}
        index = 0
        maybe_more = True
        while maybe_more:
            maybe_more = False
            for msg in page.locator(".ds-message").all():
                msg.scroll_into_view_if_needed()
                inner_html = msg.inner_html()
                if inner_html not in inner_htmls:
                    maybe_more = True
                    inner_htmls[inner_html] = index
                    index += 1
        context.close()
        browser.close()

if __name__ == "__main__":
    with sync_playwright() as playwright:
        run(playwright)
