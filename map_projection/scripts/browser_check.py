"""Optional laptop QA using an installed Chrome (not a Pi dependency)."""
import json
import os
import shutil
from pathlib import Path
import time
from urllib.request import Request, urlopen
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
URL=os.environ.get("MAP_PROJECTION_QA_URL","http://127.0.0.1:8080")

def state():
    with urlopen(URL+"/api/state") as response:return json.load(response)

def update(config):
    request=Request(URL+"/api/config",json.dumps(config).encode(),{"Content-Type":"application/json"})
    with urlopen(request) as response:return json.load(response)

def ready(timeout=30):
    stop=time.monotonic()+timeout
    while time.monotonic()<stop:
        s=state()
        if s["error"]:raise AssertionError(s["error"])
        if s["revision"]==s["rendered_revision"]:return s
        time.sleep(.1)
    raise AssertionError("Render did not finish")

original=state()["config"]
with sync_playwright() as p:
    chrome=os.environ.get("MAP_PROJECTION_CHROME") or (r"C:\Program Files\Google\Chrome\Application\chrome.exe" if os.name=="nt" else shutil.which("google-chrome") or shutil.which("chromium"))
    browser=p.chromium.launch(executable_path=chrome,headless=True)
    page=browser.new_page(viewport={"width":1440,"height":1100},device_scale_factor=1)
    errors=[]
    page.on("pageerror",lambda error:errors.append(str(error)))
    def click_request(selector,path="/api/config"):
        with page.expect_response(lambda response:response.url.endswith(path) and response.request.method=="POST") as response:
            page.locator(selector).click()
        assert response.value.ok,response.value.text()
    try:
        page.goto(URL)
        page.wait_for_function("document.getElementById('connection').textContent.includes('connected')")
        ready()
        page.locator("#preview").wait_for(state="visible")
        page.screenshot(path=str(ROOT/".cache"/"desktop.png"),full_page=True)
        page.get_by_role("button",name="Calibrate",exact=True).click()
        click_request("#show-calibration")
        ready()
        before=state()["corners"][0][0]
        button=page.get_by_role("button",name="Move corner right")
        button.hover();page.mouse.down();page.wait_for_timeout(620);page.mouse.up()
        page.wait_for_timeout(200)
        after=state()["corners"][0][0]
        assert after-before>=3,(before,after)
        page.wait_for_timeout(250)
        assert state()["corners"][0][0]==after,"Hold-repeat continued after release"
        print("hold_nudge_pixels",after-before)
        page.get_by_role("button",name="Output",exact=True).click()
        page.locator("#pattern").select_option("checkerboard")
        click_request("#pattern-apply");ready()
        assert state()["config"]["pattern"]=="checkerboard"
        page.get_by_role("button",name="Sunlight",exact=True).click()
        page.locator("#tab-sun details").evaluate("e=>e.open=true")
        with page.expect_response(lambda r:r.url.endswith("/api/config") and r.request.method=="POST"):
            page.locator("#sun-terrain").select_option("dem")
        click_request("#shadow-toggle")
        dem=ready(90)
        assert dem["config"]["sun"]["terrain"]=="dem"
        print("dem_timings",dem["timings"])
        page.get_by_role("button",name="Locate",exact=True).click()
        click_request("#target-geo","/api/query")
        stop=time.monotonic()+30
        while time.monotonic()<stop:
            query=state()["query"]
            if query and "map_mm" in query:break
            time.sleep(.1)
        assert query and query.get("visible"),query
        print("mont_blanc_query",query)
        update(original);ready()
        page.reload();page.wait_for_function("document.getElementById('connection').textContent.includes('connected')")
        page.wait_for_timeout(500)
        page.screenshot(path=str(ROOT/".cache"/"desktop.png"),full_page=True)
        page.set_viewport_size({"width":390,"height":844})
        page.screenshot(path=str(ROOT/".cache"/"mobile.png"),full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth"),"Mobile horizontal overflow"
        assert page.locator("#error").is_hidden(),page.locator("#error").inner_text()
        assert not errors,errors
        print("Browser checks passed; screenshots saved to map_projection/.cache")
    finally:
        update(original)
        browser.close()
