#!/usr/bin/env python3
"""Exercise the actual performance explorer and capture desktop/mobile views."""
import argparse
import gzip
import json
import pathlib
from playwright.sync_api import sync_playwright


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url",default="http://127.0.0.1:4173/perf/")
    p.add_argument("--browser",required=True)
    p.add_argument("--out",type=pathlib.Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    errors=[];checked=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=a.browser,headless=True,args=["--no-sandbox"])
        page=browser.new_page(viewport={"width":1440,"height":1100},device_scale_factor=1)
        page.context.set_offline(a.url.startswith("file:"))
        page.on("pageerror",lambda e:errors.append(str(e)))
        page.goto(a.url,wait_until="networkidle")
        page.locator(".metrics strong").first.wait_for()
        assert page.locator(".op-row").count()>3
        assert page.locator(".heat-cell").count()>1
        checked.append("real evidence loaded")
        page.get_by_label("Filter operators").fill("gemm")
        assert page.locator(".op-row").count()>0
        page.locator(".op-row").first.click()
        assert "gemm" in page.locator(".inspector h2").inner_text()
        page.get_by_label("Filter operators").fill("")
        page.get_by_label("Sort operators").select_option("variation")
        checked.append("operator filtering, selection and variation sorting")
        page.get_by_label("Hypothetical operator speedup").fill("3")
        assert "3.0" in page.locator(".whatif strong").inner_text()
        checked.append("interactive speedup projection")
        page.locator(".heat-cell").last.click()
        selected=page.get_by_label("Workload").input_value()
        assert selected
        assert page.locator('.heat-cell[aria-pressed="true"]').count()==1
        page.locator(".layer-track button").nth(2).click()
        assert page.locator(".inspector h2").inner_text()
        assert "IN THE REAL PROGRAM" in page.locator(".in-program").inner_text()
        checked.append("heatmap workload switching and program call inspection")
        for link in page.locator(".toolbar-links a").all():
            with page.expect_download() as pending:
                link.click()
            downloaded=pending.value
            content=pathlib.Path(downloaded.path()).read_bytes()
            assert json.loads(gzip.decompress(content) if content[:2]==b"\x1f\x8b" else content)
        checked.append("downloadable evidence and AI quick view")
        choices=page.get_by_label("Model").locator("option").evaluate_all("xs => xs.map(x => x.value)")
        for choice in choices:
            page.get_by_label("Model").select_option(choice)
            page.locator(".metrics strong").first.wait_for()
            assert page.locator(".op-row").count()>0
            scenarios=page.get_by_label("Workload").locator("option").evaluate_all("xs => xs.map(x => x.value)")
            for scenario in scenarios:
                page.get_by_label("Workload").select_option(scenario)
                assert "NaN" not in page.locator(".metrics").inner_text()
                assert page.locator(".op-row").count()>0
                assert page.locator(".layer-track button").count()>0
            checked.append(f"{choice}: all {len(scenarios)} workloads render")
        checked.append(f"{len(choices)} model dataset(s)")
        page.get_by_label("Model").select_option(choices[0])
        page.locator(".metrics strong").first.wait_for()
        page.get_by_label("Sort operators").select_option("time")
        page.get_by_label("Hypothetical operator speedup").fill("2")
        page.evaluate("scrollTo(0, 0)")
        page.screenshot(path=str(a.out/"desktop.png"),full_page=True)
        page.screenshot(path=str(a.out/"desktop-overview.png"))
        page.locator(".detail-grid").screenshot(path=str(a.out/"operators.png"))
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"desktop horizontal overflow"
        page.set_viewport_size({"width":390,"height":844})
        page.screenshot(path=str(a.out/"mobile.png"),full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"mobile horizontal overflow"
        checked.append("desktop and mobile layout without horizontal overflow")
        assert not errors,errors
        browser.close()
    result=dict(passed=True,checked=checked,browser_errors=errors)
    (a.out/"browser-check.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=="__main__":main()
