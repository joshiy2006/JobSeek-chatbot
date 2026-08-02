import pandas as pd
import numpy as np
import time
import re
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from supabase import create_client

# ══════════════════════════════════════════════════════════════════════════════
# ⚙️  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
SUPABASE_URL = "https://mwapuwqoofloyjgtngbh.supabase.co"   # ← paste yours
SUPABASE_KEY = "sb_publishable__ul6pNAaPFFItuzhpYmswQ_iVwOFS36"                  # ← paste yours
PAGES        = 9
# ══════════════════════════════════════════════════════════════════════════════

options = Options()
options.add_experimental_option("detach", True)
service = Service(ChromeDriverManager().install())
driver  = webdriver.Chrome(service=service, options=options)


def get_location(job):
    tag = job.find(attrs={'data-automation': re.compile(r'location', re.I)})
    if tag and tag.get_text(strip=True):
        return tag.get_text(separator=', ', strip=True)
    for candidate in job.find_all(True):
        classes = ' '.join(candidate.get('class', []))
        if re.search(r'\bloc', classes, re.I):
            text = candidate.get_text(separator=', ', strip=True)
            if text:
                return text
    icon = job.find('i', class_=re.compile(r'loc|pin|place|map', re.I))
    if icon:
        parent = icon.find_parent(['span', 'li', 'div'])
        if parent:
            text = parent.get_text(separator=', ', strip=True)
            if text:
                return text
    tag = job.find(attrs={'aria-label': re.compile(r'location', re.I)})
    if tag and tag.get_text(strip=True):
        return tag.get_text(separator=', ', strip=True)
    try:
        el = driver.find_element(
            By.XPATH,
            "//div[contains(@class,'srp-jobtuple-wrapper')]"
            "[.//*[contains(@class,'title')]]"
            "//*[contains(@class,'loc') or contains(@class,'location')]"
        )
        text = el.text.strip()
        if text:
            return text
    except Exception:
        pass
    return None


def days_ago_to_iso(posted_text):
    if not posted_text or (isinstance(posted_text, float) and np.isnan(posted_text)):
        return None
    t = str(posted_text).lower().strip()
    if any(w in t for w in ['just', 'hour', 'today', 'few']):
        days = 0
    elif m := re.search(r'(\d+)\+?\s*day', t):
        days = int(m.group(1))
    elif m := re.search(r'(\d+)\s*week', t):
        days = int(m.group(1)) * 7
    elif m := re.search(r'(\d+)\s*month', t):
        days = int(m.group(1)) * 30
    else:
        return None
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


# ── Scraping ──────────────────────────────────────────────────────────────────
records = []

for page in tqdm(range(1, PAGES + 1), desc="Scraping Pages"):
    url = f'https://www.naukri.com/ai-engineer-jobs-in-india-{page}'
    driver.get(url)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'srp-jobtuple-wrapper'))
        )
        time.sleep(1.5)
    except Exception:
        print(f"  ⚠ Timeout on page {page}, skipping.")
        continue

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    jobs = soup.find_all('div', class_='srp-jobtuple-wrapper')

    for job in jobs:
        try:
            title_tag   = job.find('a', class_='title')
            title       = title_tag.text.strip() if title_tag else None
            job_link    = title_tag.get('href')   if title_tag else None

            if not job_link:
                continue

            company_tag = job.find('a', class_='comp-name')
            company     = company_tag.text.strip() if company_tag else None

            stars_tag   = job.find('span', class_='main-2')
            stars       = stars_tag.text.strip() if stars_tag else None

            exp_tag     = job.find('span', class_='expwdth')
            experience  = exp_tag.text.strip() if exp_tag else None

            location    = get_location(job)

            skills_tags = job.find_all('li', class_='dot-gt')
            skills      = ', '.join(s.text.strip() for s in skills_tags) if skills_tags else None

            posted_tag  = job.find('span', class_='job-post-day')
            posted      = posted_tag.text.strip() if posted_tag else None

            records.append({
                "jobtitle":   title,
                "company":    company,
                "stars":      stars,
                "experience": experience,
                "location":   location,
                "skills":     skills,
                "posted":     posted,
                "postdate":   days_ago_to_iso(posted),
                "site_name":  "naukri.com",
                "url":    job_link,
            })
        except Exception:
            continue

driver.quit()

loc_filled = sum(1 for r in records if r.get('location'))
print(f"\n🔍 Location found for {loc_filled}/{len(records)} jobs")
print(f"📦 Total records ready to upload: {len(records)}")


# ── Upload to Supabase ────────────────────────────────────────────────────────
def upload_to_supabase(records):
    print("\n⬆️  Uploading to Supabase...")
    client    = create_client(SUPABASE_URL, SUPABASE_KEY)
    batch_size = 500
    total      = len(records)
    uploaded   = 0

    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        client.table("naukri_jobs").upsert(batch, on_conflict="url").execute()
        uploaded += len(batch)
        print(f"  Uploaded {uploaded}/{total} rows...")

    print(f"✅  {total} jobs synced to Supabase → job_listings table")

upload_to_supabase(records)
print("\n🎉  All done!")