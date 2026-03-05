import os
import time
import requests
import pandas as pd

DATASET = "wangrongsheng/ag_news"
BASE = "https://datasets-server.huggingface.co"
CHUNK = 100

SLEEP_BETWEEN_REQUESTS = 0.35

MAX_RETRIES = 8


session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; ag-news-downloader/1.0)",
    "Accept": "application/json",
})


def get_json(url: str):
    """
    GET JSON with handling for 429 + transient errors.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=30)

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(2 ** attempt, 60)
                print(f" 429 Too Many Requests. Waiting {wait:.1f}s then retrying... (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json()

        except requests.RequestException as e:
            wait = min(2 ** attempt, 60)
            print(f"️ Request failed: {e}. Waiting {wait:.1f}s... (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {url}")


def download_split(config: str, split: str, num_rows: int) -> pd.DataFrame:
    rows = []
    offset = 0

    while offset < num_rows:
        length = min(CHUNK, num_rows - offset)
        url = f"{BASE}/rows?dataset={DATASET}&config={config}&split={split}&offset={offset}&length={length}"

        payload = get_json(url)
        chunk = payload.get("rows", [])
        if not chunk:
            print(f"[{split}] empty chunk at offset {offset}, skipping...")
            offset += length
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        rows.extend([x.get("row", {}) for x in chunk])
        offset += length

        print(f"[{split}] downloaded {offset}/{num_rows}")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return pd.DataFrame(rows)


def main():
    os.makedirs("data", exist_ok=True)

    size_url = f"{BASE}/size?dataset={DATASET}"
    size_data = get_json(size_url)
    splits = size_data["size"]["splits"]

    config = splits[0]["config"]
    split_rows = {s["split"]: int(s["num_rows"]) for s in splits if s["config"] == config}

    train_df = download_split(config, "train", split_rows["train"])
    test_df  = download_split(config, "test",  split_rows["test"])

    full_df = pd.concat([train_df, test_df], ignore_index=True)

    out_path = "ag_news_full.csv"
    full_df.to_csv(out_path, index=False, encoding="utf-8")
    print("Saved:", out_path, "Rows:", len(full_df))


if __name__ == "__main__":
    main()
