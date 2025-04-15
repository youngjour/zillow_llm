import os
import requests
from dotenv import load_dotenv

# Load Dropbox URLs from .env or external config
load_dotenv()

# === File mapping ===
DROPBOX_FILE_LINKS = {
    "1. zillow.csv": "https://www.dropbox.com/scl/fi/5fwv477wonfi49h5z25r6/1.-zillow.csv?rlkey=9scjdub4ecnil6dsu8u98evrb&dl=1",
    "2. zillow_cleaned.csv": "https://www.dropbox.com/scl/fi/jnhxx9a8pljpate4uwavt/2.-zillow_cleaned.csv?rlkey=uqgx253rsruppr2qot0y42ihk&dl=1",
    "2. zillow_cleaned.geojson": "https://www.dropbox.com/scl/fi/a59jgmctcx5zh5918zpzg/2.-zillow_cleaned.geojson?rlkey=01g9ywlz48wlag6ykbhc7e0ai&dl=1",
    "3. ha_freq_binary.h5": "https://www.dropbox.com/scl/fi/vfbosqv1p9ylsjc3hxam1/3.-ha_freq_binary.h5?rlkey=n7be0icn2roqxqofr39opsiey&dl=1",
    "3. tfidf.h5": "https://www.dropbox.com/scl/fi/ovm3zaxj7p6beyvz5hx10/3.-tfidf.h5?rlkey=nk8fswuxrdulm1cjqb4j7pxk4&dl=1",
    "3. zillow_tokenized.geojson": "https://www.dropbox.com/scl/fi/ovm3zaxj7p6beyvz5hx10/3.-tfidf.h5?rlkey=nk8fswuxrdulm1cjqb4j7pxk4&dl=1",
    "4. bert_embedding.csv": "https://www.dropbox.com/scl/fi/o6u48ps4nugydf2b0owj5/4.-bert_embedding.csv?rlkey=4r0towx4o2zbzo7rzq6crolsx&dl=1",
    "4. bert_pca.csv": "https://www.dropbox.com/scl/fi/yea8cmkb9d8e0dx2quskw/4.-bert_pca.csv?rlkey=190ewldtgcpvftcqsqb0nq3s6&dl=1",
    "4. w2v_embedding.csv": "https://www.dropbox.com/scl/fi/7cjbxoals8eqo36vpvnt8/4.-w2v_embedding.csv?rlkey=zkfkizntvp96lx294d3mu3a2k&dl=1",
    "4. w2v_pca.csv": "https://www.dropbox.com/scl/fi/u3vdpaywdrnwou299gq0d/4.-w2v_pca.csv?rlkey=4czchjv1id134rxn77v3ym7ev&dl=1",
    "5. gpt_embedding.csv": "https://www.dropbox.com/scl/fi/nl39azhgbcjor5uvjencf/5.-gpt_embedding.csv?rlkey=8dfs3gn5wytitojn0n78nnahw&dl=1",
    "5. gpt_pca.csv": "https://www.dropbox.com/scl/fi/y1chwab5jatjkrcip6obe/5.-gpt_pca.csv?rlkey=bn1kvvgtxn5fo5mjitm8gilnj&dl=1",
    "5. stf_embedding.csv": "https://www.dropbox.com/scl/fi/gcrk2mejy3su7nt9gf30i/5.-stf_embedding.csv?rlkey=d5uy0qm80geh81qxhxbtyms66&dl=1",
    "5. stf_pca.csv": "https://www.dropbox.com/scl/fi/6gdiftk79r00a3uecf9zv/5.-stf_pca.csv?rlkey=j8e7e5tt81w2yt6fd3968mdwp&dl=1",
    "6. llama_extracted.h5": "https://www.dropbox.com/scl/fi/ph2nflpnedczn2yohxgy1/6.-llama_extracted.h5?rlkey=0qffb09lqz2flkcof86pmf7ma&dl=1",
    "7.gpt_cluster.h5": "https://www.dropbox.com/scl/fi/kxn7pzxblj5kxtf295noi/7.-gpt_cluster.h5?rlkey=oz634zfm4erlo5emmvt6tmmmd&dl=1"
}

def download_file(filename, save_dir="dataset"):
    """Download a file from Dropbox using its mapped name."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    if not os.path.exists(filepath):
        if filename not in DROPBOX_FILE_LINKS:
            raise ValueError(f"No Dropbox link defined for: {filename}")

        url = DROPBOX_FILE_LINKS[filename]
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Saved to {filepath}")
        else:
            raise Exception(f"Failed to download {filename}. HTTP {response.status_code}")
    else:
        print(f"File already exists: {filepath}")

    return filepath