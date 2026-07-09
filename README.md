# 🎵 LyricistGPT

**A specialized conversational AI for Hindi film lyrics** — it identifies lyricists, explains themes and writing style, recommends similar songs, and generates original lyrics in a chosen style.

Built by combining **instruction fine-tuning (QLoRA)** with **Retrieval-Augmented Generation (RAG)** on a base [`Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) model. Runs on a single free Google Colab **T4 GPU**.

> **TL;DR (fastest path)** — Copy the shared `LyricistGPT` folder into your **My Drive**, open the notebook in Colab with a **T4 GPU**, then **skip the training section** and start running from **Part B — Inference** (see [Where to start](#-where-to-start-executing)). Everything is pre-trained; you only load and chat.

---

## ✨ Features

- **Lyricist identification** — paste lyrics, get the most likely writer (grounded in retrieved evidence)
- **Theme & style analysis** — literary devices, mood, Hindi–Urdu register
- **Song recommendation** — *retrieval-first*, so it never invents titles
- **Original lyric generation** — new lines in a chosen style, no copyrighted text reproduced
- **Intent router** — each query is classified into one of six tasks and sent to a dedicated handler
- **Transparent UI** — every answer shows the **detected intent** and the **retrieved songs with similarity scores**

---

## 🏗️ Architecture

```
User query
   │
   ▼
Intent Router          (classify into one of six tasks)
   │
   ▼
FAISS Retriever        (embed query with BGE-M3, fetch top-k similar songs)
   │
   ▼
Grounded Prompt Builder (RAG: answer only from retrieved context)
   │
   ▼
Fine-tuned Qwen2.5-1.5B (base + LoRA adapter, 4-bit)
   │
   ▼
Response  +  detected intent  +  retrieved songs
```

**Design principle:** fine-tuning supplies the *voice*, retrieval supplies the *facts*, the router keeps each task focused, and retrieval-first design keeps the system honest.

---

## 📦 What's in the Drive folder

The shared Google Drive folder `LyricistGPT/` contains everything pre-built — **no training required to run the app**:
Link to folder: https://drive.google.com/drive/folders/18ShFiJqZPYTpVWO5N9kD5J_BGFyszTuH?usp=drive_link
```
LyricistGPT/
├── LyricistGPT.ipynb        # ← the single notebook (data prep + training + inference + app)
├── final_adapter/           # ✅ ALREADY-TRAINED LoRA adapter + tokenizer
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── ...
├── vector_db/               # ✅ prebuilt retrieval index
│   ├── faiss.index
│   ├── metadata.pkl
│   └── embedding_model.txt
├── documents.csv            # curated song knowledge base
├── lyrics_master.csv        # cleaned master dataset
└── chatml_dataset.json      # instruction dataset (for reference / retraining)
```

Because `final_adapter/` and `vector_db/` already exist, **you do not need to run the training cells** — doing so would retrain the model from scratch (~50+ minutes) for no benefit.

---

## ⏱️ The notebook has two parts

| Part | Cells | What it does | Do testers run it? |
|---|---|---|---|
| **Part A — Data Prep & Training** | Phase 1–3 | Cleans data, builds the instruction dataset, fine-tunes the LoRA adapter (5 epochs) | ❌ **Skip** — already done |
| **Part B — Inference & App** | from "Load the Fine-Tuned Model" onward | Loads the trained adapter + FAISS index, RAG, router, Streamlit app | ✅ **Run this** |

---

## 🏁 Where to start executing

> **Start at the cell titled `## Part B — Inference & Application` (the "Load the Fine-Tuned Model" section).**
> Everything above it is training and is already done for you.

Two ways to do this in Colab:

- **Option 1 (recommended):** scroll to the **Part B — Inference** markdown header, click the first code cell under it, then use **Runtime → Run after** (runs that cell and everything below).
- **Option 2:** run the cells manually from that point downward.

> ⚠️ Do **not** use *Runtime → Run all* — that starts from Part A and retrains the model.

---

## 🚀 Quick Start (for testers)

### Prerequisites
- A Google account with Google Drive
- Google Colab with a **T4 GPU** runtime

### Step 1 — Copy the shared folder into your own Drive
1. Open the shared Drive link.
2. Right-click the **`LyricistGPT`** folder → **Make a copy** (or *Organise → Add shortcut*) into your **My Drive**.
3. Confirm it now lives at **`My Drive/LyricistGPT`**.

> ⚠️ The path **must** be `MyDrive/LyricistGPT`. If you place it elsewhere or rename it, update the `PROJECT_DIR` line in the notebook (Step 4).

### Step 2 — Open the notebook & select the GPU
1. In your Drive, open the `LyricistGPT` folder and double-click **`LyricistGPT.ipynb`**.
2. If prompted, open with **Google Colaboratory**.
3. Set the GPU: **Runtime → Change runtime type → T4 GPU → Save**.

### Step 3 — Install dependencies
Run the **first install cell** (near the top of Part B, or the notebook's setup cell):
```bash
!pip -q install transformers==4.52.4 peft==0.15.2 accelerate==1.8.1 \
  bitsandbytes==0.46.1 sentence-transformers==3.0.1 faiss-cpu \
  sentencepiece pandas numpy streamlit
```
> If Colab shows a **Restart runtime** prompt, click **Restart**, then continue from Step 4.

### Step 4 — Mount Drive & verify the files are found
```python
from google.colab import drive
drive.mount('/content/drive')

import os
P = '/content/drive/MyDrive/LyricistGPT'
print('adapter:',   os.listdir(f'{P}/final_adapter'))
print('vector_db:', os.listdir(f'{P}/vector_db'))
```
You should see the adapter files and the `vector_db` files. If you get a *not found* error, fix the path in the config cell.

### Step 5 — Run Part B cells (from the "start here" point, top to bottom)
Starting at **Part B — Inference** (see [Where to start](#-where-to-start-executing)), run each cell in order. This loads the 4-bit base + trained adapter, the **BGE-M3** embedder, and the saved **FAISS** index, then defines the intent router and handlers.

⏱️ Load cells take **~1–2 minutes**. Nothing is trained and nothing is re-embedded — the ready-made index is loaded directly.

### Step 6 — Launch the Streamlit chatbot
Run the cell that writes `app.py`, then run the launch cell and wait for `up`:

```python
import subprocess, time, requests

!pkill -9 -f streamlit 2>/dev/null
time.sleep(2)

subprocess.Popen(
    ['streamlit', 'run', 'app.py', '--server.port', '8502',
     '--server.address', '0.0.0.0', '--server.headless', 'true',
     '--server.enableCORS', 'false', '--server.enableXsrfProtection', 'false'],
    stdout=open('/content/logs.txt', 'w'), stderr=subprocess.STDOUT)

for i in range(80):
    try:
        if requests.get('http://localhost:8502').status_code == 200:
            print('up'); break
    except Exception:
        pass
    time.sleep(3)

from google.colab.output import eval_js
print(eval_js('google.colab.kernel.proxyPort(8502)'))
```

Click the printed URL. Wait ~1 minute for the model to finish loading, then start chatting.

---

## 💬 Things to try

```
Who wrote these lyrics?
तुम हो पास मेरे, साथ मेरे हो तुम यूँ...

Recommend songs about heartbreak and longing.

Write four original Hindi lines about hope after separation.
```

Each reply shows the **answer**, the **detected intent**, and the **retrieved songs** with similarity scores.

---

## 🔁 Want to retrain from scratch? (optional)

If you actually want to reproduce the training instead of using the pre-built adapter:

1. Set a **T4 GPU** runtime.
2. Use **Runtime → Run all** to execute **Part A — Data Prep & Training** first (~50+ minutes). This regenerates `final_adapter/`.
3. Continue into **Part B** to build the index and serve the app.

Otherwise, ignore Part A entirely and follow the Quick Start above.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| **Adapter / files not found** | The folder isn't at `MyDrive/LyricistGPT`; fix the path in the config cell. |
| **Accidentally started training** | You ran from Part A. Stop the cell, and start from **Part B — Inference** instead (see [Where to start](#-where-to-start-executing)). |
| **Blank page or `Bad Gateway`** | The app is still loading; wait a minute and refresh once. If it persists, run `!cat /content/logs.txt` and read the error. |
| **`Port 8502 is not available`** | Re-run the `pkill` line, or change `8502` → `8503` in both places. |
| **No GPU / very slow** | Confirm **Runtime → Change runtime type → T4 GPU**. |
| **Restart prompt after `pip`** | Restart once, then resume from Step 4. |

---

## 🧠 Tech stack

| Component | Choice |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Fine-tuning | QLoRA (4-bit NF4 + LoRA adapters), 5 epochs |
| Embeddings | `BAAI/bge-m3` (multilingual) |
| Vector search | FAISS (`IndexFlatIP`, cosine similarity) |
| Serving | Streamlit on Colab |
| Hardware | Google Colab T4 GPU |

---

## ⚠️ Notes & limitations

- Grounded in a curated corpus of **~350 songs across seven lyricists** — authoritative on that collection, not all of Hindi cinema.
- As a 1.5B model, it depends on retrieval for factual reliability; it is **not** a free-recall fact source.
- Generation produces **original** lyrics only and does not reproduce copyrighted text.
- Training/validation loss decreased together across 5 epochs, indicating learning without overfitting.

---

## 📄 License

Specify your license here (e.g. MIT). Dataset and model artifacts are for educational/demonstration use.
