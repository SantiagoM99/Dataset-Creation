# Spanish Medical Dataset Creator

A toolkit for collecting and translating Spanish medical literature from PubMed, breaking the 9999 article limit.

## 📋 Overview

This project provides two main scripts:

1. **`pub_downloader.py`** - Collects copyright-free Spanish medical articles with abstracts and MeSH terms
2. **`medical_translator.py`** - Translates English abstracts to Spanish with medical terminology optimization

## 🛠 Installation

```bash
# Clone the repository
git clone <repository-url>
cd spanish-medical-dataset

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```
requests>=2.28.0
torch>=1.12.0
transformers>=4.20.0
tqdm>=4.60.0
psutil>=5.8.0
```

## 🚀 Usage

### 1. Spanish Article Collector

Collects Spanish medical articles from PubMed with strict quality filters.

```bash
python pub_downloader.py [OPTIONS]
```

**Options:**
- `--target` - Number of articles to collect (default: 50,000)
- `--output` - Output JSON filename (default: `spanish_medical_articles.json`)
- `--api-key` - NCBI API key for faster access (recommended)

**Examples:**
```bash
# Basic collection
python pub_downloader.py

# Custom target with API key
python pub_downloader.py --target 25000 --api-key YOUR_API_KEY

# Custom output file
python pub_downloader.py --target 10000 --output cardiology_dataset.json
```

**Quality Filters Applied:**
- ✅ Spanish language content
- ✅ Has abstract (minimum 50 words)
- ✅ Has MeSH terms
- ✅ Copyright-free (open access/free full text)

### 2. Medical Abstract Translator

Translates English abstracts to Spanish with medical terminology optimization.

```bash
python medical_translator.py INPUT_FILE [OPTIONS]
```

**Options:**
- `-o, --output` - Output filename (default: `INPUT_translated.json`)
- `-m, --model` - Translation model (default: `Helsinki-NLP/opus-mt-en-es`)
- `-b, --batch-size` - Force specific batch size (default: auto-detected)
- `--no-tune` - Skip auto-tuning batch size

**Examples:**
```bash
# Auto-tuned translation (recommended)
python medical_translator.py spanish_medical_articles.json

# Custom model and output
python medical_translator.py dataset.json -m facebook/nllb-200-distilled-600M -o translated.json

# Fixed batch size for limited memory
python medical_translator.py dataset.json -b 4 --no-tune

# Skip tuning with custom settings
python medical_translator.py dataset.json -o output.json --no-tune -b 8
```

**Available Models:**
- `Helsinki-NLP/opus-mt-en-es` (300MB) - Fast, good quality
- `facebook/nllb-200-distilled-600M` (2.4GB) - Better quality
- `facebook/nllb-200-1.3B` (5.2GB) - Best quality, slower

## 📊 Complete Workflow

```bash
# Step 1: Collect Spanish medical articles
python pub_downloader.py --target 50000 --api-key YOUR_API_KEY

# Step 2: Translate English abstracts to Spanish
python medical_translator.py spanish_medical_articles.json -o translated_dataset.json
```

## 🔧 System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- 10GB free disk space

**Recommended:**
- 16GB+ RAM
- GPU (for faster translation)
- NCBI API key


## 📄 Output Structure

The scripts generate JSON files with complete article metadata:

```json
{
  "metadata": {
    "creation_date": "2024-01-15T10:30:00",
    "total_articles": 50000,
    "filters_applied": ["Spanish language", "Has abstract", "Has MeSH terms", "Copyright-free"]
  },
  "statistics": {
    "language_distribution": {"spanish": 35000, "english": 12000, "mixed": 3000},
    "successful_translations": 11800
  },
  "articles": [
    {
      "pmid": "12345678",
      "title": "Article Title",
      "abstract": {
        "full_text": "Original abstract...",
        "spanish_translation": {
          "text": "Translated abstract...",
          "success": true,
          "model_used": "Helsinki-NLP/opus-mt-en-es"
        }
      },
      "abstract_language": "english",
      "mesh_terms": [...],
      "journal": {...},
      "authors": [...]
    }
  ]
}
```


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.