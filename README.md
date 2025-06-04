# Financial Headlines Sentiment Analysis with Hugging Face Transformers

This project demonstrates how to perform sentiment analysis on financial headlines using state-of-the-art transformer models from Hugging Face, specifically **FinBERT** - a model fine-tuned for financial text analysis.

> **Advanced Financial Sentiment Analysis using Hugging Face Transformers and FinBERT**

This project demonstrates state-of-the-art sentiment analysis on financial headlines using transformer models from Hugging Face, specifically **FinBERT** - a BERT model fine-tuned for financial text analysis. The system provides comprehensive sentiment classification with confidence scores, rich visualizations, and real-time data integration.

## 🌟 **Key Features**

- 🤖 **Advanced NLP Models**: Uses FinBERT (`ProsusAI/finbert`) specifically trained on financial text
- 📊 **Multiple Data Sources**: Fetches headlines from Yahoo Finance, News API, or uses sample data
- 🎯 **Comprehensive Analysis**: Provides sentiment classification with confidence scores
- 📈 **Rich Visualizations**: Creates professional charts showing sentiment distribution, trends, and confidence metrics
- ⏰ **Time-based Insights**: Analyzes sentiment trends over the past month
- 💾 **Export Results**: Saves analysis results to CSV for further processing
- 🎨 **Intuitive Color Coding**: Green (positive), Red (negative), Grey (neutral)

## 📊 **What It Analyzes**

The system classifies financial headlines into three sentiment categories:

| Sentiment | Color | Description | Examples |
|-----------|-------|-------------|----------|
| **Positive** 🟢 | Green | Optimistic news | Earnings beats, market rallies, positive announcements |
| **Negative** 🔴 | Red | Pessimistic news | Market crashes, layoffs, disappointing results |
| **Neutral** ⚪ | Grey | Factual news | Neutral reporting without clear sentiment direction |

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.7 or higher
- Internet connection (for downloading models and fetching data)
- ~2GB disk space (for transformer models)
- Optional: GPU for faster processing

### **Installation**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/1NT9NS9/transformer-hugging-face-financial-headlines.git
   cd transformer-hugging-face-financial-headlines
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Set up API keys** (for real-time data):
   ```bash
   # Copy the example environment file
   cp env_example.txt .env
   
   # Edit .env and add your API keys
   NEWS_API_KEY=your_newsapi_key_here
   ```
   Get a free API key from [NewsAPI.org](https://newsapi.org/)

### **Usage**

#### **Simple Example** (Recommended for first run)
```bash
python simple_example.py
```

#### **Full Analysis** (With real-time data)
```bash
python financial_sentiment_analysis.py
```

## 📈 **Sample Output**

```
🚀 Financial Sentiment Analysis with Hugging Face Transformers
============================================================
📊 Analyzing 15 financial headlines...

🤖 Loading FinBERT model (specialized for financial text)...
✅ FinBERT loaded successfully!

📈 SENTIMENT ANALYSIS RESULTS
----------------------------------------
Overall Sentiment Distribution:
  🟢 Positive: 6 headlines (40.0%)
  🔴 Negative: 5 headlines (33.3%)
  🟡 Neutral: 4 headlines (26.7%)

🎯 Average Confidence Score: 0.847

📰 Sample Headlines by Sentiment:

🟢 POSITIVE Examples:
   • Apple reports record quarterly earnings beating analyst expectations (confidence: 0.924)
   • Tech stocks rally as inflation shows signs of cooling (confidence: 0.891)

🔴 NEGATIVE Examples:
   • Tesla stock plunges after disappointing delivery numbers (confidence: 0.876)
   • Major bank announces significant layoffs amid economic uncertainty (confidence: 0.823)
```

## 🏗️ **Project Structure**

```
transformer-hugging-face-financial-headlines/
├── 📄 README.md                           # Project documentation
├── 📋 requirements.txt                    # Python dependencies
├── 🐍 financial_sentiment_analysis.py    # Complete analysis system
├── 🐍 simple_example.py                  # Streamlined example
├── 📝 env_example.txt                    # Environment variables template
├── 📄 LICENSE                           # MIT License
├── 📄 .gitignore                        # Git ignore rules
├── 📊 financial_sentiment_analysis.png   # Generated visualization (full)
├── 📊 financial_sentiment_chart.png      # Generated visualization (simple)
├── 📈 financial_sentiment_results.csv    # Analysis results (full)
└── 📈 sentiment_results.csv              # Analysis results (simple)
```

## 🔧 **Code Architecture**

### **Main Components**

1. **`FinancialSentimentAnalyzer` Class** (`financial_sentiment_analysis.py`):
   - Complete sentiment analysis pipeline
   - Multiple data source integration
   - Advanced visualizations and insights
   - Real-time data fetching

2. **Simple Example** (`simple_example.py`):
   - Streamlined version for quick testing
   - Uses sample data for immediate results
   - Basic visualizations
   - Perfect for learning and demonstration

### **Key Methods**

| Method | Description |
|--------|-------------|
| `analyze_sentiment()` | Core sentiment analysis using transformer models |
| `get_yahoo_finance_headlines()` | Fetch real headlines from Yahoo Finance |
| `get_newsapi_headlines()` | Fetch headlines using News API |
| `create_visualizations()` | Generate comprehensive charts |
| `generate_insights()` | Provide detailed analysis summary |

## 📊 **Generated Outputs**

The analysis creates several files:

### **CSV Results**
- `sentiment_results.csv` (simple) / `financial_sentiment_results.csv` (full)
- Contains: headlines, timestamps, sentiment, confidence scores, sources

### **Visualizations**
- `financial_sentiment_chart.png` (simple) / `financial_sentiment_analysis.png` (full)
- **Charts included**:
  - Sentiment distribution pie chart
  - Confidence score histogram
  - Sentiment timeline
  - Source-based analysis

## 🤖 **AI Models Used**

### **Primary Model: FinBERT**
- **Model**: [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert)
- **Specialization**: Financial text analysis
- **Training**: Fine-tuned on financial news and reports
- **Output**: Positive, Negative, Neutral with confidence scores
- **Advantage**: Superior performance on financial terminology

### **Fallback Model: RoBERTa**
- **Model**: [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- **Use Case**: When FinBERT is unavailable
- **Training**: General sentiment analysis
- **Advantage**: Robust general-purpose sentiment analysis

## 📚 **Understanding the Results**

### **Sentiment Classification**
- **Positive**: Bullish news, good earnings, market gains, positive announcements
- **Negative**: Bearish news, losses, negative announcements, market downturns
- **Neutral**: Factual reporting without clear sentiment direction

### **Confidence Scores**
- **0.8-1.0**: High confidence predictions (very reliable)
- **0.6-0.8**: Medium confidence predictions (reliable)
- **0.0-0.6**: Lower confidence predictions (less certain)

### **Key Insights Provided**
- Overall sentiment distribution across all headlines
- Time-based trends (recent vs. historical performance)
- High-confidence predictions analysis
- Peak news hours and timing patterns
- Source-based sentiment patterns and bias analysis

## 🔍 **Customization & Extension**

### **Adding New Data Sources**
```python
def get_custom_headlines(self):
    """Add your custom data source here."""
    headlines_data = []
    # Implement your data fetching logic
    # ... fetch and format data
    return pd.DataFrame(headlines_data)
```

### **Modifying Analysis Parameters**
```python
# Change date range
days_back = 30  # Analyze past 30 days

# Filter by confidence threshold
high_confidence = df[df['confidence'] > 0.8]

# Add custom stock symbols for Yahoo Finance
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META']

# Customize visualization colors
color_map = {
    'positive': '#2ecc71',  # Green
    'negative': '#e74c3c',  # Red  
    'neutral': '#95a5a6'    # Grey
}
```

## 📦 **Dependencies**

| Library | Version | Purpose |
|---------|---------|---------|
| `transformers` | 4.36.0 | Hugging Face Transformers (FinBERT, RoBERTa) |
| `torch` | 2.1.0 | PyTorch backend for transformer models |
| `pandas` | 2.1.3 | Data manipulation and analysis |
| `numpy` | 1.24.3 | Numerical computing |
| `matplotlib` | 3.8.2 | Basic plotting and visualization |
| `seaborn` | 0.13.0 | Statistical data visualization |
| `yfinance` | 0.2.28 | Yahoo Finance data fetching |
| `newsapi-python` | 0.2.7 | News API client |
| `requests` | 2.31.0 | HTTP requests |
| `beautifulsoup4` | 4.12.2 | HTML parsing |
| `python-dotenv` | 1.0.0 | Environment variable management |
| `scikit-learn` | 1.3.2 | Additional ML utilities |

## 🚨 **System Requirements**

- **Python**: 3.7 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: ~2GB for transformer models
- **Internet**: Required for model downloads and data fetching
- **GPU**: Optional (CUDA-compatible) for faster processing

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ **Disclaimer**

This project is for educational and research purposes. Please respect the terms of service of data providers (Yahoo Finance, News API) when using their services. The sentiment analysis results should not be used as the sole basis for financial decisions.

## 🔗 **Useful Links**

- 🤗 [Hugging Face Transformers](https://huggingface.co/transformers/)
- 🏦 [FinBERT Model](https://huggingface.co/ProsusAI/finbert)
- 📰 [News API](https://newsapi.org/)
- 📈 [Yahoo Finance API](https://pypi.org/project/yfinance/)
- 🐍 [Python Official](https://www.python.org/)
- 🔬 [PyTorch](https://pytorch.org/)

---

**Happy Analyzing! 📈🤖** 