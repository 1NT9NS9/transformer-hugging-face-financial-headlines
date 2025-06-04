"""
Simple Financial Sentiment Analysis Example
Using Hugging Face Transformers for Financial Headlines

This script demonstrates how to perform sentiment analysis on financial headlines
using pre-trained transformer models, specifically FinBERT.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from transformers import pipeline
import torch

def analyze_financial_sentiment():
    """Simple example of financial sentiment analysis."""
    
    print("🚀 Financial Sentiment Analysis with Hugging Face Transformers")
    print("=" * 60)
    
    # Sample financial headlines from the past month
    sample_headlines = [
        "Apple reports record quarterly earnings beating analyst expectations",
        "Tesla stock plunges after disappointing delivery numbers", 
        "Federal Reserve hints at potential interest rate cuts next quarter",
        "Major bank announces significant layoffs amid economic uncertainty",
        "Tech stocks rally as inflation shows signs of cooling",
        "Oil prices surge following geopolitical tensions in Middle East",
        "Cryptocurrency market crashes amid regulatory concerns",
        "Amazon announces massive investment in AI infrastructure",
        "Housing market shows resilience despite rising mortgage rates",
        "Gold prices hit new highs as investors seek safe haven assets",
        "Retail sales disappoint as consumer spending slows down",
        "Biotech company's breakthrough drug receives FDA approval",
        "Energy sector faces headwinds from renewable transition",
        "Financial markets volatile ahead of key economic data release",
        "Merger announcement sends both company stocks soaring"
    ]
    
    # Create DataFrame with timestamps
    end_date = datetime.now()
    headlines_data = []
    
    for i, headline in enumerate(sample_headlines):
        days_back = (i * 2) % 30  # Distribute over past month
        timestamp = end_date - timedelta(days=days_back)
        headlines_data.append({
            'headline': headline,
            'timestamp': timestamp,
            'source': f'Financial News {i % 3 + 1}'
        })
    
    df = pd.DataFrame(headlines_data)
    print(f"📊 Analyzing {len(df)} financial headlines...")
    
    # Initialize sentiment analysis pipeline
    print("\n🤖 Loading FinBERT model (specialized for financial text)...")
    try:
        # Try to load FinBERT - specifically trained on financial text
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=0 if torch.cuda.is_available() else -1
        )
        print("✅ FinBERT loaded successfully!")
    except Exception as e:
        print(f"⚠️  FinBERT not available, using general model: {e}")
        # Fallback to general sentiment model
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if torch.cuda.is_available() else -1
        )
    
    # Analyze sentiment for each headline
    print("\n🔍 Analyzing sentiment...")
    sentiments = []
    confidence_scores = []
    
    for headline in df['headline']:
        result = sentiment_pipeline(headline)[0]
        
        # Normalize labels (different models use different labels)
        label = result['label'].upper()
        if label in ['POSITIVE', 'POS', 'LABEL_2']:
            sentiment = 'positive'
        elif label in ['NEGATIVE', 'NEG', 'LABEL_0']:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        sentiments.append(sentiment)
        confidence_scores.append(result['score'])
    
    # Add results to DataFrame
    df['sentiment'] = sentiments
    df['confidence'] = confidence_scores
    
    # Display results
    print("\n📈 SENTIMENT ANALYSIS RESULTS")
    print("-" * 40)
    
    # Overall sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    sentiment_percentages = df['sentiment'].value_counts(normalize=True) * 100
    
    print("Overall Sentiment Distribution:")
    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment in sentiment_counts:
            count = sentiment_counts[sentiment]
            percentage = sentiment_percentages[sentiment]
            emoji = "🟢" if sentiment == 'positive' else "🔴" if sentiment == 'negative' else "🟡"
            print(f"  {emoji} {sentiment.capitalize()}: {count} headlines ({percentage:.1f}%)")
    
    print(f"\n🎯 Average Confidence Score: {df['confidence'].mean():.3f}")
    
    # Show sample headlines by sentiment
    print("\n📰 Sample Headlines by Sentiment:")
    for sentiment in ['positive', 'negative', 'neutral']:
        sample = df[df['sentiment'] == sentiment].head(2)
        if not sample.empty:
            emoji = "🟢" if sentiment == 'positive' else "🔴" if sentiment == 'negative' else "🟡"
            print(f"\n{emoji} {sentiment.upper()} Examples:")
            for _, row in sample.iterrows():
                print(f"   • {row['headline']} (confidence: {row['confidence']:.3f})")
    
    # Create simple visualization
    create_simple_visualization(df)
    
    # Save results
    df.to_csv('sentiment_results.csv', index=False)
    print(f"\n💾 Results saved to 'sentiment_results.csv'")
    
    return df

def create_simple_visualization(df):
    """Create a simple visualization of the sentiment analysis results."""
    
    plt.figure(figsize=(12, 8))
    
    # Define consistent color mapping
    color_map = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}  # green, red, grey
    
    # Subplot 1: Sentiment Distribution Pie Chart
    plt.subplot(2, 2, 1)
    sentiment_counts = df['sentiment'].value_counts()
    # Ensure colors match sentiment order
    colors = [color_map[sentiment] for sentiment in sentiment_counts.index]
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Sentiment Distribution')
    
    # Subplot 2: Confidence Score Distribution
    plt.subplot(2, 2, 2)
    plt.hist(df['confidence'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(df['confidence'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["confidence"].mean():.3f}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Score Distribution')
    plt.legend()
    
    # Subplot 3: Sentiment Over Time
    plt.subplot(2, 2, 3)
    df_sorted = df.sort_values('timestamp')
    sentiment_numeric = df_sorted['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
    plt.plot(range(len(sentiment_numeric)), sentiment_numeric, marker='o', alpha=0.7)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Headlines (chronological order)')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Timeline')
    plt.ylim(-1.2, 1.2)
    
    # Subplot 4: Sentiment by Source
    plt.subplot(2, 2, 4)
    source_sentiment = df.groupby(['source', 'sentiment']).size().unstack(fill_value=0)
    # Reorder columns to ensure consistent color mapping
    column_order = ['positive', 'negative', 'neutral']
    source_sentiment = source_sentiment.reindex(columns=[col for col in column_order if col in source_sentiment.columns])
    plot_colors = [color_map[col] for col in source_sentiment.columns]
    source_sentiment.plot(kind='bar', ax=plt.gca(), color=plot_colors)
    plt.xlabel('News Source')
    plt.ylabel('Number of Headlines')
    plt.title('Sentiment by Source')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('financial_sentiment_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Visualization saved as 'financial_sentiment_chart.png'")

if __name__ == "__main__":
    # Run the analysis
    results = analyze_financial_sentiment()
    
    print("\n🎉 Analysis Complete!")
    print("\nGenerated Files:")
    print("  • sentiment_results.csv - Detailed results")
    print("  • financial_sentiment_chart.png - Visualization charts")
    print("\nTo run this script:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run: python simple_example.py") 