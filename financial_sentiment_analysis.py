import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from bs4 import BeautifulSoup
import yfinance as yf
from newsapi import NewsApiClient
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class FinancialSentimentAnalyzer:
    def __init__(self):
        """Initialize the Financial Sentiment Analyzer with pre-trained models."""
        print("Initializing Financial Sentiment Analyzer...")
        
        # Initialize sentiment analysis pipeline with FinBERT
        # FinBERT is specifically trained on financial text
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✓ FinBERT model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load FinBERT, falling back to general model: {e}")
            # Fallback to general sentiment model
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
        
        # Initialize News API client (optional - requires API key)
        self.news_api_key = os.getenv('NEWS_API_KEY')
        if self.news_api_key:
            self.newsapi = NewsApiClient(api_key=self.news_api_key)
            print("✓ News API initialized")
        else:
            self.newsapi = None
            print("⚠ News API key not found - will use alternative sources")
    
    def get_sample_headlines(self):
        """Get sample financial headlines for demonstration."""
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
        
        # Create timestamps for the past month
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        headlines_data = []
        for i, headline in enumerate(sample_headlines):
            # Distribute headlines across the past month
            days_back = (i * 2) % 30
            timestamp = end_date - timedelta(days=days_back)
            
            headlines_data.append({
                'headline': headline,
                'timestamp': timestamp,
                'source': f'Financial News {i % 5 + 1}'
            })
        
        return pd.DataFrame(headlines_data)
    
    def get_yahoo_finance_headlines(self, symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']):
        """Fetch recent headlines from Yahoo Finance for given symbols."""
        headlines_data = []
        
        print("Fetching headlines from Yahoo Finance...")
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                news = ticker.news
                
                for article in news[:5]:  # Get top 5 articles per symbol
                    headlines_data.append({
                        'headline': article.get('title', ''),
                        'timestamp': datetime.fromtimestamp(article.get('providerPublishTime', 0)),
                        'source': article.get('publisher', 'Yahoo Finance'),
                        'symbol': symbol,
                        'url': article.get('link', '')
                    })
            except Exception as e:
                print(f"Error fetching news for {symbol}: {e}")
                continue
        
        return pd.DataFrame(headlines_data)
    
    def get_newsapi_headlines(self, days_back=30):
        """Fetch financial headlines using News API."""
        if not self.newsapi:
            return pd.DataFrame()
        
        print("Fetching headlines from News API...")
        headlines_data = []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        try:
            # Search for financial news
            articles = self.newsapi.get_everything(
                q='finance OR stock OR market OR economy OR earnings',
                language='en',
                sort_by='publishedAt',
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                page_size=50
            )
            
            for article in articles['articles']:
                headlines_data.append({
                    'headline': article['title'],
                    'timestamp': datetime.strptime(article['publishedAt'][:19], '%Y-%m-%dT%H:%M:%S'),
                    'source': article['source']['name'],
                    'url': article['url'],
                    'description': article.get('description', '')
                })
                
        except Exception as e:
            print(f"Error fetching from News API: {e}")
        
        return pd.DataFrame(headlines_data)
    
    def analyze_sentiment(self, headlines_df):
        """Analyze sentiment of headlines using the transformer model."""
        print("Analyzing sentiment...")
        
        sentiments = []
        scores = []
        
        for headline in headlines_df['headline']:
            try:
                # Get sentiment prediction
                result = self.sentiment_pipeline(headline)[0]
                
                # Normalize labels for different models
                label = result['label'].upper()
                if label in ['POSITIVE', 'POS', 'LABEL_2']:
                    sentiment = 'positive'
                elif label in ['NEGATIVE', 'NEG', 'LABEL_0']:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                sentiments.append(sentiment)
                scores.append(result['score'])
                
            except Exception as e:
                print(f"Error analyzing headline: {headline[:50]}... - {e}")
                sentiments.append('neutral')
                scores.append(0.5)
        
        headlines_df['sentiment'] = sentiments
        headlines_df['confidence'] = scores
        
        return headlines_df
    
    def create_visualizations(self, df):
        """Create comprehensive visualizations of the sentiment analysis."""
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Financial Headlines Sentiment Analysis - Past Month', fontsize=16, fontweight='bold')
        
        # Define consistent color mapping
        color_map = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}  # green, red, grey
        
        # 1. Sentiment Distribution
        sentiment_counts = df['sentiment'].value_counts()
        # Ensure colors match sentiment order
        colors = [color_map[sentiment] for sentiment in sentiment_counts.index]
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                       colors=colors, startangle=90)
        axes[0, 0].set_title('Overall Sentiment Distribution')
        
        # 2. Sentiment Over Time
        df['date'] = df['timestamp'].dt.date
        daily_sentiment = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        # Reorder columns to ensure consistent color mapping
        column_order = ['positive', 'negative', 'neutral']
        daily_sentiment = daily_sentiment.reindex(columns=[col for col in column_order if col in daily_sentiment.columns])
        plot_colors = [color_map[col] for col in daily_sentiment.columns]
        daily_sentiment.plot(kind='bar', stacked=True, ax=axes[0, 1], color=plot_colors)
        axes[0, 1].set_title('Daily Sentiment Trends')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Number of Headlines')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend(title='Sentiment')
        
        # 3. Confidence Score Distribution
        axes[1, 0].hist(df['confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Confidence Score Distribution')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(df['confidence'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["confidence"].mean():.2f}')
        axes[1, 0].legend()
        
        # 4. Sentiment by Source (if available)
        if 'source' in df.columns:
            source_sentiment = df.groupby(['source', 'sentiment']).size().unstack(fill_value=0)
            # Reorder columns to ensure consistent color mapping
            source_sentiment = source_sentiment.reindex(columns=[col for col in column_order if col in source_sentiment.columns])
            plot_colors = [color_map[col] for col in source_sentiment.columns]
            source_sentiment.plot(kind='barh', ax=axes[1, 1], color=plot_colors)
            axes[1, 1].set_title('Sentiment by News Source')
            axes[1, 1].set_xlabel('Number of Headlines')
            axes[1, 1].legend(title='Sentiment')
        else:
            # Alternative: Show sentiment score over time
            df_sorted = df.sort_values('timestamp')
            sentiment_numeric = df_sorted['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
            axes[1, 1].plot(range(len(sentiment_numeric)), sentiment_numeric, marker='o', alpha=0.6)
            axes[1, 1].set_title('Sentiment Score Timeline')
            axes[1, 1].set_ylabel('Sentiment Score')
            axes[1, 1].set_xlabel('Headlines (chronological)')
            axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('financial_sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_insights(self, df):
        """Generate insights from the sentiment analysis."""
        print("\n" + "="*60)
        print("FINANCIAL SENTIMENT ANALYSIS INSIGHTS")
        print("="*60)
        
        # Basic statistics
        total_headlines = len(df)
        sentiment_dist = df['sentiment'].value_counts(normalize=True) * 100
        avg_confidence = df['confidence'].mean()
        
        print(f"📊 Total Headlines Analyzed: {total_headlines}")
        print(f"🎯 Average Confidence Score: {avg_confidence:.2f}")
        print("\n📈 Sentiment Distribution:")
        for sentiment, percentage in sentiment_dist.items():
            emoji = "🟢" if sentiment == 'positive' else "🔴" if sentiment == 'negative' else "🟡"
            print(f"   {emoji} {sentiment.capitalize()}: {percentage:.1f}%")
        
        # Time-based insights
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            peak_hour = df['hour'].mode().iloc[0]
            print(f"\n⏰ Peak News Hour: {peak_hour}:00")
            
            # Recent trend (last 7 days vs previous)
            recent_cutoff = datetime.now() - timedelta(days=7)
            recent_df = df[df['timestamp'] >= recent_cutoff]
            older_df = df[df['timestamp'] < recent_cutoff]
            
            if len(recent_df) > 0 and len(older_df) > 0:
                recent_positive = (recent_df['sentiment'] == 'positive').mean() * 100
                older_positive = (older_df['sentiment'] == 'positive').mean() * 100
                trend = recent_positive - older_positive
                
                trend_emoji = "📈" if trend > 5 else "📉" if trend < -5 else "➡️"
                print(f"\n{trend_emoji} Recent Trend (Last 7 days):")
                print(f"   Positive sentiment: {recent_positive:.1f}% (vs {older_positive:.1f}% previously)")
                print(f"   Change: {trend:+.1f} percentage points")
        
        # High confidence insights
        high_conf_df = df[df['confidence'] > 0.8]
        if len(high_conf_df) > 0:
            high_conf_sentiment = high_conf_df['sentiment'].value_counts(normalize=True) * 100
            print(f"\n🎯 High Confidence Predictions (>{0.8:.1f} confidence, {len(high_conf_df)} headlines):")
            for sentiment, percentage in high_conf_sentiment.items():
                emoji = "🟢" if sentiment == 'positive' else "🔴" if sentiment == 'negative' else "🟡"
                print(f"   {emoji} {sentiment.capitalize()}: {percentage:.1f}%")
        
        # Sample headlines by sentiment
        print(f"\n📰 Sample Headlines by Sentiment:")
        for sentiment in ['positive', 'negative', 'neutral']:
            sample_headlines = df[df['sentiment'] == sentiment]['headline'].head(2)
            if len(sample_headlines) > 0:
                emoji = "🟢" if sentiment == 'positive' else "🔴" if sentiment == 'negative' else "🟡"
                print(f"\n{emoji} {sentiment.capitalize()} Examples:")
                for headline in sample_headlines:
                    print(f"   • {headline}")
        
        print("\n" + "="*60)
    
    def save_results(self, df, filename='financial_sentiment_results.csv'):
        """Save the analysis results to a CSV file."""
        df.to_csv(filename, index=False)
        print(f"💾 Results saved to {filename}")
    
    def run_analysis(self):
        """Run the complete sentiment analysis pipeline."""
        print("🚀 Starting Financial Sentiment Analysis...")
        print("-" * 50)
        
        # Collect headlines from multiple sources
        all_headlines = []
        
        # Try to get real headlines first
        try:
            yahoo_headlines = self.get_yahoo_finance_headlines()
            if not yahoo_headlines.empty:
                all_headlines.append(yahoo_headlines)
                print(f"✓ Collected {len(yahoo_headlines)} headlines from Yahoo Finance")
        except Exception as e:
            print(f"⚠ Could not fetch Yahoo Finance headlines: {e}")
        
        if self.newsapi:
            try:
                news_headlines = self.get_newsapi_headlines()
                if not news_headlines.empty:
                    all_headlines.append(news_headlines)
                    print(f"✓ Collected {len(news_headlines)} headlines from News API")
            except Exception as e:
                print(f"⚠ Could not fetch News API headlines: {e}")
        
        # If no real headlines, use sample data
        if not all_headlines:
            print("📝 Using sample headlines for demonstration")
            sample_headlines = self.get_sample_headlines()
            all_headlines.append(sample_headlines)
        
        # Combine all headlines
        if len(all_headlines) > 1:
            headlines_df = pd.concat(all_headlines, ignore_index=True)
        else:
            headlines_df = all_headlines[0]
        
        # Remove duplicates
        headlines_df = headlines_df.drop_duplicates(subset=['headline']).reset_index(drop=True)
        print(f"📊 Total unique headlines: {len(headlines_df)}")
        
        # Analyze sentiment
        results_df = self.analyze_sentiment(headlines_df)
        
        # Create visualizations
        self.create_visualizations(results_df)
        
        # Generate insights
        self.generate_insights(results_df)
        
        # Save results
        self.save_results(results_df)
        
        return results_df

def main():
    """Main function to run the financial sentiment analysis."""
    # Initialize analyzer
    analyzer = FinancialSentimentAnalyzer()
    
    # Run analysis
    results = analyzer.run_analysis()
    
    print("\n🎉 Analysis complete! Check the generated files:")
    print("   • financial_sentiment_analysis.png - Visualization charts")
    print("   • financial_sentiment_results.csv - Detailed results")
    
    return results

if __name__ == "__main__":
    results = main() 