from typing import Dict, List

import numpy as np
import openai
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.infra.logging import logger


class SentimentAnalyzer:
    """Advanced financial sentiment analysis with LLMs and dimensionality reduction."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.pca = PCA(n_components=3)
        self.cache_dir = settings.PROCESSED_DATA_DIR / "sentiment_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def analyze_transcript(self, text: str, ticker: str) -> Dict[str, float]:
        """Analyze earnings call transcript using LLM with cache."""
        cache_file = self.cache_dir / f"{ticker}_{hash(text)}.parquet"

        if cache_file.exists():
            return pd.read_parquet(cache_file).to_dict("records")[0]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst specializing in sentiment analysis. Provide quantitative scores for the following aspects:",
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this {ticker} earnings transcript:\n\n{text}\n\nProvide scores for:",
                    },
                ],
                temperature=0,
                max_tokens=500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            # Parse response into structured format
            result = self._parse_llm_response(response.choices[0].message.content)
            pd.DataFrame([result]).to_parquet(cache_file)

            return result
        except Exception as e:
            logger.error("LLM sentiment analysis failed", error=str(e))
            raise

    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Create dense embeddings of text using sentence transformers."""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        reduced_embeddings = self.pca.fit_transform(embeddings)
        return reduced_embeddings

    def create_sentiment_features(self, transcripts: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive sentiment features from transcripts."""
        features = pd.DataFrame(index=transcripts["date"].unique())

        # LLM-based sentiment scores
        sentiment_scores = []
        for _, row in transcripts.iterrows():
            scores = self.analyze_transcript(row["text"], row["ticker"])
            scores["date"] = row["date"]
            sentiment_scores.append(scores)

        sentiment_df = pd.DataFrame(sentiment_scores).set_index("date")
        features = features.join(sentiment_df)

        # Text embedding features
        texts = transcripts.groupby("date")["text"].apply(lambda x: " ".join(x))
        embeddings = self.embed_text(texts.tolist())
        for i in range(embeddings.shape[1]):
            features[f"text_embedding_{i}"] = embeddings[:, i]

        # Rolling sentiment metrics
        for col in sentiment_df.columns:
            if sentiment_df[col].dtype in [np.float64, np.int64]:
                features[f"{col}_zscore"] = (
                    sentiment_df[col] - sentiment_df[col].rolling(21).mean()
                ) / sentiment_df[col].rolling(21).std()

                features[f"{col}_trend"] = (
                    sentiment_df[col].rolling(5).mean()
                    - sentiment_df[col].rolling(21).mean()
                )

        return features.dropna()

    @staticmethod
    def _parse_llm_response(text: str) -> Dict[str, float]:
        """Parse LLM response into structured sentiment scores."""
        # This would be customized based on your prompt engineering
        return {
            "sentiment_score": 0.5,  # Placeholder
            "uncertainty_score": 0.3,
            "growth_mentions": 0.7,
            "risk_mentions": 0.4,
        }
