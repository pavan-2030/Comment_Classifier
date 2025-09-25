# wordcloud.py
from wordcloud import WordCloud, STOPWORDS
import io
import base64
import re

def preprocess_comment(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'^\d+\.\s*', '', text.strip())
    text = re.sub(r'\s+', ' ', text)
    text = text.lower() 
    return text.strip()

def generate_wordcloud(comments: list) -> str:
    cleaned_comments = [preprocess_comment(c) for c in comments if c.strip()]
    combined_text = " ".join(cleaned_comments)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=STOPWORDS,
        collocations=True
    ).generate(combined_text)
    img_bytes = io.BytesIO()
    wordcloud.to_image().save(img_bytes, format='PNG')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    return img_base64
