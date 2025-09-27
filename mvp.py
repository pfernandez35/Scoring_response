"""Generate top profile matches using lexical, emotional, and contextual signals."""

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return TOKEN_PATTERN.findall(text.lower())


def sentiment_to_scalar(label: str) -> float:
    if not label:
        return 0.5
    match = re.search(r"(\d)", label)
    if match:
        value = int(match.group(1))
        value = max(1, min(value, 5))
        return (value - 1) / 4
    label_lower = label.lower()
    mapping = {
        "very negative": 0.0,
        "negative": 0.25,
        "neutral": 0.5,
        "positive": 0.75,
        "very positive": 1.0,
    }
    return mapping.get(label_lower, 0.5)


def clean_column_label(label: str) -> str:
    if not label:
        return ""
    return re.sub(r"\s+", " ", label).strip()


@dataclass
class ProfileFeatures:
    index: int
    name: str
    answers: Dict[str, str]
    combined_text: str
    tokens: List[str]
    lexical_diversity: float
    word_count: int
    sentiment_label: str
    sentiment_score: float
    sentiment_value: float


class ProfileMatcher:
    def __init__(
        self,
        embedder_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        emotion_model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment",
    ) -> None:
        self.embedder_name = embedder_name
        self.emotion_model_name = emotion_model_name
        self.embedder = SentenceTransformer(embedder_name)
        self.emotion_classifier = pipeline("sentiment-analysis", model=emotion_model_name)
        self.features: List[ProfileFeatures] = []
        self.questions: List[str] = []
        self.content_similarity: Optional[np.ndarray] = None
        self.avg_question_similarity: Optional[np.ndarray] = None
        self.lexical_similarity: Optional[np.ndarray] = None
        self.length_similarity: Optional[np.ndarray] = None
        self.emotion_similarity: Optional[np.ndarray] = None
        self.composite_score: Optional[np.ndarray] = None
        self.question_similarity_mats: Dict[str, np.ndarray] = {}

    def fit(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Input dataframe is empty.")
        self.questions = list(df.columns[1:])
        features: List[ProfileFeatures] = []
        for idx, row in df.iterrows():
            name = str(row.iloc[0]).strip()
            answers: Dict[str, str] = {}
            for question in self.questions:
                value = row.get(question, "")
                if isinstance(value, float) and math.isnan(value):
                    value = ""
                answers[question] = str(value).strip()
            combined_text = " ".join(ans for ans in answers.values() if ans)
            tokens = tokenize(combined_text)
            word_count = len(tokens)
            lexical_diversity = len(set(tokens)) / word_count if word_count else 0.0
            truncated_text = combined_text[:512] if combined_text else "neutre"
            sentiment = self.emotion_classifier(truncated_text)[0]
            sentiment_label = sentiment.get("label", "")
            sentiment_score = float(sentiment.get("score", 0.0))
            sentiment_value = sentiment_to_scalar(sentiment_label)
            features.append(
                ProfileFeatures(
                    index=idx,
                    name=name,
                    answers=answers,
                    combined_text=combined_text,
                    tokens=tokens,
                    lexical_diversity=lexical_diversity,
                    word_count=word_count,
                    sentiment_label=sentiment_label,
                    sentiment_score=sentiment_score,
                    sentiment_value=sentiment_value,
                )
            )
        self.features = features
        self._compute_similarity_matrices()

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.embedder.encode(
                list(texts),
                batch_size=min(32, max(1, len(texts))),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )

    @staticmethod
    def _cosine_to_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        sims = embeddings @ embeddings.T
        sims = np.clip(sims, -1.0, 1.0)
        sims = (sims + 1.0) / 2.0
        return sims

    def _compute_similarity_matrices(self) -> None:
        if not self.features:
            raise RuntimeError("No features to compute similarities from.")
        combined_texts = [feat.combined_text if feat.combined_text else " " for feat in self.features]
        combined_embeddings = self._encode_texts(combined_texts)
        self.content_similarity = self._cosine_to_similarity_matrix(combined_embeddings)
        self.question_similarity_mats = {}
        if self.questions:
            question_sims = []
            for question in self.questions:
                texts = [feat.answers.get(question, "") or " " for feat in self.features]
                embeddings = self._encode_texts(texts)
                sim_matrix = self._cosine_to_similarity_matrix(embeddings)
                self.question_similarity_mats[question] = sim_matrix
                question_sims.append(sim_matrix)
            self.avg_question_similarity = np.mean(np.stack(question_sims), axis=0)
        else:
            self.avg_question_similarity = np.zeros_like(self.content_similarity)
        lexical = np.array([feat.lexical_diversity for feat in self.features], dtype=float)
        lexical_matrix = 1.0 - np.abs(lexical[:, None] - lexical[None, :])
        lexical_matrix = np.clip(lexical_matrix, 0.0, 1.0)
        self.lexical_similarity = lexical_matrix
        word_counts = np.array([feat.word_count for feat in self.features], dtype=float)
        if word_counts.size == 0:
            max_word_count = 1.0
        else:
            max_word_count = float(word_counts.max())
            if max_word_count == 0:
                max_word_count = 1.0
        length_matrix = 1.0 - np.abs(word_counts[:, None] - word_counts[None, :]) / max_word_count
        length_matrix = np.clip(length_matrix, 0.0, 1.0)
        self.length_similarity = length_matrix
        emotion_values = np.array([feat.sentiment_value for feat in self.features], dtype=float)
        emotion_matrix = 1.0 - np.abs(emotion_values[:, None] - emotion_values[None, :])
        emotion_matrix = np.clip(emotion_matrix, 0.0, 1.0)
        self.emotion_similarity = emotion_matrix
        composite = (
            0.5 * self.content_similarity
            + 0.2 * self.avg_question_similarity
            + 0.1 * self.lexical_similarity
            + 0.1 * self.length_similarity
            + 0.1 * self.emotion_similarity
        )
        np.fill_diagonal(composite, -1.0)
        self.composite_score = composite

    def top_matches(self, top_k: int = 10, return_review: bool = False):
        if not self.features or self.composite_score is None:
            raise RuntimeError("Model has not been fitted.")
        rows: List[Dict[str, str]] = []
        review_rows: List[Dict[str, object]] = []
        for base_idx, base in enumerate(self.features):
            sorted_indices = np.argsort(-self.composite_score[base_idx])
            rank = 1
            for candidate_idx in sorted_indices:
                if candidate_idx == base_idx:
                    continue
                candidate = self.features[candidate_idx]
                explanation = self._build_explanation(base_idx, candidate_idx)
                rows.append(
                    {
                        "Nom": base.name,
                        "Ranking": f"#{rank:02d}",
                        "Nom Ranking": candidate.name,
                        "Explication": explanation,
                    }
                )
                if return_review:
                    review_rows.append(self._build_review_row(base_idx, candidate_idx, rank))
                rank += 1
                if rank > top_k:
                    break
        top_df = pd.DataFrame(rows)
        if return_review:
            review_df = pd.DataFrame(review_rows)
            return top_df, review_df
        return top_df

    def _best_question_alignment(self, base_idx: int, candidate_idx: int) -> Optional[Tuple[str, float]]:
        if not self.question_similarity_mats:
            return None
        question, score = max(
            (
                (question, self.question_similarity_mats[question][base_idx, candidate_idx])
                for question in self.questions
            ),
            key=lambda item: item[1],
        )
        return question, float(score)

    def _build_explanation(self, base_idx: int, candidate_idx: int) -> str:
        base = self.features[base_idx]
        candidate = self.features[candidate_idx]
        fragments: List[str] = []
        base_first = base.name.split()[0] if base.name else "Ce profil"
        opener = f"{base_first}, {candidate.name} se distingue par une compatibilite elevee avec ton profil."
        fragments.append(opener)
        if self.emotion_similarity is not None:
            if base.sentiment_label and base.sentiment_label == candidate.sentiment_label:
                fragments.append(
                    f"Vos reponses partagent la meme tonalite emotionnelle ({base.sentiment_label})."
                )
            elif self.emotion_similarity[base_idx, candidate_idx] > 0.85:
                fragments.append(
                    "Vos ressentis se repondent presque a l'identique, un vrai atout pour l'affinite."
                )
        alignment = self._best_question_alignment(base_idx, candidate_idx)
        if alignment is not None:
            question, score = alignment
            if score > 0.55:
                fragments.append(
                    f"Vous avez une vision tres proche sur la question \"{question}\"."
                )
        if self.lexical_similarity is not None and self.lexical_similarity[base_idx, candidate_idx] > 0.8:
            fragments.append("Vos styles d'expression affichent une richesse tres proche.")
        elif self.length_similarity is not None and self.length_similarity[base_idx, candidate_idx] > 0.8:
            fragments.append("Vous partagez un niveau de detail similaire dans vos reponses.")
        if self.content_similarity is not None and self.content_similarity[base_idx, candidate_idx] > 0.75:
            fragments.append("La proximite globale de vos narrations renforce ce match.")
        return " ".join(fragments)

    def _build_review_row(self, base_idx: int, candidate_idx: int, rank: int) -> Dict[str, object]:
        base = self.features[base_idx]
        candidate = self.features[candidate_idx]
        review_row: Dict[str, object] = {
            "Profil source": base.name,
            "Profil compare": candidate.name,
            "Rang": rank,
        }
        composite = max(0.0, float(self.composite_score[base_idx, candidate_idx]))
        review_row["Score global"] = round(composite * 100, 2)
        if self.content_similarity is not None:
            review_row["Similarite contenu"] = round(
                max(0.0, float(self.content_similarity[base_idx, candidate_idx])) * 100, 2
            )
        if self.avg_question_similarity is not None:
            review_row["Similarite moyenne questions"] = round(
                max(0.0, float(self.avg_question_similarity[base_idx, candidate_idx])) * 100, 2
            )
        if self.lexical_similarity is not None:
            review_row["Similarite lexicale"] = round(
                max(0.0, float(self.lexical_similarity[base_idx, candidate_idx])) * 100, 2
            )
        if self.length_similarity is not None:
            review_row["Similarite longueur"] = round(
                max(0.0, float(self.length_similarity[base_idx, candidate_idx])) * 100, 2
            )
        if self.emotion_similarity is not None:
            review_row["Similarite emotionnelle"] = round(
                max(0.0, float(self.emotion_similarity[base_idx, candidate_idx])) * 100, 2
            )
        alignment = self._best_question_alignment(base_idx, candidate_idx)
        if alignment is not None:
            question, score = alignment
            review_row["Question la plus proche"] = clean_column_label(question)
            review_row["Similarite question la plus proche"] = round(max(0.0, score) * 100, 2)
        review_row["Sentiment source"] = base.sentiment_label
        review_row["Sentiment compare"] = candidate.sentiment_label
        review_row["Score sentiment source"] = round(base.sentiment_score, 3)
        review_row["Score sentiment compare"] = round(candidate.sentiment_score, 3)
        for question in self.questions:
            col_source = f"Reponse source - {clean_column_label(question)}"
            col_compare = f"Reponse compare - {clean_column_label(question)}"
            review_row[col_source] = base.answers.get(question, "")
            review_row[col_compare] = candidate.answers.get(question, "")
        return review_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcule un top K de profils similaires a partir de reponses libres."
    )
    parser.add_argument(
        "--input",
        default="data/topten-answers-week1.csv",
        help="Chemin du fichier source (CSV separe par des points-virgules).",
    )
    parser.add_argument(
        "--output",
        default="data/top_ten_answers.csv",
        help="Chemin du fichier de sortie contenant les top matches.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Nombre de correspondances a retourner par profil.",
    )
    parser.add_argument(
        "--embedder",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Modele sentence-transformers a utiliser pour les embeddings.",
    )
    parser.add_argument(
        "--emotion-model",
        default="nlptown/bert-base-multilingual-uncased-sentiment",
        help="Modele Transformers pour l'analyse des emotions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, sep=";")
    matcher = ProfileMatcher(embedder_name=args.embedder, emotion_model_name=args.emotion_model)
    matcher.fit(df)
    top_df, review_df = matcher.top_matches(top_k=args.top_k, return_review=True)
    top_df.to_csv(args.output, sep=";", index=False)
    review_path = Path(args.output).with_name("review_score_per_profile.csv")
    review_df.to_csv(review_path, sep=";", index=False)


if __name__ == "__main__":
    main()
