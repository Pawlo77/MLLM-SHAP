"""Coalition utility scoring plugins.

@domain value
@see [[knowledge/architecture/value]]
"""

from .bertscore import BertScoreValue
from .contrastive import ContrastiveValue
from .embedding import CosineEmbeddingValue
from .judge import JudgeGradeValue
from .logprob import LogprobValue, SelfLogprobValue
from .nli import NliLogOddsValue
from .semascore import SemaScoreValue
from .speech import SpeechBertScoreValue, TranscribedValue
from .teacher_forced import TfJsdValue, TfLogprobValue
from .tfidf import TfIdfCosineValue
from .wer import ReferenceWerValue, word_error_rate

__all__ = [
    "BertScoreValue",
    "ContrastiveValue",
    "CosineEmbeddingValue",
    "JudgeGradeValue",
    "LogprobValue",
    "NliLogOddsValue",
    "ReferenceWerValue",
    "SelfLogprobValue",
    "SemaScoreValue",
    "SpeechBertScoreValue",
    "TfIdfCosineValue",
    "TfJsdValue",
    "TfLogprobValue",
    "TranscribedValue",
    "word_error_rate",
]
