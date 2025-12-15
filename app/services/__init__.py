# Services package

from .analysis_service import AnalysisService
from .ollama_service import OllamaService
from .questionnaire_service import QuestionnaireService

# 初始化服務
# Services: create objects only;
# any I/O or network health checks should be executed asynchronously on startup
# Instantiate lazily / tolerantly so importing the package does not fail when
# optional heavy deps (e.g. torch) are not available in the environment.
try:
    analysisService = AnalysisService()
except Exception:
    analysisService = None

try:
    ollamaService = OllamaService()
except Exception:
    ollamaService = None

try:
    questionnaireService = QuestionnaireService()
except Exception:
    questionnaireService = None
