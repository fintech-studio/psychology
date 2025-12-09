# Services package

from .analysis_service import AnalysisService
from .gemini_service import GeminiService
from .questionnaire_service import QuestionnaireService

# 初始化服務
# Services: create objects only; any I/O or network health checks should be executed asynchronously on startup
analysisService = AnalysisService()
geminiService = GeminiService()
# Ollama-based alias (保留向後相容性)
ollamaService = geminiService
questionnaireService = QuestionnaireService()
