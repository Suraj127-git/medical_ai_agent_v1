from flask import Blueprint, request, jsonify
from app.models.medical_query_model import MedicalQueryModel
from app.services.medical_crew_service import MedicalCrewService
from app.utils.logger import get_logger

logger = get_logger(__name__)
medical_query_bp = Blueprint('medical_query', __name__)

@medical_query_bp.route('/medical-query', methods=['POST'])
def medical_query():
    try:
        logger.debug("Received medical query request: %s", request.json)
        query_data = MedicalQueryModel(**request.json)
        
        logger.debug("Model validation successful")
        logger.info("Received query: %s", query_data.query)

        medical_crew = MedicalCrewService.create_medical_crew()
        
        logger.debug("Medical crew created")

        result = medical_crew.kickoff(inputs={
            "query": query_data.query,
            "current_date": "2025-01-30"
        })
        logger.debug("Medical crew kickoff successful")

        response_text = getattr(result, 'result', str(result))
        logger.info("Response text: %s", response_text)

        return jsonify({
            "query": query_data.query,
            "response": response_text
        })
        
    except ValidationError as e:
        logger.error("Input validation error: %s", str(e))
        return jsonify({
            "error": "Invalid request format",
            "details": e.errors(),
            "example": {"query": "What is dialysis?"}
        }), 400
    except Exception as e:
        logger.error("Unexpected error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "support": "contact support@medicalai.com"
        }), 500
