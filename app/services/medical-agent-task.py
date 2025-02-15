import os
import json
import yaml
import requests
from crewai import Agent, Task, Crew
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from app.utils.logger import get_logger
from typing import Union

logger = get_logger(__name__)

def load_config(config_path: str = "medical_config.yaml") -> dict:
    """
    Load static configuration from a YAML file.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

# Global configuration object
CONFIG = load_config()

class MedicalQueryInput(BaseModel):
    query: str = Field(..., description="Medical topic search query as string")

class MedicalCrewService:
    @classmethod
    def create_medical_crew(cls, llm, max_iterations: int = 3):
        """
        Factory method to create a configured medical research crew.
        """
        try:
            logger.info(f"Initializing medical crew with LLM: {llm.model} and max_iterations: {max_iterations}")

            def parse_input(input_data: Union[dict, str]) -> str:
                """Universal input parser with validation."""
                logger.debug(f"Parsing input data: {input_data}")
                try:
                    if isinstance(input_data, str):
                        logger.debug("Converting string input to JSON")
                        input_data = json.loads(input_data)
                    
                    query_data = input_data.get("query", "")
                    if isinstance(query_data, dict):
                        query = query_data.get("query", "")
                        logger.debug(f"Extracted nested query: {query}")
                        return query
                    
                    logger.debug(f"Extracted direct query: {query_data}")
                    return query_data
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.warning(f"Input parsing error: {e}, raw input: {input_data}")
                    return ""

            def fetch_clinical_data(query: str) -> dict:
                """Fetch real clinical data from the NIH API."""
                logger.info(f"Fetching clinical data for query: {query}")
                try:
                    base_url = CONFIG["nih_api"]["base_url"]
                    params = {
                        "terms": query,
                        "df": CONFIG["nih_api"]["params"]["df"],
                        "maxList": CONFIG["nih_api"]["params"]["maxList"]
                    }
                    timeout = CONFIG.get("request", {}).get("timeout", 300)
                    logger.debug(f"Making API request to {base_url} with params: {params} and timeout: {timeout}")

                    response = requests.get(base_url, params=params, timeout=timeout)
                    response.raise_for_status()
                    data = response.json()
                    logger.debug(f"Raw API response: {data}")
                    
                    if data[0] == 0:
                        logger.warning(f"No clinical data found for query: {query}")
                        return {"error": "No clinical data found"}
                    
                    results = []
                    for item in data[3]:
                        try:
                            icd10_raw = item[1]
                            if icd10_raw:
                                icd10_data = json.loads(icd10_raw)
                                if isinstance(icd10_data, dict):
                                    icd10_codes = [icd10_data.get("code", "")]
                                else:
                                    icd10_codes = [entry.get("code", "") for entry in icd10_data]
                            else:
                                icd10_codes = []
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.error(f"JSON parsing error for ICD-10 data: {e}")
                            icd10_codes = []
                        
                        result = {
                            "primary_name": item[0],
                            "icd10_codes": icd10_codes
                        }
                        results.append(result)
                    
                    logger.info(f"Found {len(results)} clinical results")
                    return {"count": data[0], "results": results}
                    
                except requests.exceptions.Timeout:
                    logger.error(f"NIH API timeout after {timeout}s for query: {query}")
                    return {"error": "API request timed out"}
                except requests.exceptions.RequestException as e:
                    logger.error(f"NIH API request failed: {e}")
                    return {"error": f"API request failed: {e}"}
                except Exception as e:
                    logger.error(f"Clinical data processing error: {e}")
                    return {"error": f"Failed to process clinical data: {e}"}

            def clinical_search(**kwargs) -> str:
                """Dynamic clinical knowledge base combining LLM and NIH data."""
                logger.info("Starting clinical search")
                query = kwargs.get("query", "")
                logger.debug(f"Parsed query: {query}")
                
                if not query:
                    logger.warning("Empty query received")
                    return "Validation Error: Empty medical query"
                
                api_data = fetch_clinical_data(query)
                logger.debug(f"Clinical data response: {api_data}")
                
                if "error" in api_data:
                    logger.error(f"Clinical data fetch failed: {api_data['error']}")
                    return f"Clinical Data Error: {api_data['error']}"
                
                prompt_template = CONFIG["prompt"]["clinical_summary"]
                prompt = prompt_template.format(query=query, api_data=json.dumps(api_data))
                logger.debug(f"LLM prompt: {prompt}")
                
                try:
                    response = llm.chat(prompt)
                    logger.info("Successfully generated clinical summary")
                    logger.debug(f"LLM response: {response}")
                    return response
                except Exception as e:
                    logger.error(f"LLM chat failed: {e}", exc_info=True)
                    return f"System Error: Clinical search unavailable - {e}"

            logger.info("Configuring clinical tool")
            clinical_tool = Tool.from_function(
                func=clinical_search,
                name=CONFIG["tool"]["name"],
                description=CONFIG["tool"]["description"],
                args_schema=MedicalQueryInput
            )

            logger.info("Creating researcher agent")
            researcher = Agent(
                role=CONFIG["agent"]["role"],
                goal=CONFIG["agent"]["goal"],
                backstory=CONFIG["agent"]["backstory"],
                tools=[clinical_tool],
                verbose=True,
                llm=llm,
                max_iter=max_iterations,
                memory=True
            )

            logger.info("Creating research task")
            research_task = Task(
                description=CONFIG["task"]["description"],
                expected_output=CONFIG["task"]["expected_output"],
                agent=researcher,
                input_schema=MedicalQueryInput
            )

            logger.info("Initializing crew")
            return Crew(
                agents=[researcher],
                tasks=[research_task],
                verbose=True,
                memory=False,
                process="sequential"
            )

        except Exception as e:
            logger.critical(f"Medical crew initialization failed: {e}", exc_info=True)
            raise RuntimeError("Medical AI system initialization error") from e
