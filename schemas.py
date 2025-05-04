from pydantic import BaseModel
from typing_extensions import TypedDict, Any


class State(TypedDict):
    '''Define app state and the data types that will flow through the app'''
    verbose: bool
    llm: Any
    business_goal: str
    raw_data_path: str
    raw_data_report: str
    clean_data_report: str
    model_name: str
    model_selection_report: str
    model_path: str
    evaluation_report: str
    summary: str


class TargetFeatureByLLM(BaseModel):
    '''If the model has to predict the most likely target feature in the dataset, configure it to provide it in a JSON format'''
    target: str
    reasoning: str
