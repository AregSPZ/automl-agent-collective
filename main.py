import os
import re
import streamlit as st
from utils import clear_files, ext_exists
from schemas import State
from prompts import agent_actions, prompts
from tools import toolkit
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent
import tempfile # Import tempfile
import pandas as pd # Import pandas if not already imported


model_extensions = ['pkl', 'sav', 'joblib', 'pt', 'pth', 'h5', 'keras', 'tflite', 'pmml', 'onnx', 'json']
# Clear files before the run
clear_files(['csv', 'png', 'jpg'] + model_extensions)

# Setup
#os.environ["GOOGLE_API_KEY"] = google_api_key
# langsmith is useful for tracking what your app's LLMs do under the hood
#os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
#os.environ["LANGSMITH_TRACING"] = "true"

# define the base LLM which will be used to power each agent
llm = ChatGoogleGenerativeAI(
    # conservative settings to ensure precision in responses
    model="gemini-2.0-flash", # Changed model name
    timeout=None,
    max_retries=2,
    temperature=0,
    top_p=0.8,
    top_k=5
)

# initialize the agents with their corresponding tools
agents = {agent: create_react_agent(llm, agent_tools) for agent, agent_tools in toolkit.items()}

# define the application steps (the action of each agent)
# all the steps follow the same format: pass the necessary variables to the agent's prompt, invoke the agent, add the agent's output to the app's state

def frame_problem(state: State):
    '''Agent 1: Problem Framer - Frame the problem, extract key insights about the data'''
    st.subheader("1. Framing the problem")
    # get the agent's prompt
    problem_framer_prompt = prompts['problem_framer'].invoke(
        {
        "action": agent_actions['problem_framer'], "business_goal": state["business_goal"],
        "raw_data_path": state["raw_data_path"] # This will now be a file path string
        }
    ).text

    # invoke the agent
    response = agents['problem_framer'].invoke(
        {"messages": [HumanMessage(content=problem_framer_prompt)]})

    if state['verbose']:
        st.write(response["messages"][-1].content)
    # store the agent's generated text (EDA report)
    return {"raw_data_report": response["messages"][-1].content}


def preprocess_data(state: State):
    '''Agent 2 - Data Preprocessor: Prepare the data for training based on the previous agent's report'''
    st.subheader("2. Preparing the data for training")
    data_preprocessor_prompt = prompts['data_preprocessor'].invoke(
        {
        "action": agent_actions['data_preprocessor'], "business_goal": state["business_goal"],
        "raw_data_report": state["raw_data_report"],
        "raw_data_path": state["raw_data_path"] 
        }
    ).text

    response = agents['data_preprocessor'].invoke(
        {"messages": [HumanMessage(content=data_preprocessor_prompt)]})

    if state['verbose']:
        st.write(response["messages"][-1].content)
    # store the agent's generated text (the cleaned data report)
    return {'clean_data_report': response["messages"][-1].content}


def select_model(state: State):
    '''Agent 3: Model Selector - choose the model for the given goal and dataset'''
    st.subheader("3. Choosing the best model")
    model_selector_prompt = prompts['model_selector'].invoke(
        {
        "action": agent_actions['model_selector'], "business_goal": state["business_goal"],
        "clean_data_report": state["clean_data_report"],
        }
    ).text

    response = agents['model_selector'].invoke(
        {"messages": [HumanMessage(content=model_selector_prompt)]})

    # store the agent's generated text (the model selection report)
    agent_output = response["messages"][-1].content
    if state['verbose']:
        st.write(agent_output)
    return {'model_name': re.findall(r"\{\{(.*?)\}\}", agent_output)[0], 'model_selection_report': agent_output}


def train_evaluate(state: State):
    '''Agent 4: Evaluator - Train and Evaluate the model'''
    st.subheader("4. Testing the model")
    evaluator_prompt = prompts['evaluator'].invoke(
        {
        "action": agent_actions['evaluator'],
        "business_goal": state["business_goal"],
        "clean_data_report": state["clean_data_report"],
        "model_name": state["model_name"],
        }
    ).text

    response = agents['evaluator'].invoke(
        {"messages": [HumanMessage(content=evaluator_prompt)]})

    if state['verbose']:
        st.write(response["messages"][-1].content)

    # Display generated images
    st.subheader("Generated Visualizations:")
    image_files = [f for f in os.listdir('.') if f.endswith('.png')]
    if image_files:
        for img_file in image_files:
            try:
                st.image(img_file)
            except Exception as img_e:
                st.error(f"Error displaying image {img_file}: {img_e}")
    else:
        st.write("No visualizations were generated or found.")

    # store the evaluation report
    return {'evaluation_report': response["messages"][-1].content}

def summarize(state: State):
    '''Summarize the process to non-technical stakeholders (agentless final step)'''
    st.subheader('5. Wrapping things up')

    summarizer_prompt = prompts['summarizer'].invoke({
        "business_goal": state["business_goal"],
        "raw_data_report": state["raw_data_report"],
        "clean_data_report": state["clean_data_report"],
        "model_selection_report": state["model_selection_report"],
        "evaluation_report": state["evaluation_report"]
        }
        ).text
    
    # Use invoke and access the content attribute
    final_summary = llm.invoke(summarizer_prompt).content 
    st.write(final_summary)

    # Find the saved model file
    model_file = None
    for ext in model_extensions:
        potential_model_files = [f for f in os.listdir('.') if f.endswith(f'.{ext}')]
        if potential_model_files:
            # Assuming the evaluator saved only one model file with the correct extension
            # Let's try to find one matching the selected model name if possible
            expected_model_name_part = state.get('model_name', 'model').lower() # Get model name from state if available
            found_match = False
            for potential_file in potential_model_files:
                if expected_model_name_part in potential_file.lower():
                    model_file = potential_file
                    found_match = True
                    break
            if not found_match:
                 model_file = potential_model_files[0] # Fallback to the first found file
            break # Found a model file with some extension

    # Provide download button for the model
    if model_file and os.path.exists(model_file):
        st.subheader("Download Trained Model")
        try:
            with open(model_file, "rb") as fp:
                st.download_button(
                    label=f"Download {model_file}",
                    data=fp,
                    file_name=model_file,
                    mime="application/octet-stream" # Generic binary file type
                )
        except Exception as dl_e:
            st.error(f"Error preparing model for download: {dl_e}")
    else:
        st.warning("Could not find the saved model file to download.")

    # this is what the end users will see
    return {'summary': final_summary}


# orchestrate the agent system using LangGraph 
graph_constructor = StateGraph(State).add_sequence([frame_problem, preprocess_data, select_model, train_evaluate, summarize])
# set the starting node
graph_constructor.add_edge(START, "frame_problem")
# compile the graph
graph = graph_constructor.compile()


# application running logic
def run_automl(business_goal, dataset_filepath, verbose, max_retries): # Renamed dataset_path to dataset_filepath

    for attempt in range(max_retries):
        try:

            # run the application with business goal, dataset path, and llm as input
            graph.invoke({
                'business_goal': business_goal,
                'raw_data_path': dataset_filepath, # Pass the filepath string
                'verbose': verbose,
                'llm': llm
            }, config={'recursion_limit': 10})

            # If all the tools were successfully utilized, break out of the loop
            files_to_check = ["X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv"]
            data_exists = all(os.path.exists(file) for file in files_to_check)

            model_exists = any(ext_exists('.', ext) for ext in model_extensions)

            if data_exists and model_exists:
                st.success("Pipeline completed successfully!")
                break
            else:
                st.warning(f"Attempt {attempt + 1} failed: The tools weren't properly utilized.") 

        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                st.error("Max retries reached. Please try again.") 
                raise  # Re-raise the exception when the program should stop

# Streamlit UI
st.title("AutoML Agent Collective")

business_goal = st.text_input("Enter your business goal:")
uploaded_file = st.file_uploader("Submit your dataset (.csv):") # Changed variable name
verbose = st.checkbox("Display the decision making process")
max_retries = st.number_input("Max Retries", min_value=1, max_value=10, value=3)

if st.button("Run AutoML"):
    if uploaded_file is not None:
        # Create a temporary directory to store the uploaded file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_filepath = os.path.join(temp_dir, uploaded_file.name)
            
            # Save the uploaded file to the temporary path
            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.info(f"Dataset saved temporarily to: {temp_filepath}") # Optional: show temp path
            
            # Run the AutoML process with the file path
            run_automl(business_goal, temp_filepath, verbose, max_retries)
            
            # Temporary directory and file are automatically cleaned up after exiting the 'with' block
            
    else:
        st.warning("Please upload a dataset first.")
