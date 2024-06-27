import streamlit as st
import boto3
import botocore
import json

#################
# Initial Setup #
#################

# Initialize boto3 clients
session = boto3.Session()
config = botocore.config.Config(connect_timeout=30000, read_timeout=30000)
bedrock = session.client('bedrock')
bedrock_runtime = session.client('bedrock-runtime','us-east-1',config = config)
translate = session.client('translate')


#################
# Methods #
#################

#parse xml helper method
def parse_xml(xml, tag):
    temp=xml.split(">")
    
    tag_to_extract="</"+tag

    for line in temp:
        if tag_to_extract in line:
            parsed_value=line.replace(tag_to_extract, "")
            return parsed_value

# Returns a list of languages supported by Amazon and caches the data
@st.cache_data
def lst_langs():
    return translate.list_languages()['Languages']

# Lists all Bedrock models
@st.cache_data
def lst_models():
    return bedrock.list_foundation_models(
        byProvider='Anthropic',
        byOutputModality='TEXT',
        byInferenceType='ON_DEMAND'
    )['modelSummaries']

# Translates input text from the source language to the target language using Bedrock
def transl_txt_bedrock(input_txt, src_lang, tgt_lang, model_id):
    
    ##Setup Prompt
    prompt = f"""    
        Human:
        Task:  An expert in IT and AWS, translate the given text from the source language to the target language.

        Source language: {src_lang}
        Target language: {tgt_lang}
        
        Text to translate: {input_txt}
        
        Translation steps:
        Translate the following blog content from English to Vietnamese, ensuring the context is preserved and all IT and AWS technical terms are kept intact. Maintain the accuracy of technical details and use appropriate Vietnamese terminology where necessary. Follow the steps for accurate translation and contextual adaptation:
        
        1. Read and understand the content thoroughly to grasp the context and purpose.
        2. Identify and note down the specific IT and AWS technical terms to be retained.
        3. Prepare supporting materials, such as technical dictionaries or official AWS documentation, for reference.
        4. Perform a rough translation, maintaining the technical terms and focusing on conveying the correct meaning.
        5. Adjust grammar and sentence structure to ensure fluency and readability.
        6. Compare the translation with the original to ensure completeness and accuracy.
        7. Review and edit the translation for coherence, spelling, and grammatical errors.
        8. If possible, have an expert in IT and AWS review the translation for accuracy and context.
        9. Finalize the translation, ensuring proper formatting and layout for publication.
        Here’s an example paragraph for translation:
        
        Original English Text:
        Amazon S3 provides scalable object storage with high availability and low latency. It is ideal for use cases such as backup and restore, big data analytics, and content distribution.
        
        Translated Vietnamese Text:
        Amazon S3 cung cấp lưu trữ đối tượng có khả năng mở rộng với tính sẵn sàng cao và độ trễ thấp. Nó lý tưởng cho các trường hợp sử dụng như sao lưu và khôi phục, phân tích dữ liệu lớn và phân phối nội dung.

        Only output the response in the target language
        
        <translated_text>
        </translated_text>

        Assistant:
    """

    body=json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 10000,
            "temperature":0,
            "top_k":250,
            "top_p":0.5,
            "stop_sequences":[],
            "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        }  
    )  

    #Run Inference
    modelId = model_id  # change this to use a different version from the model provider if you want to switch 
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    llmOutput = response_body.get("content")[0]["text"]

    return llmOutput

#Translates chat messages to the target language using Bedrock"""
def transl_chat_bedrock(input_txt, tgt_lang, model_id):
    

    ##Setup Prompt
    prompt = f"""    
        Human:
        Task: An expert in IT and AWS, Respond to the given text in the target language.

        Target language: {tgt_lang}
        
        Text to respond to: {input_txt}
        
        Steps:
        Translate the following blog content from English to Vietnamese, ensuring the context is preserved and all IT and AWS technical terms are kept intact. Maintain the accuracy of technical details and use appropriate Vietnamese terminology where necessary. Follow the steps for accurate translation and contextual adaptation:
        
        1. Read and understand the content thoroughly to grasp the context and purpose.
        2. Identify and note down the specific IT and AWS technical terms to be retained.
        3. Prepare supporting materials, such as technical dictionaries or official AWS documentation, for reference.
        4. Perform a rough translation, maintaining the technical terms and focusing on conveying the correct meaning.
        5. Adjust grammar and sentence structure to ensure fluency and readability.
        6. Compare the translation with the original to ensure completeness and accuracy.
        7. Review and edit the translation for coherence, spelling, and grammatical errors.
        8. If possible, have an expert in IT and AWS review the translation for accuracy and context.
        9. Finalize the translation, ensuring proper formatting and layout for publication.
        Here’s an example paragraph for translation:
        
        Original English Text:
        Amazon S3 provides scalable object storage with high availability and low latency. It is ideal for use cases such as backup and restore, big data analytics, and content distribution.
        
        Translated Vietnamese Text:
        Amazon S3 cung cấp lưu trữ đối tượng có khả năng mở rộng với tính sẵn sàng cao và độ trễ thấp. Nó lý tưởng cho các trường hợp sử dụng như sao lưu và khôi phục, phân tích dữ liệu lớn và phân phối nội dung.
        
        Only output the response in the target language
        
        <response>
        </response>

        Assistant:
    """

    body=json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 10000,
            "temperature":0,
            "top_k":250,
            "top_p":0.5,
            "stop_sequences":[],
            "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        }  
    )  

    #Run Inference
    modelId = model_id  # change this to use a different version from the model provider if you want to switch 
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    llmOutput = response_body.get("content")[0]["text"]
    print("bedrock" + llmOutput)
    return llmOutput

# Analyzes the quality of the translation
def analyze_responses(input_txt, bedrock_txt, model_id):

    ##Setup Prompt
    prompt = f"""    
        Human:
        Role: You are An expert in IT and AWS, a professional translator tasked with reviewing and analyzing translated text.
        
        Original text: {input_txt}
        Bedrock Translated text: {bedrock_txt}
        
        Review process:
        1. Read the original text to understand the context and meaning,  including specialized IT and AWS terminology.
        2. Review the bedrock translated text for accuracy, fluency, and adherence to the original context,
        3. Evaluate the quality of the translation

        Ensure your analysis and response is in English

        <analysis>
        </analysis>
        
        Assistant:
    """

    body=json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 10000,
            "temperature":0,
            "top_k":250,
            "top_p":0.5,
            "stop_sequences":[],
            "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        }  
    )  

    #Run Inference
    modelId = model_id  # change this to use a different version from the model provider if you want to switch 
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    llmOutput = response_body.get("content")[0]["text"]
    
    return llmOutput