from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field, EmailStr

from StructuredOutputs.with_structured_output import ReviewSummary

from StructuredOutputs.with_structured_output import ReviewSummary

load_dotenv()  # Load environment variables from .env file

model = OpenAI(model = 'gpt-3.5-turbo')



#schema for the structured output
{
    "title": "ReviewSummary",
    "type": "object",
    "properties": {
        "key_themes": {
        "type": "array",
        "items": {
            "type": "string"
        },
        "description": "A list of key themes or topics mentioned in the review"
        },
        "summary": {
        "type": "string",
        "description": "A brief summary of the review"
        },
        "sentiment": {
        "type": "string",
        "enum": ["positive", "negative"],
        "description": "The sentiment of the review (positive, negative, or neutral)"
        },
        "pros": {
        "type": ["array", "null"],
        "items": {
            "type": "string"
        },
        "description": "A list of positive aspects mentioned in the review, if any"
        },
        "cons": {
        "type": ["array", "null"],
        "items": {
            "type": "string"
        },
        "description": "A list of negative aspects mentioned in the review, if any"
        },
        "name": {
        "type": ["string", "null"],
        "description": "The name of the reviewer, if available"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}


structured_model = model.with_structured_output(ReviewSummary) 

result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I never use, and they take up valuable storage space. I wish there was a way to remove them or at least disable them. The user interface is also not very intuitive, and it takes some time to figure out how to navigate through the settings. Overall, I'm satisfied with the performance, but the software experience could be improved.""") 

print(result)