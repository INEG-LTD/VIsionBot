from openai import OpenAI
from dataclasses import dataclass, field
from pydantic import BaseModel
import json

client = OpenAI()

class FormInput(BaseModel):
    input_description: str
    input_id: str
    input_class: str
    input_type: str
    input_value: str
    input_placeholder: str
    input_required: bool
    input_disabled: bool
    input_readonly: bool
    input_multiple: bool

    def to_dict(self):
        return {
            "input_description": self.input_description,
            "input_id": self.input_id,
            "input_class": self.input_class,
            "input_type": self.input_type,
            "input_value": self.input_value,
            "input_placeholder": self.input_placeholder,
            "input_required": self.input_required,
            "input_disabled": self.input_disabled,
            "input_readonly": self.input_readonly,
            "input_multiple": self.input_multiple
        }
    
    def to_json(self):
        return json.dumps(self.to_dict())

class Response(BaseModel):
    form_inputs: list[FormInput] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "form_inputs": [input.to_dict() for input in self.form_inputs]
        }
    
    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)
    

def get_html_form_inputs(html_content):
    response = client.responses.parse(
        model="gpt-4.1",
        input=[
            {
                "role": "system",
                "content": """
                    You will be given the HTML source code for a job board home page. 
                    Your task is to identify the input elements that collect the user's job preferences and return their HTML 'id' and 'class' attributes.

                    Always return a list, even if there is only one input. 
                    Preserve the order in which the inputs appear in the source HTML. 
                    If you encounter grouped inputs (like radio button sets for a single preference), include each input as a separate object.
                """
            },
            {
                "role": "user",
                "content": f"Extract all form inputs from the following HTML: {html_content}"
            }
        ],
        text_format=Response
    )

    response = response.output_parsed
    return response.form_inputs

if __name__ == "__main__":
    with open("html_content.html", "r") as file:
        html_content = file.read()
    get_html_form_inputs(html_content)