import os

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.chains import LLMChain

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile
from output_parsers import person_intel_parser, PersonIntel

def ice_break(name:str)-> tuple[PersonIntel, str]:
    linkedin_profile_url = linkedin_lookup_agent(name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url)
    
    summary_template = """
        given the Linkedin information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
        3. a topic that may interest them
        4. 2 creative Ice breakers to open a conversation with them {format_instructions}
    """
    
    summary_prompt_template = PromptTemplate(
        input_variables=['information'], 
        template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        }
    )
    
    llm = AzureChatOpenAI(
        temperature=0, 
        model_name='gpt-35-turbo-16k', 
        deployment_name='gpt-35-turbo-16k'
    )
    
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    result = chain.run(information=linkedin_data)
    print(result)
    
    return person_intel_parser.parse(result), linkedin_data.get("profile_pic_url")

if __name__ == '__main__':
    print('Hello LangChain!!')
    
    ice_break(name="Bernardo Llamas Verna")
