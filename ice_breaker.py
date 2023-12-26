import os

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.chains import LLMChain

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile

if __name__ == '__main__':
    print('Hello LangChain!!')
    
    linkedin_profile_url = linkedin_lookup_agent(name='Bernardo Llamas Verna')
    
    summary_template = """
        given the Linkedin information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """
    
    summary_prompt_template = PromptTemplate(input_variables=['information'], 
                                             template=summary_template)
    
    llm = AzureChatOpenAI(temperature=0, 
                     model_name='gpt-35-turbo-16k', 
                     deployment_name='gpt-35-turbo-16k'
                     )
    
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    
    linkedin_data = scrape_linkedin_profile(
        'https://www.linkedin.com/in/bernardo-llamas-verna-55bb5343/'
    )
    
    print(chain.run(information=linkedin_data))
