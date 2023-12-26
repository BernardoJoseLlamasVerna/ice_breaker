import os
import requests

def scrape_linkedin_profile(linkedin_profile_url: str):
    """scrape info from Linkedin profiles,
    Manually scrape the info from the Linkedin profile"""
    
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    headers_dic = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}
    
    response = requests.get(
        api_endpoint,
        params={"url": linkedin_profile_url}, 
        headers=headers_dic
    )
    
    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None) and k not in ["people_also_viewed", "certifications"]
    }
    
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")
    
    return data