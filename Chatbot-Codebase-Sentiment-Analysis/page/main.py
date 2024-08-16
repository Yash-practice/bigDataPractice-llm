from constants import domain_constant
from page.general import general_page
from page.social_media import social_media_page

def page(domain_name):
    if domain_name==domain_constant.GENERAL:
        general_page.general_search(domain_name)
    elif domain_name==domain_constant.SOCIAL_MEDIA:
        social_media_page.social_media_analysis(domain_name)