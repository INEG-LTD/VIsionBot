# job_app_automation/main.py

from playwright.sync_api import sync_playwright
from time import sleep
from find_job_listings import visit_site
from job_pref import JobPreferences
from employment_types import EmploymentTypes
from remote_flexibility import RemoteFlexibility
from input_handling import get_html_form_inputs, get_html_button_inputs

def run():
    preferences = JobPreferences(
        job_titles=['Software Engineer', 'Backend Developer'],
        locations=['Remote', 'New York', 'San Francisco'],
        salary_min=120000,
        employment_types=[EmploymentTypes.FULL_TIME],
        industries=['Technology', 'Finance'],
        company_sizes=['Startup'],
        required_skills=['Python', 'Django', 'REST APIs'],
        experience_levels=['Mid', 'Senior'],
        visa_sponsorship_required=True,
        remote_flexibility=[RemoteFlexibility.REMOTE],
        desired_benefits=['Health Insurance', 'Stock Options'],
        exclude_keywords=['unpaid', 'internship']
    )
    
    job_sites = [
        'https://www.glassdoor.com',
        'https://www.ziprecruiter.com',
        'https://www.monster.co.uk',
    ]
    
    for job_site in job_sites:
        # 1. visit the job site
        page_source, page = visit_site(job_site)

        # 2. get the input fields for the job site
        form_inputs = get_html_form_inputs(page_source)
        button_inputs = get_html_button_inputs(page_source)

        # 3. fill in the input fields
        for form_input in form_inputs:
            page.fill(form_input.input_id, form_input.input_value)
        print(form_inputs)
        print(button_inputs)

if __name__ == "__main__":
    run()
