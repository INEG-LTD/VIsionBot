# job_app_automation/main.py

from playwright.sync_api import sync_playwright
from time import sleep
from find_job_listings import find_job_listings_with_playwright
from job_pref import JobPreferences
from employment_types import EmploymentTypes
from remote_flexibility import RemoteFlexibility

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
    
    jobs = find_job_listings_with_playwright(preferences)
    print(jobs)

if __name__ == "__main__":
    run()
