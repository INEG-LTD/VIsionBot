from dataclasses import dataclass, field
from typing import List, Optional
from employment_types import EmploymentTypes
from remote_flexibility import RemoteFlexibility

@dataclass
class JobPreferences:
    job_titles: List[str]
    locations: List[str]
    salary_min: Optional[int] = None  # Annual USD or your currency
    employment_types: List[EmploymentTypes] = field(default_factory=list)  # e.g., ['Full-time', 'Contract']
    industries: List[str] = field(default_factory=list)
    company_sizes: List[str] = field(default_factory=list)  # e.g., ['Startup', 'Enterprise']
    required_skills: List[str] = field(default_factory=list)
    experience_levels: List[str] = field(default_factory=list)  # e.g., ['Entry', 'Mid', 'Senior']
    visa_sponsorship_required: Optional[bool] = None
    remote_flexibility: List[RemoteFlexibility] = field(default_factory=list)  # e.g., ['Remote', 'Hybrid', 'On-site']
    desired_benefits: List[str] = field(default_factory=list)
    exclude_keywords: List[str] = field(default_factory=list)

# Example usage
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