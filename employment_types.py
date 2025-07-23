from enum import Enum

class EmploymentTypes(Enum):
    FULL_TIME = "Full-time"
    PART_TIME = "Part-time"
    CONTRACT = "Contract"
    TEMPORARY = "Temporary"
    INTERNSHIP = "Internship"
    VOLUNTEER = "Volunteer"
    FREELANCE = "Freelance"
    OTHER = "Other"
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value
    
    def __eq__(self, other):
        return self.value == other.value