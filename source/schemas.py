from pydantic import BaseModel
from typing import Literal


class FAQAIResponse(BaseModel):
    intent: Literal[
        'Placement',
        'International',
        'Transport',
        'Examination',
        'Library',
        'Migration',
        'Contact',
        'Entry_Test',
        'Departments',
        'Hostel',
        'Merit_List',
        'Fee_Structure',
        'Scholarship',
        'Eligibility',
        'Admission_Dates'
    ]
    answer: str
