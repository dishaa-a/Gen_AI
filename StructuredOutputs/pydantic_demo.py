from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str
    age: Optional[int] = None
    email: Optional[EmailStr] = None
    cgpa: float = Field(ge=0.0, le=4.0, description="CGPA must be between 0.0 and 4.0")

new_student = {'age': 20, 'name': 'John Doe', 'email': 'john.doe@example.com', 'cgpa': 3.5}

student = Student(**new_student)
print(student)