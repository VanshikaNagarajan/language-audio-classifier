from typing import List
from pydantic import BaseModel




class MFCCInput(BaseModel):
    MFCCs: List[List[float]]