from pydantic import BaseModel

class Numbers(BaseModel):
    a: int
    b: int

def test_numbers_sum():
    n = Numbers(a=2, b=2)
    assert n.a + n.b == 3
