from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
app.state.num = 0

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/")
def read_root():
    data = {
        "How we do it in Puerto Rico": "Hey man, this is how we do it down in Puerto Rico.",
        "Cos im out here screaming down and ditto": "DESPACITOOOO",
    }
    return data


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_price": item.price, "is_offer": item.is_offer, "item_id": item_id}

