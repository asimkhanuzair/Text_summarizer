
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import Body
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# bert_model = Summarizer()
model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
tokenizer = AutoTokenizer.from_pretrained('t5-base')

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/summarize/")
async def read_item(image: str = Body(...)):
    tokens_input = tokenizer.encode("summarize: "+image, return_tensors='pt', 
                                    max_length=1024, 
                                    truncation=True)


    summary_ids = model.generate(tokens_input, min_length=80,
                                max_length=200,
                                length_penalty=20,
                                top_k=300,top_p=0.94,
                                num_beams=6
                                )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {summary}

