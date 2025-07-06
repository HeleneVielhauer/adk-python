from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from google.adk.agents import Agent
import os

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

persona_agent = Agent(
    name="persona_generator",
    model="gemini-2.0-flash",
    instruction=(
        "Du bist ein Experte für Zielgruppenanalyse. "
        "Erstelle einen ausführlichen Steckbrief einer Userpersona auf Basis der folgenden Informationen: "
        "Branche und Produktbeschreibung. "
        "Der Steckbrief soll Name, Alter, Beruf, Interessen, Herausforderungen, Ziele und typische Verhaltensweisen enthalten."
    ),
    description="Generiert Userpersonas für Marketing und Produktentwicklung."
)

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("persona_form.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
def form_post(request: Request, branche: str = Form(...), produktbeschreibung: str = Form(...)):
    prompt = f"Branche: {branche}\nProduktbeschreibung: {produktbeschreibung}"
    result = persona_agent(prompt)
    return templates.TemplateResponse("persona_form.html", {"request": request, "result": result, "branche": branche, "produktbeschreibung": produktbeschreibung})
