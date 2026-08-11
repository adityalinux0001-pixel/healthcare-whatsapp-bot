
import logging
from pathlib import Path

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.web.config import site

logger = logging.getLogger("website")

router = APIRouter(tags=["Website"])

TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals["site"] = site


def _render(request: Request, template: str, **ctx):
    return templates.TemplateResponse(request, template, {**ctx})


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return _render(request, "index.html", page_title=f"{site.PRODUCT_NAME} | AI Health Support on WhatsApp")


@router.get("/features", response_class=HTMLResponse)
async def features(request: Request):
    return _render(request, "features.html", page_title=f"Features | {site.PRODUCT_NAME}")


@router.get("/how-it-works", response_class=HTMLResponse)
async def how_it_works(request: Request):
    return _render(request, "how_it_works.html", page_title=f"How It Works | {site.PRODUCT_NAME}")


@router.get("/pricing", response_class=HTMLResponse)
async def pricing(request: Request):
    
    return _render(request, "pricing.html", page_title=f"Pricing | {site.PRODUCT_NAME}")


@router.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return _render(request, "about.html", page_title=f"About | {site.PRODUCT_NAME}")


@router.get("/contact", response_class=HTMLResponse)
async def contact(request: Request):
    return _render(request, "contact.html", page_title=f"Contact | {site.PRODUCT_NAME}")


@router.post("/contact", response_class=HTMLResponse)
async def contact_submit(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    message: str = Form(...),
):

    logger.info(f"[contact form] {name} <{email}>: {message[:200]}")
    return _render(
        request,
        "contact.html",
        page_title=f"Contact | {site.PRODUCT_NAME}",
        submitted=True,
    )


@router.get("/faq", response_class=HTMLResponse)
async def faq(request: Request):
    return _render(request, "faq.html", page_title=f"FAQ | {site.PRODUCT_NAME}")


@router.get("/privacy-policy", response_class=HTMLResponse)
async def privacy_policy(request: Request):
    return _render(request, "privacy_policy.html", page_title=f"Privacy Policy | {site.PRODUCT_NAME}")


@router.get("/terms-and-conditions", response_class=HTMLResponse)
async def terms_and_conditions(request: Request):
    return _render(request, "terms_and_conditions.html", page_title=f"Terms & Conditions | {site.PRODUCT_NAME}")


@router.get("/refund-policy", response_class=HTMLResponse)
async def refund_policy(request: Request):
    return _render(request, "refund_policy.html", page_title=f"Cancellation & Refund Policy | {site.PRODUCT_NAME}")


@router.get("/shipping-policy", response_class=HTMLResponse)
async def shipping_policy(request: Request):
    return _render(request, "shipping_policy.html", page_title=f"Shipping Policy | {site.PRODUCT_NAME}")
