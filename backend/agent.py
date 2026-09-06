"""
SceneSync AI - Multi-Agent Orchestration Module
==================================================
This module implements the autonomous pre-production copilot pipeline:

    RAW SCRIPT (text / PDF)
        │
        ▼
    [1] DocumentProcessor        -> plain script text
        │
        ▼
    [2] ScriptBreakdownAgent     -> structured Scene[] (Gemini multi-step reasoning)
        │
        ▼
    [3] AssetExtractionAgent     -> Cast / Props / Locations / Wardrobe per scene
        │
        ▼
    [4] StoryboardAgent          -> Vertex AI Imagen 3 cinematic 16:9 concept art
        │
        ▼
    SceneSyncOrchestrator        -> ties it all together as one ADK Agent

Design principles:
  * Every network-bound call (Gemini, Imagen 3, Document AI, Vertex AI Search)
    is wrapped so that ANY failure (missing credentials, quota, network) falls
    back to a deterministic local simulation ("mock mode") rather than raising
    an unhandled exception. This is what makes the whole app zero-error, even
    when demoed with no Google Cloud project attached.
  * The real, production Vertex AI / Imagen 3 / ADK code paths are fully
    implemented (not stubbed) — they activate automatically the moment valid
    GOOGLE_CLOUD_PROJECT credentials are present.
  * A native Google Cloud Agent Development Kit `Agent` is constructed with
    FunctionTools mirroring each pipeline stage, so the same logic can be
    deployed to Vertex AI Agent Engine (`agent_engines`) unchanged.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import textwrap
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.config import get_settings

logger = logging.getLogger("scenesync.agent")
settings = get_settings()

# ---------------------------------------------------------------------------
# Optional heavy dependencies — imported lazily / defensively so the module
# always loads even if a given Google Cloud extra isn't installed yet.
# ---------------------------------------------------------------------------
try:
    import vertexai
    from vertexai.generative_models import (
        GenerationConfig,
        GenerativeModel,
        HarmBlockThreshold,
        HarmCategory,
        SafetySetting,
    )

    VERTEX_GENAI_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    VERTEX_GENAI_AVAILABLE = False
    logger.debug("vertexai.generative_models not available: %s", exc)

try:
    from vertexai.preview.vision_models import ImageGenerationModel

    IMAGEN_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    IMAGEN_AVAILABLE = False
    logger.debug("Imagen (vertexai.preview.vision_models) not available: %s", exc)

try:
    from google.cloud import documentai_v1 as documentai  # Document AI (Document Processing)

    DOCAI_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    DOCAI_AVAILABLE = False
    logger.debug("Document AI client not available: %s", exc)

try:
    from pypdf import PdfReader

    PYPDF_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    PYPDF_AVAILABLE = False
    logger.debug("pypdf not available: %s", exc)

try:
    from google.adk.agents import Agent as AdkAgent  # type: ignore
    from google.adk.tools import FunctionTool  # type: ignore

    ADK_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    ADK_AVAILABLE = False
    logger.debug("Google ADK (google-adk) not available: %s", exc)


_VERTEX_INITIALIZED = False


def _init_vertexai() -> bool:
    """Idempotently initialize the Vertex AI SDK. Returns True on success."""
    global _VERTEX_INITIALIZED
    if _VERTEX_INITIALIZED:
        return True
    if settings.effective_mock_mode or not VERTEX_GENAI_AVAILABLE:
        return False
    try:
        vertexai.init(project=settings.GOOGLE_CLOUD_PROJECT, location=settings.GOOGLE_CLOUD_LOCATION)
        _VERTEX_INITIALIZED = True
        logger.info(
            "Vertex AI initialized for project=%s location=%s",
            settings.GOOGLE_CLOUD_PROJECT,
            settings.GOOGLE_CLOUD_LOCATION,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vertex AI init failed, falling back to mock mode: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class SceneAssets:
    cast: List[str] = field(default_factory=list)
    props: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    wardrobe: List[str] = field(default_factory=list)
    sfx_notes: List[str] = field(default_factory=list)


@dataclass
class Scene:
    scene_number: int
    slugline: str
    setting: str
    time_of_day: str
    summary: str
    mood: str
    raw_text: str
    assets: SceneAssets = field(default_factory=SceneAssets)
    storyboard_prompt: str = ""
    storyboard_image_base64: Optional[str] = None
    storyboard_mime_type: str = "image/png"
    generation_source: str = "pending"  # "imagen3" | "mock"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_number": self.scene_number,
            "slugline": self.slugline,
            "setting": self.setting,
            "time_of_day": self.time_of_day,
            "summary": self.summary,
            "mood": self.mood,
            "assets": {
                "cast": self.assets.cast,
                "props": self.assets.props,
                "locations": self.assets.locations,
                "wardrobe": self.assets.wardrobe,
                "sfx_notes": self.assets.sfx_notes,
            },
            "storyboard_prompt": self.storyboard_prompt,
            "storyboard_image_base64": self.storyboard_image_base64,
            "storyboard_mime_type": self.storyboard_mime_type,
            "generation_source": self.generation_source,
        }


# ---------------------------------------------------------------------------
# [1] Document Processing
# ---------------------------------------------------------------------------
class DocumentProcessor:
    """
    Converts a raw upload (PDF or plain text) into clean screenplay text.

    Preferred path: Google Cloud Document AI (structured OCR / layout
    parsing) when a processor is configured. Falls back to local `pypdf`
    text extraction, and finally to naive UTF-8 decoding for .txt/.fountain
    uploads. Every branch is exception-safe.
    """

    def extract_text(self, file_bytes: bytes, content_type: str, filename: str) -> str:
        filename_lower = (filename or "").lower()
        is_pdf = "pdf" in (content_type or "") or filename_lower.endswith(".pdf")

        if not is_pdf:
            return self._decode_plain_text(file_bytes)

        if DOCAI_AVAILABLE and settings.GOOGLE_CLOUD_PROJECT and not settings.effective_mock_mode:
            try:
                return self._extract_with_document_ai(file_bytes)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Document AI extraction failed (%s); falling back to pypdf.", exc)

        if PYPDF_AVAILABLE:
            try:
                return self._extract_with_pypdf(file_bytes)
            except Exception as exc:  # noqa: BLE001
                logger.warning("pypdf extraction failed (%s); falling back to raw decode.", exc)

        return self._decode_plain_text(file_bytes)

    @staticmethod
    def _decode_plain_text(file_bytes: bytes) -> str:
        for encoding in ("utf-8", "latin-1"):
            try:
                return file_bytes.decode(encoding, errors="ignore")
            except Exception:  # noqa: BLE001
                continue
        return ""

    @staticmethod
    def _extract_with_pypdf(file_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001
                continue
        return "\n".join(pages_text).strip()

    @staticmethod
    def _extract_with_document_ai(file_bytes: bytes) -> str:
        """Uses a generic Document AI OCR processor for high-fidelity script extraction."""
        client = documentai.DocumentProcessorServiceClient()
        processor_path = client.processor_path(
            settings.GOOGLE_CLOUD_PROJECT, settings.GOOGLE_CLOUD_LOCATION, "scenesync-doc-processor"
        )
        raw_document = documentai.RawDocument(content=file_bytes, mime_type="application/pdf")
        request = documentai.ProcessRequest(name=processor_path, raw_document=raw_document)
        result = client.process_document(request=request)
        return result.document.text or ""


# ---------------------------------------------------------------------------
# Gemini reasoning wrapper (shared by breakdown + asset extraction agents)
# ---------------------------------------------------------------------------
class GeminiReasoner:
    """Thin, defensive wrapper around Vertex AI Gemini for structured JSON reasoning."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.GEMINI_REASONING_MODEL
        self._model = None

    def _get_model(self):
        if not _init_vertexai():
            return None
        if self._model is None:
            self._model = GenerativeModel(self.model_name)
        return self._model

    def _safety_settings(self):
        mapping = {
            "HARM_CATEGORY_HATE_SPEECH": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            "HARM_CATEGORY_DANGEROUS_CONTENT": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            "HARM_CATEGORY_HARASSMENT": HarmCategory.HARM_CATEGORY_HARASSMENT,
        }
        threshold_map = {
            "BLOCK_MEDIUM_AND_ABOVE": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            "BLOCK_ONLY_HIGH": HarmBlockThreshold.BLOCK_ONLY_HIGH,
            "BLOCK_LOW_AND_ABOVE": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            "BLOCK_NONE": HarmBlockThreshold.BLOCK_NONE,
        }
        settings_list = []
        for cat_name, threshold_name in settings.safety_settings.items():
            if cat_name in mapping:
                settings_list.append(
                    SafetySetting(category=mapping[cat_name], threshold=threshold_map.get(threshold_name, HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE))
                )
        return settings_list

    def generate_json(self, system_instruction: str, user_prompt: str, max_output_tokens: int = 4096) -> Optional[Any]:
        """Calls Gemini and parses a JSON response. Returns None on any failure."""
        model = self._get_model()
        if model is None:
            return None
        try:
            generation_config = GenerationConfig(
                temperature=0.4,
                top_p=0.95,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
            )
            response = model.generate_content(
                [f"SYSTEM INSTRUCTIONS:\n{system_instruction}\n\nUSER INPUT:\n{user_prompt}"],
                generation_config=generation_config,
                safety_settings=self._safety_settings(),
            )
            text = (response.text or "").strip()
            return _safe_json_parse(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini generation failed (%s); pipeline will use local fallback.", exc)
            return None


def _safe_json_parse(text: str) -> Optional[Any]:
    """Strips markdown code fences and parses JSON, tolerant of minor formatting noise."""
    if not text:
        return None
    cleaned = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
        return None


# ---------------------------------------------------------------------------
# [2] Script Breakdown Agent
# ---------------------------------------------------------------------------
SCENE_HEADING_RE = re.compile(
    r"^\s*(INT|EXT|INT\.?/EXT\.?|I/E)[\.\s]+(.+?)(?:\s*[-–—]\s*(DAY|NIGHT|DAWN|DUSK|CONTINUOUS|LATER|MORNING|EVENING|AFTERNOON))?\s*$",
    re.IGNORECASE,
)


class ScriptBreakdownAgent:
    """
    Splits raw screenplay text into structured Scene objects.

    Primary path: Gemini multi-step reasoning produces a scene-by-scene JSON
    breakdown with mood/summary interpretation. Deterministic fallback: a
    regex-based INT./EXT. slugline parser guarantees scenes are always
    produced, even fully offline.
    """

    SYSTEM_INSTRUCTION = textwrap.dedent(
        """
        You are an expert Hollywood script supervisor and AI production analyst.
        Break the provided screenplay into an ordered JSON array of scenes.
        For EVERY scene slugline (INT./EXT.) found in the text, output an object with:
          - scene_number (integer, sequential starting at 1)
          - slugline (the exact or cleaned scene heading)
          - setting (short location description)
          - time_of_day (DAY, NIGHT, DAWN, DUSK, CONTINUOUS, or UNKNOWN)
          - summary (2-3 sentence plain-English summary of the scene's action)
          - mood (one or two words describing cinematic tone, e.g. "tense", "romantic", "chaotic")
        Return ONLY a valid JSON array, no prose, no markdown fences.
        """
    ).strip()

    def __init__(self, reasoner: Optional[GeminiReasoner] = None):
        self.reasoner = reasoner or GeminiReasoner()

    def breakdown(self, script_text: str) -> List[Scene]:
        script_text = (script_text or "").strip()
        if not script_text:
            return []

        truncated = script_text[: 60_000]  # guard against runaway token usage
        result = self.reasoner.generate_json(self.SYSTEM_INSTRUCTION, truncated)

        scenes: List[Scene] = []
        if isinstance(result, list) and result:
            for i, item in enumerate(result[: settings.MAX_SCENES_PER_SCRIPT], start=1):
                if not isinstance(item, dict):
                    continue
                scenes.append(
                    Scene(
                        scene_number=int(item.get("scene_number") or i),
                        slugline=str(item.get("slugline") or f"SCENE {i}").strip(),
                        setting=str(item.get("setting") or "Unspecified location").strip(),
                        time_of_day=str(item.get("time_of_day") or "UNKNOWN").strip().upper(),
                        summary=str(item.get("summary") or "").strip(),
                        mood=str(item.get("mood") or "neutral").strip(),
                        raw_text="",
                    )
                )
            if scenes:
                logger.info("ScriptBreakdownAgent: Gemini produced %d scenes.", len(scenes))
                return scenes

        logger.info("ScriptBreakdownAgent: using deterministic regex fallback parser.")
        return self._fallback_breakdown(script_text)

    def _fallback_breakdown(self, script_text: str) -> List[Scene]:
        lines = script_text.splitlines()
        heading_indices: List[int] = []
        for idx, line in enumerate(lines):
            if SCENE_HEADING_RE.match(line.strip()):
                heading_indices.append(idx)

        scenes: List[Scene] = []
        if not heading_indices:
            # No formal sluglines found — synthesize scenes from paragraph breaks.
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", script_text) if p.strip()]
            for i, para in enumerate(paragraphs[: settings.MAX_SCENES_PER_SCRIPT], start=1):
                summary = " ".join(para.split())[:280]
                scenes.append(
                    Scene(
                        scene_number=i,
                        slugline=f"SCENE {i}",
                        setting="Unspecified location",
                        time_of_day="UNKNOWN",
                        summary=summary,
                        mood="neutral",
                        raw_text=para,
                    )
                )
            return scenes

        heading_indices.append(len(lines))
        for i in range(len(heading_indices) - 1):
            if i + 1 > settings.MAX_SCENES_PER_SCRIPT:
                break
            start, end = heading_indices[i], heading_indices[i + 1]
            heading_line = lines[start].strip()
            body_lines = lines[start + 1 : end]
            body_text = " ".join(l.strip() for l in body_lines if l.strip())
            match = SCENE_HEADING_RE.match(heading_line)
            setting = match.group(2).strip() if match and match.group(2) else heading_line
            time_of_day = (match.group(3) or "UNKNOWN").upper() if match else "UNKNOWN"
            scenes.append(
                Scene(
                    scene_number=i + 1,
                    slugline=heading_line,
                    setting=setting,
                    time_of_day=time_of_day,
                    summary=(body_text[:280] + ("…" if len(body_text) > 280 else "")) or "No action lines detected.",
                    mood="neutral",
                    raw_text=body_text,
                )
            )
        return scenes


# ---------------------------------------------------------------------------
# [3] Asset Extraction Agent
# ---------------------------------------------------------------------------
class AssetExtractionAgent:
    """
    Extracts structured production assets (cast, props, locations, wardrobe,
    SFX notes) for each scene using Gemini. Falls back to lightweight
    heuristic NLP (capitalized-word / keyword extraction) when Gemini is
    unavailable, so the breakdown table is never empty.
    """

    SYSTEM_INSTRUCTION = textwrap.dedent(
        """
        You are an AI production coordinator. Given a single screenplay scene,
        extract production-ready assets as a JSON object with these keys:
          - cast: array of character names appearing/speaking in the scene
          - props: array of notable physical objects/props referenced
          - locations: array of specific location descriptors
          - wardrobe: array of clothing/costume notes (infer tastefully if implied)
          - sfx_notes: array of sound design or visual effects cues
        Keep every array concise (max 8 items). Return ONLY valid JSON, no prose.
        """
    ).strip()

    WARDROBE_KEYWORDS = ("wearing", "dressed", "jacket", "suit", "gown", "uniform", "costume", "coat", "dress")
    PROP_KEYWORDS = ("gun", "phone", "knife", "car", "letter", "briefcase", "sword", "camera", "laptop", "glass", "bottle")
    SFX_KEYWORDS = ("explosion", "gunshot", "crash", "thunder", "music swells", "silence", "scream", "roar")

    def __init__(self, reasoner: Optional[GeminiReasoner] = None):
        self.reasoner = reasoner or GeminiReasoner()

    def extract(self, scene: Scene) -> SceneAssets:
        context = f"Slugline: {scene.slugline}\nSetting: {scene.setting}\nTime: {scene.time_of_day}\n" \
                  f"Summary: {scene.summary}\nRaw excerpt: {scene.raw_text[:2000]}"
        result = self.reasoner.generate_json(self.SYSTEM_INSTRUCTION, context, max_output_tokens=1024)

        if isinstance(result, dict):
            assets = SceneAssets(
                cast=_as_str_list(result.get("cast")),
                props=_as_str_list(result.get("props")),
                locations=_as_str_list(result.get("locations")) or [scene.setting],
                wardrobe=_as_str_list(result.get("wardrobe")),
                sfx_notes=_as_str_list(result.get("sfx_notes")),
            )
            if any([assets.cast, assets.props, assets.wardrobe, assets.sfx_notes]):
                return assets

        return self._fallback_extract(scene)

    def _fallback_extract(self, scene: Scene) -> SceneAssets:
        text = f"{scene.summary} {scene.raw_text}"
        # Heuristic cast detection: consecutive ALL-CAPS tokens of length >= 2,
        # a common screenplay convention for character cue names.
        cast_candidates = set(re.findall(r"\b([A-Z][A-Z]{1,}(?:\s[A-Z][A-Z]{1,})?)\b", text))
        blacklist = {"INT", "EXT", "CONTINUOUS", "DAY", "NIGHT", "later", "SCENE"}
        cast = sorted({c.title() for c in cast_candidates if c not in blacklist and len(c) < 25})[:8]

        lowered = text.lower()
        props = [kw.title() for kw in self.PROP_KEYWORDS if kw in lowered][:8]
        wardrobe = [kw.title() for kw in self.WARDROBE_KEYWORDS if kw in lowered][:8]
        sfx = [kw.title() for kw in self.SFX_KEYWORDS if kw in lowered][:8]

        return SceneAssets(
            cast=cast,
            props=props,
            locations=[scene.setting] if scene.setting else [],
            wardrobe=wardrobe,
            sfx_notes=sfx,
        )


def _as_str_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()][:8]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


# ---------------------------------------------------------------------------
# [4] Storyboard Agent (Vertex AI Imagen 3)
# ---------------------------------------------------------------------------
class StoryboardAgent:
    """
    Generates cinematic 16:9 concept art / storyboard panels per scene using
    Vertex AI Imagen 3. Falls back to a deterministic, elegant generated SVG
    placeholder (base64-encoded) so the storyboard grid always renders,
    even fully offline.
    """

    def __init__(self):
        self._model = None

    def _get_model(self):
        if not _init_vertexai() or not IMAGEN_AVAILABLE:
            return None
        if self._model is None:
            try:
                self._model = ImageGenerationModel.from_pretrained(settings.IMAGEN_MODEL)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load Imagen model '%s': %s", settings.IMAGEN_MODEL, exc)
                return None
        return self._model

    @staticmethod
    def build_prompt(scene: Scene) -> str:
        cast_str = ", ".join(scene.assets.cast[:4]) or "the scene's characters"
        props_str = ", ".join(scene.assets.props[:4])
        wardrobe_str = ", ".join(scene.assets.wardrobe[:3])
        prompt = (
            f"Cinematic film storyboard concept art, 16:9 widescreen composition, dramatic {scene.mood} lighting. "
            f"Scene: {scene.setting}, {scene.time_of_day.lower()}. {scene.summary} "
            f"Featuring {cast_str}. "
        )
        if props_str:
            prompt += f"Key props visible: {props_str}. "
        if wardrobe_str:
            prompt += f"Wardrobe notes: {wardrobe_str}. "
        prompt += (
            "Professional pre-visualization illustration style, moody color grading, "
            "detailed environment, film-grain texture, high production value, no on-image text, no watermark."
        )
        return prompt

    def generate(self, scene: Scene) -> Scene:
        prompt = self.build_prompt(scene)
        scene.storyboard_prompt = prompt

        model = self._get_model()
        if model is not None:
            try:
                images = model.generate_images(
                    prompt=prompt,
                    number_of_images=max(1, settings.STORYBOARD_IMAGES_PER_SCENE),
                    aspect_ratio=settings.STORYBOARD_ASPECT_RATIO,
                    safety_filter_level="block_medium_and_above",
                    person_generation="allow_adult",
                    add_watermark=False,
                )
                if images and len(images) > 0:
                    image_bytes = images[0]._image_bytes  # underlying PNG bytes
                    scene.storyboard_image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                    scene.storyboard_mime_type = "image/png"
                    scene.generation_source = "imagen3"
                    return scene
            except Exception as exc:  # noqa: BLE001
                logger.warning("Imagen 3 generation failed for scene %s (%s); using placeholder art.", scene.scene_number, exc)

        scene.storyboard_image_base64 = _generate_placeholder_svg(scene)
        scene.storyboard_mime_type = "image/svg+xml"
        scene.generation_source = "mock"
        return scene


_PALETTES = [
    ("#0f172a", "#7c3aed", "#22d3ee"),
    ("#1a1025", "#db2777", "#f59e0b"),
    ("#0b1120", "#059669", "#38bdf8"),
    ("#171018", "#dc2626", "#fbbf24"),
    ("#0c1220", "#2563eb", "#a78bfa"),
]


def _generate_placeholder_svg(scene: Scene) -> str:
    """Builds a tasteful, dependency-free cinematic placeholder panel as base64 SVG."""
    palette = _PALETTES[scene.scene_number % len(_PALETTES)]
    bg, accent1, accent2 = palette
    width, height = 960, 540

    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    title = esc(scene.slugline[:60])
    subtitle = esc(f"{scene.setting} — {scene.time_of_day}"[:70])
    mood = esc(scene.mood.upper())

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{bg}"/>
      <stop offset="100%" stop-color="#000000"/>
    </linearGradient>
    <radialGradient id="glow1" cx="20%" cy="20%" r="60%">
      <stop offset="0%" stop-color="{accent1}" stop-opacity="0.45"/>
      <stop offset="100%" stop-color="{accent1}" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="glow2" cx="85%" cy="80%" r="55%">
      <stop offset="0%" stop-color="{accent2}" stop-opacity="0.35"/>
      <stop offset="100%" stop-color="{accent2}" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="{width}" height="{height}" fill="url(#bgGrad)"/>
  <rect width="{width}" height="{height}" fill="url(#glow1)"/>
  <rect width="{width}" height="{height}" fill="url(#glow2)"/>
  <g opacity="0.15" stroke="#ffffff" stroke-width="1">
    <line x1="0" y1="180" x2="{width}" y2="180"/>
    <line x1="0" y1="360" x2="{width}" y2="360"/>
    <line x1="320" y1="0" x2="320" y2="{height}"/>
    <line x1="640" y1="0" x2="640" y2="{height}"/>
  </g>
  <rect x="24" y="24" width="220" height="40" rx="8" fill="#000000" fill-opacity="0.45"/>
  <text x="40" y="50" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="{accent2}" font-weight="700">SCENE {scene.scene_number:02d} · {mood}</text>
  <g transform="translate({width/2}, {height/2 - 20})" text-anchor="middle">
    <circle r="46" fill="none" stroke="{accent1}" stroke-width="2" opacity="0.7"/>
    <circle r="30" fill="none" stroke="{accent2}" stroke-width="2" opacity="0.5"/>
    <path d="M -14 -18 L 14 0 L -14 18 Z" fill="{accent2}" opacity="0.9"/>
  </g>
  <text x="{width/2}" y="{height/2 + 70}" font-family="Georgia, serif" font-size="28" fill="#f8fafc" text-anchor="middle" font-weight="700">{title}</text>
  <text x="{width/2}" y="{height/2 + 104}" font-family="Helvetica, Arial, sans-serif" font-size="17" fill="#94a3b8" text-anchor="middle">{subtitle}</text>
  <text x="{width - 24}" y="{height - 20}" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#64748b" text-anchor="end">SceneSync AI · concept preview</text>
</svg>"""
    return base64.b64encode(svg.encode("utf-8")).decode("utf-8")


# ---------------------------------------------------------------------------
# Orchestrator — wires all four stages together
# ---------------------------------------------------------------------------
class SceneSyncOrchestrator:
    """
    Top-level multi-agent orchestrator. Exposes both:
      * `run_pipeline(...)` — direct, synchronous/deterministic execution used
        by the FastAPI runtime (fast, reliable, always succeeds).
      * `build_adk_agent()` — constructs a native Google Cloud ADK `Agent`
        wrapping the same tool functions, ready for `agent_engines.create()`
        deployment to Vertex AI Agent Engine.
    """

    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.breakdown_agent = ScriptBreakdownAgent()
        self.asset_agent = AssetExtractionAgent()
        self.storyboard_agent = StoryboardAgent()

    # -- Stage-level tool functions (also used as ADK FunctionTools) --------
    def tool_extract_script_text(self, file_bytes: bytes, content_type: str, filename: str) -> str:
        """ADK tool: extract plain text from an uploaded script (PDF or text)."""
        return self.doc_processor.extract_text(file_bytes, content_type, filename)

    def tool_breakdown_scenes(self, script_text: str) -> List[Dict[str, Any]]:
        """ADK tool: break a script into structured scenes."""
        scenes = self.breakdown_agent.breakdown(script_text)
        return [s.to_dict() for s in scenes]

    def tool_extract_assets(self, scene_dict: Dict[str, Any]) -> Dict[str, Any]:
        """ADK tool: extract cast/props/locations/wardrobe for one scene."""
        scene = _scene_from_dict(scene_dict)
        assets = self.asset_agent.extract(scene)
        return {
            "cast": assets.cast,
            "props": assets.props,
            "locations": assets.locations,
            "wardrobe": assets.wardrobe,
            "sfx_notes": assets.sfx_notes,
        }

    def tool_generate_storyboard(self, scene_dict: Dict[str, Any]) -> Dict[str, Any]:
        """ADK tool: generate a 16:9 cinematic storyboard panel for one scene via Imagen 3."""
        scene = _scene_from_dict(scene_dict)
        scene = self.storyboard_agent.generate(scene)
        return scene.to_dict()

    # -- Full pipeline --------------------------------------------------
    def run_pipeline(
        self,
        script_text: str,
        generate_storyboards: bool = True,
        progress_callback: Optional[Any] = None,
    ) -> List[Scene]:
        """
        Runs the full four-stage multi-agent pipeline synchronously and
        returns fully populated Scene objects (breakdown + assets +
        storyboard art). `progress_callback(stage: str, pct: int)` is
        invoked between stages for live UI progress updates.
        """

        def _report(stage: str, pct: int):
            if progress_callback:
                try:
                    progress_callback(stage, pct)
                except Exception:  # noqa: BLE001
                    pass

        _report("Breaking down script into scenes", 10)
        scenes = self.breakdown_agent.breakdown(script_text)
        if not scenes:
            return []

        total = len(scenes)
        for i, scene in enumerate(scenes):
            pct = 15 + int((i / max(total, 1)) * 45)
            _report(f"Extracting production assets for {scene.slugline}", pct)
            scene.assets = self.asset_agent.extract(scene)

        if generate_storyboards:
            for i, scene in enumerate(scenes):
                pct = 60 + int((i / max(total, 1)) * 38)
                _report(f"Rendering storyboard art for {scene.slugline}", pct)
                self.storyboard_agent.generate(scene)

        _report("Finalizing production breakdown", 100)
        return scenes

    # -- ADK Agent construction (Vertex AI Agent Engine deployable) --------
    def build_adk_agent(self):
        """
        Builds a native Google Cloud Agent Development Kit `Agent` that wraps
        this orchestrator's tool functions, suitable for local ADK evaluation
        or deployment via `vertexai.agent_engines.create(agent)`.

        Returns None gracefully if the `google-adk` package is not installed
        in the current environment (keeps `import backend.agent` error-free
        everywhere, including plain Replit/FastAPI-only deployments).
        """
        if not ADK_AVAILABLE:
            logger.info("google-adk not installed; skipping native ADK agent construction.")
            return None

        try:
            tools = [
                FunctionTool(self.tool_breakdown_scenes),
                FunctionTool(self.tool_extract_assets),
                FunctionTool(self.tool_generate_storyboard),
            ]
            agent = AdkAgent(
                name=settings.ADK_APP_NAME,
                model=settings.GEMINI_REASONING_MODEL,
                description="Autonomous pre-production copilot that breaks down film scripts into "
                             "scenes, extracts production assets, and generates cinematic storyboard art.",
                instruction=textwrap.dedent(
                    """
                    You are SceneSync AI, an autonomous film pre-production copilot.
                    Given a screenplay, call `tool_breakdown_scenes` to segment it into
                    scenes, then for each scene call `tool_extract_assets` to identify
                    cast/props/locations/wardrobe, and finally call
                    `tool_generate_storyboard` to render cinematic concept art. Always
                    reason step by step and report structured results.
                    """
                ).strip(),
                tools=tools,
            )
            logger.info("Native Google Cloud ADK Agent '%s' constructed successfully.", settings.ADK_APP_NAME)
            return agent
        except Exception as exc:  # noqa: BLE001
            logger.warning("ADK agent construction failed (%s); FastAPI pipeline will run standalone.", exc)
            return None


def _scene_from_dict(d: Dict[str, Any]) -> Scene:
    assets_dict = d.get("assets") or {}
    return Scene(
        scene_number=int(d.get("scene_number", 0)),
        slugline=str(d.get("slugline", "")),
        setting=str(d.get("setting", "")),
        time_of_day=str(d.get("time_of_day", "UNKNOWN")),
        summary=str(d.get("summary", "")),
        mood=str(d.get("mood", "neutral")),
        raw_text=str(d.get("raw_text", "")),
        assets=SceneAssets(
            cast=_as_str_list(assets_dict.get("cast")),
            props=_as_str_list(assets_dict.get("props")),
            locations=_as_str_list(assets_dict.get("locations")),
            wardrobe=_as_str_list(assets_dict.get("wardrobe")),
            sfx_notes=_as_str_list(assets_dict.get("sfx_notes")),
        ),
        storyboard_prompt=str(d.get("storyboard_prompt", "")),
        storyboard_image_base64=d.get("storyboard_image_base64"),
        storyboard_mime_type=str(d.get("storyboard_mime_type", "image/png")),
        generation_source=str(d.get("generation_source", "pending")),
    )


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]
