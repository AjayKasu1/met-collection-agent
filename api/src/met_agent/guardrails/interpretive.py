# ruff: noqa: RUF001
"""Provide a factual, non-interpretive policy response in the user's supported language."""

from met_agent.agent.models import Language

POLICY: dict[Language, str] = {
    "en": (
        "I can report the collection record, but I do not interpret artistic "
        "meaning or judge artistic quality. I can share the Met's own "
        "curatorial text when available, or help with the artist, materials, "
        "date, and other documented facts."
    ),
    "fr": (
        "Je peux présenter les informations du catalogue, mais je n'interprète "
        "pas le sens des œuvres et ne juge pas leur qualité artistique. Je peux"
        " citer les textes du Met lorsqu'ils sont disponibles, ou vous aider "
        "sur l'artiste, les matériaux, la date et les faits documentés."
    ),
    "es": (
        "Puedo informar sobre el registro de la colección, pero no interpreto "
        "el significado ni juzgo la calidad artística. Puedo compartir textos "
        "del Met cuando estén disponibles, o ayudar con el artista, los "
        "materiales, la fecha y otros datos documentados."
    ),
    "zh": (
        "我可以介绍馆藏记录，但不解读艺术含义，也不评价艺术品质。"
        "如有提供，我可以引用大都会艺术博物馆的策展文字，"
        "或帮助查找艺术家、材料、年代等有据可查的信息。"
    ),
}
UNVERIFIED: dict[Language, str] = {
    "en": (
        "I couldn't verify that from the available museum sources. Please try a"
        " more specific question or contact info@metmuseum.org."
    ),
    "fr": (
        "Je n'ai pas pu vérifier cela dans les sources disponibles du musée. "
        "Précisez votre question ou contactez info@metmuseum.org."
    ),
    "es": (
        "No pude verificarlo con las fuentes disponibles del museo. Intente una"
        " pregunta más concreta o contacte con info@metmuseum.org."
    ),
    "zh": "我无法通过现有博物馆资料核实这一点。请提供更具体的问题，或联系 info@metmuseum.org。",
}
