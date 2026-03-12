import re
class ScopeResolver:
    CERT = re.compile(
        r"\b(cert|certificate|certification|credential|badge)\b", re.I)
    PROJECT = re.compile(
        r"\b(project|projects|portfolio|dashboard|chatbot|solution)\b", re.I)
    RECOMMEND = re.compile(
        r"\b(recommendation|reference letter|referee|endorsement)\b", re.I)
    EXPERIENCE = re.compile(
        r"\b(experience letter|employment letter|employment verification|service letter)\b", re.I)
    CV = re.compile(r"\b(cv|resume)\b", re.I)
    CONTACT = re.compile(r"\b(contact|email|phone|whatsapp|linkedin)\b", re.I)

    @classmethod
    def resolve_filters(cls, message: str):
        msg = (message or "").strip()
        if not msg:
            return None

        if cls.CERT.search(msg):
            return {"document_type": "certificates"}
        if cls.EXPERIENCE.search(msg):
            return {"document_type": "experience_letter"}
        if cls.RECOMMEND.search(msg):
            return {"document_type": "recommendation"}
        if cls.PROJECT.search(msg):
            return {"document_type": "projects"}
        if cls.CONTACT.search(msg) or cls.CV.search(msg):
            return {"document_type": "cv"}

        return None
