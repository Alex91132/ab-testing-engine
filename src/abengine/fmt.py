"""Числа по-русски: десятичная запятая, узкий пробел между разрядами, настоящий минус."""


def num(x: float, d: int = 0) -> str:
    s = f"{x:,.{d}f}".replace(",", " ").replace(".", ",")
    return s.replace("-", "−")


def pct(x: float, d: int = 1, sign: bool = False) -> str:
    s = num(100 * x, d) + " %"
    return ("+" + s) if sign and x > 0 else s


def pval(p: float) -> str:
    if p < 0.0001:
        return "< 0,0001" if p >= 1e-300 else "≈ 0"
    return num(p, 4 if p < 0.01 else 3)


def ci(lo: float, hi: float, d: int = 1) -> str:
    return f"[{pct(lo, d, True)}; {pct(hi, d, True)}]"


def p_eq(p: float) -> str:
    """«p = 0,0081» или «p < 0,0001»."""
    s = pval(p)
    return "p " + s if s[0] in "<≈" else "p = " + s
