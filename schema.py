from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class GalazPrawa(str, Enum):
    """Gałąź prawa określająca ogólną kategorię orzeczenia."""

    CYWILNE = "cywilne"
    KARNE = "karne"
    ADMINISTRACYJNE = "administracyjne"
    PRACY = "pracy"
    UBEZPIECZEN = "ubezpieczeń społecznych"
    GOSPODARCZE = "gospodarcze"
    INNE = "inne"


class StatusLiniiOrzeczniczej(str, Enum):
    """Status orzeczenia w kontekście linii orzeczniczej."""

    KONTYNUACJA = "kontynuacja"
    ZMIANA = "zmiana"
    PRZELOM = "przełom"
    NOWE_ZAGADNIENIE = "nowe zagadnienie"


class MetodaWykladni(str, Enum):
    """Metody wykładni prawa zastosowane w orzeczeniu."""

    JEZYKOWA = "językowa"
    SYSTEMOWA = "systemowa"
    FUNKCJONALNA = "funkcjonalna"
    CELOWOSCIOWA = "celowościowa"
    HISTORYCZNA = "historyczna"
    DYNAMICZNA = "dynamiczna"
    PROUNIJNA = "prounijna"
    KONSTYTUCYJNA = "konstytucyjna"


class Klasyfikacja(BaseModel):
    """Klasyfikacja prawna orzeczenia."""

    galaz_prawa: GalazPrawa = Field(
        ..., description="Główna gałąź prawa, której dotyczy orzeczenie"
    )
    glowna_kategoria_prawna: str = Field(
        ...,
        description="Główna kategoria prawna, np. 'prawo zobowiązań', 'prawo karne materialne'",
    )
    podkategoria_prawna: str = Field(
        ..., description="Podkategoria prawna uszczegóławiająca główną kategorię"
    )
    instytucje_prawne: List[str] = Field(
        ...,
        description="Lista kluczowych instytucji prawnych, których dotyczy orzeczenie",
        min_length=1,
        max_length=10,
    )


class SlowaKluczoweFrazy(BaseModel):
    """Słowa kluczowe i frazy charakteryzujące orzeczenie."""

    slowa_kluczowe: List[str] = Field(
        ...,
        description="Pojedyncze słowa kluczowe charakteryzujące orzeczenie",
        min_length=3,
        max_length=5,
    )
    kluczowe_frazy: List[str] = Field(
        ...,
        description="Krótkie frazy (max 5 słów) charakteryzujące orzeczenie",
    )


class PodstawyPrawnePowiazania(BaseModel):
    """Podstawy prawne i powiązania z innymi aktami i orzeczeniami."""

    podstawy_prawne: List[str] = Field(
        ..., description="Lista przepisów, na które powołuje się sąd"
    )
    przywolane_akty_prawne: List[str] = Field(
        ..., description="Lista sygnatur przywołanych aktów prawnych"
    )
    relacje_z_innymi_orzeczeniami: list[str] = Field(
        default_factory=list,
        description="Informacje o relacjach z innymi orzeczeniami (cytowania, sprzeczności, rozwinięcia)",
    )
    status_linii_orzeczniczej: StatusLiniiOrzeczniczej = Field(
        ..., description="Status w kontekście linii orzeczniczej"
    )
    metody_wykladni: list[MetodaWykladni] = Field(
        default_factory=list, description="Zastosowane metody interpretacji przepisów"
    )


class AnalizaPrawna(BaseModel):
    """Analiza prawna zawarta w orzeczeniu."""

    dylematy_prawne: List[str] = Field(
        ..., description="Kluczowe problemy interpretacyjne rozstrzygnięte w orzeczeniu"
    )
    kluczowe_fragmenty: List[str] = Field(
        ...,
        description="Najważniejsze fragmenty uzasadnienia, które mogą stanowić precedens lub istotną interpretację",
    )
    kontrowersje_i_krytyka: Optional[str] = Field(
        default=None,
        description="Informacje o kontrowersjach lub krytyce związanej z orzeczeniem",
    )


class WyszukiwanieZastosowanie(BaseModel):
    """Informacje przydatne do wyszukiwania i praktycznego zastosowania orzeczenia."""

    potencjalne_pytania: List[str] = Field(
        ...,
        description="Pytania, które mogłyby skierować prawnika do tego orzeczenia",
        min_length=3,
        max_length=5,
    )
    praktyczne_zastosowanie: str = Field(
        ..., description="Wskazówki dotyczące praktycznego wykorzystania orzeczenia"
    )
    waga_precedensowa: int = Field(
        ..., description="Ocena wagi precedensowej w skali 1-5", ge=1, le=5
    )


class Streszczenie(BaseModel):
    """Streszczenie orzeczenia."""

    pelne_streszczenie: str = Field(
        ...,
        description="Pełne streszczenie orzeczenia zawierające od 70 do 300 słów",
        min_length=500,
        max_length=2500,
    )


class OrzeczenieSN(BaseModel):
    """Model reprezentujący ustrukturyzowane podsumowanie orzeczenia Sądu Najwyższego."""

    tytul: str = Field(
        ..., description="Zwięzłe, jednozdaniowe określenie przedmiotu sprawy"
    )
    klasyfikacja: Klasyfikacja
    slowa_kluczowe_frazy: SlowaKluczoweFrazy
    podstawy_prawne_powiazania: PodstawyPrawnePowiazania
    analiza_prawna: AnalizaPrawna
    wyszukiwanie_zastosowanie: WyszukiwanieZastosowanie
    streszczenie: Streszczenie
