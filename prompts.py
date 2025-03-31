from jinja2 import Template

from utils import calculate_summary_length


def get_system_prompt(text_length: int) -> str:
    """
    Generate a system prompt with dynamically calculated summary length.

    Args:
        text_length: Length of the original text in characters

    Returns:
        Formatted system prompt with calculated summary length
    """
    calculated_length = calculate_summary_length(text_length)

    template: Template = Template("""
Wygeneruj ustrukturyzowane podsumowanie orzeczenia Sądu Najwyższego, zawierające następujące elementy:

1. Tytuł [zwięzłe, jednozdaniowe określenie przedmiotu sprawy]
2. Klasyfikacja
   - Gałąź prawa: [cywilne/karne/administracyjne/inne]
   - Główna kategoria prawna: 
   - Podkategoria prawna:
   - Instytucje prawne: [kluczowe instytucje prawne, których dotyczy orzeczenie]
4. Słowa kluczowe i frazy
   - Słowa kluczowe [3–5 pojedynczych słów]
   - Kluczowe frazy [2-5 fraz, każda max. 5 słów]
5. Podstawy prawne i powiązania
   - Podstawy prawne: [przepisy, na które powołuje się sąd]
   - Przywołane akty prawne: [sygnatury przywołanych aktów prawnych]
   - Relacje z innymi orzeczeniami: [cytowania, sprzeczności, rozwinięcia]   
   - Status linii orzeczniczej: [kontynuacja/zmiana/przełom w linii orzeczniczej]
   - Metody wykładni: [zastosowane metody interpretacji przepisów]
6. Analiza prawna
   - Dylematy prawne: [kluczowe problemy interpretacyjne]
   - Kluczowe fragmenty:  [ekstrakcja kluczowych fragmentów - wyodrębnienie najważniejszych fragmentów uzasadnienia, które mogą stanowić precedens lub istotną interpretację przepisów]
   - Kontrowersje i krytyka: [opcjonalnie]
7. Wyszukiwanie i zastosowanie
   - Potencjalne pytania: [3-5 pytań, które mogłyby skierować prawnika do tego orzeczenia]
   - Praktyczne zastosowanie: [wskazówki do wykorzystania]
   - Waga precedensowa: [1-5]
8. Streszczenie składające się z około {{ summary_length }} słów zawierające:
   - Definicję problemu prawnego
   - Kluczowe argumenty i uzasadnienia sądu
   - Konkluzję i implikacje rozstrzygnięcia
Streszczenie powinno unikać szczegółów dotyczących konkretnych osób czy stron postępowania, koncentrując się na ogólnych zasadach prawnych i argumentacji istotnej dla przyszłych spraw.]

WYTYCZNE:
- Koncentruj się na aspektach prawnych, nie na detalach faktycznych sprawy
- Używaj precyzyjnej terminologii prawniczej
- Pomijaj informacje identyfikujące konkretne osoby fizyczne
- Unikaj stronniczości i ocen wartościujących
- Podkreślaj elementy precedensowe i uniwersalne, które mogą mieć zastosowanie w innych sprawach
""")

    return template.render(summary_length=calculated_length)
