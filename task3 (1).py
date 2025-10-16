import json
import re
import requests
from geopy.distance import geodesic
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pymorphy3
from rapidfuzz import fuzz

morph = pymorphy3.MorphAnalyzer()

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def load_gazetteer_from_overpass(city="Санкт-Петербург"):
    streets_query = f"""
    [out:json][timeout:180];
    area["name"="{city}"]->.spb;
    (
      way["highway"]["name"](area.spb);
    );
    out center;
    """
    metro_query = f"""
    [out:json][timeout:180];
    area["name"="{city}"]->.spb;
    (
      node["railway"="station"]["station"="subway"](area.spb);
    );
    out body;
    """

    streets, metro = [], []

    r = requests.post(OVERPASS_URL, data={"data": streets_query})
    streets_res = r.json()
    for el in streets_res.get("elements", []):
        name = el.get("tags", {}).get("name")
        if not name:
            continue
        if "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue
        streets.append({"type": "street", "name": name.lower(), "lat": lat, "lon": lon})

    r = requests.post(OVERPASS_URL, data={"data": metro_query})
    metro_res = r.json()
    for el in metro_res.get("elements", []):
        name = el.get("tags", {}).get("name")
        if not name:
            continue
        if el.get("tags", {}).get("station") != "subway":
            continue
        lat, lon = el["lat"], el["lon"]
        metro.append({"type": "metro", "name": name.lower(), "lat": lat, "lon": lon})

    gazetteer = streets + metro
    print(f"Улиц найдено: {len(streets)}")
    print(f"Станций метро найдено: {len(metro)}")
    print(f"Газетир готов. Всего объектов: {len(gazetteer)}")
    return gazetteer

def load_ner_model():
    model_name = "aidarmusin/address-ner-ru"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner_pipeline

STOP_WORDS = {"с", "на", "у", "около", "перекресток", "перекр", "улица", "проспект", "шоссе",
              "киа", "машина", "автомобиль", "авто", "поворот", "светофор", "дтп"}
INVALID_LOCATION_WORDS = {"метро", "улица", "проспект", "шоссе", "переулок", "площадь", "перекресток"}

def lemmatize_phrase(phrase):
    words = phrase.split()
    lemmas = [morph.parse(w)[0].normal_form for w in words if len(w) > 2 and w not in STOP_WORDS]
    return " ".join(lemmas)

def clean_word(word):
    word = word.lower()
    word = re.sub(r"[.,]", "", word)
    word = re.sub(r"##", "", word)
    return word.strip()

def is_valid_location(name):
    name_clean = name.lower().strip()
    if name_clean in INVALID_LOCATION_WORDS or len(name_clean) < 3:
        return False
    return True

def extract_locations(text, ner_pipeline):
    entities = ner_pipeline(text)
    found_locations = []

    sentences = re.split(r"[.!?]", text)
    for sent in sentences:
        if not any(k in sent.lower() for k in ["дтп", "авария", "столкновение", "перекресток", "метро"]):
            continue
        for ent in entities:
            label = ent.get("entity_group", "")
            word = clean_word(ent["word"])
            if label in ["Street", "Road", "Address", "Location", "Metro"] and word:
                lemma = lemmatize_phrase(word)
                if lemma and is_valid_location(lemma):
                    found_locations.append(lemma)
        if found_locations:
            break

    return found_locations[:1]

def find_coords_by_name(name, gazetteer):
    if not is_valid_location(name):
        return None, None, None

    name_lower = name.lower()
    is_metro = "метро" in name_lower
    candidates = [obj for obj in gazetteer if (obj["type"] == "metro") == is_metro]

    best_match = None
    best_score = 0
    for obj in candidates:
        score = fuzz.partial_ratio(name_lower, obj["name"])
        if score > best_score:
            best_score = score
            best_match = obj

    if best_match and best_score >= 60:
        return best_match["lat"], best_match["lon"], best_match
    else:
        return None, None, None

if __name__ == "__main__":
    gazetteer = load_gazetteer_from_overpass()
    ner_pipeline = load_ner_model()

    with open("rta_texts.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, entry in enumerate(data["text_list"], start=1):
        text = entry["text"]
        true_coords = entry.get("rta_coords")

        found_locs = extract_locations(text, ner_pipeline)

        print(f"\nТекст {i}")
        if not found_locs:
            print("Найденные улицы/метро: нет")
            continue

        dtp_lat, dtp_lon, obj = find_coords_by_name(found_locs[0], gazetteer)

        if dtp_lat is not None:
            print(f"Использованная для расчета локация: {found_locs[0]} → ({obj['lat']:.6f}, {obj['lon']:.6f}) | Газетир: {obj}")
            print(f"Найденная точка ДТП: ({dtp_lat:.6f}, {dtp_lon:.6f})")
            if true_coords:
                dist = geodesic(true_coords, (dtp_lat, dtp_lon)).meters
                print(f"Расстояние до истинной точки: {dist:.1f} м")
        else:
            print("Координаты не найдены")
