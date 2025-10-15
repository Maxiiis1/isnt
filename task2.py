import requests
from shapely.geometry import LineString, Point
import time

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def get_street_geometries(street_name):
    query = f"""
    [out:json][timeout:200];
    area["name"="Санкт-Петербург"]["boundary"="administrative"]->.spb;
    way["highway"]["name"="{street_name}"](area.spb);
    out geom;
    """

    response = requests.post(OVERPASS_URL, data={"data": query})
    if response.status_code != 200:
        return []

    data = response.json()
    lines = []
    for el in data.get("elements", []):
        if "geometry" in el:
            coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(coords) > 1:
                lines.append(LineString(coords))
    return lines


def find_intersections(street1, street2):
    lines1 = get_street_geometries(street1)
    lines2 = get_street_geometries(street2)
    intersections = set()

    for l1 in lines1:
        for l2 in lines2:
            inter = l1.intersection(l2)
            if inter.is_empty:
                continue
            if isinstance(inter, Point):
                intersections.add((round(inter.y, 6), round(inter.x, 6)))
            elif inter.geom_type == "MultiPoint":
                for p in inter.geoms:
                    intersections.add((round(p.y, 6), round(p.x, 6)))
    return sorted(intersections)


if __name__ == "__main__":
    pairs = [
        ("Большая Пушкарская улица", "Съезжинская улица"),
        ("Московский проспект", "Благодатная улица"),
        ("Московский проспект", "Киевская улица"),
        ("Шамшева улица", "Большая Пушкарская улица"),
        ("Приморский проспект", "Невский проспект"),
        ("Невский проспект", "Литейный проспект"),
         ]

    for s1, s2 in pairs:
        print(f"\n{s1} × {s2}:")
        try:
            result = find_intersections(s1, s2)
            if result:
                for c in result:
                    print("  →", c)
            else:
                print("  → пересечений нет")
        except Exception as e:
            print(" ️ Ошибка:", e)
        time.sleep(1)
