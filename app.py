import os
import csv
import itertools
import math
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, send_file
import io
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key')

def price_per_ton(price, bag_size):
    return round(price / bag_size * 2000, 2)

def price_per_bag(price_ton, bag_size):
    return round(price_ton / 2000 * bag_size, 2)

def load_fertilizer_blends(filename="fertilizers.csv"):
    blends = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            bag_size = float(row.get("bag_size", 50))
            price_bag = float(row["price"])
            price_ton_val = price_per_ton(price_bag, bag_size)
            blend = {
                "name": row["name"],
                "n": float(row["n"]),
                "p": float(row["p"]),
                "k": float(row["k"]),
                "price": price_bag,
                "bag_size": bag_size,
                "price_ton": price_ton_val,
                "price_bag": price_bag
            }
            blends.append(blend)
    return blends

def parse_float(val, default=0):
    try:
        if val is None or val == '':
            return default
        return float(val)
    except (TypeError, ValueError):
        return default

def round_half(n):
    return round(n * 2) / 2

def solve_single_blend(n_req, p_req, k_req, blend, price_unit="bag"):
    if (n_req > 0 and blend["n"] == 0) or (p_req > 0 and blend["p"] == 0) or (k_req > 0 and blend["k"] == 0):
        return None
    if price_unit == "ton":
        # tons per acre
        tons_per_acre = 0
        if blend["n"] > 0:
            tons_per_acre = max(tons_per_acre, n_req / (blend["n"] / 100))
        if blend["p"] > 0:
            tons_per_acre = max(tons_per_acre, p_req / (blend["p"] / 100))
        if blend["k"] > 0:
            tons_per_acre = max(tons_per_acre, k_req / (blend["k"] / 100))
        applied_n = tons_per_acre * (blend["n"] / 100)
        applied_p = tons_per_acre * (blend["p"] / 100)
        applied_k = tons_per_acre * (blend["k"] / 100)
        if applied_n < n_req - 1e-6 or applied_p < p_req - 1e-6 or applied_k < k_req - 1e-6:
            return None
        return {
            "type": "single",
            "blends": [(blend, tons_per_acre)],
            "names": [blend["name"]],
            "applied_n": applied_n,
            "applied_p": applied_p,
            "applied_k": applied_k,
            "custom": blend.get("custom", False),
        }
    else:
        lbs_per_acre = 0
        if blend["n"] > 0:
            lbs_per_acre = max(lbs_per_acre, n_req / (blend["n"] / 100))
        if blend["p"] > 0:
            lbs_per_acre = max(lbs_per_acre, p_req / (blend["p"] / 100))
        if blend["k"] > 0:
            lbs_per_acre = max(lbs_per_acre, k_req / (blend["k"] / 100))
        applied_n = lbs_per_acre * (blend["n"] / 100)
        applied_p = lbs_per_acre * (blend["p"] / 100)
        applied_k = lbs_per_acre * (blend["k"] / 100)
        if applied_n < n_req - 1e-6 or applied_p < p_req - 1e-6 or applied_k < k_req - 1e-6:
            return None
        return {
            "type": "single",
            "blends": [(blend, lbs_per_acre)],
            "names": [blend["name"]],
            "applied_n": applied_n,
            "applied_p": applied_p,
            "applied_k": applied_k,
            "custom": blend.get("custom", False),
        }

def solve_two_blend(n_req, p_req, k_req, blend1, blend2, price_unit="bag"):
    if blend1["name"] == blend2["name"]:
        return None
    for idx, req in enumerate([n_req, p_req, k_req]):
        if req > 0 and blend1[["n", "p", "k"][idx]] == 0 and blend2[["n", "p", "k"][idx]] == 0:
            return None
    if price_unit == "ton":
        A = np.array([
            [blend1["n"] / 100, blend2["n"] / 100],
            [blend1["p"] / 100, blend2["p"] / 100],
            [blend1["k"] / 100, blend2["k"] / 100]
        ])
        b = np.array([n_req, p_req, k_req])
        rows = [i for i, req in enumerate(b) if req > 0]
        if not rows:
            return None
        A = A[rows]
        b = b[rows]
        try:
            res = np.linalg.lstsq(A, b, rcond=None)
            x = np.maximum(res[0], 0)
            if any(xi < 1e-3 for xi in x):
                return None
            applied = A @ x
            if any(applied[i] < b[i] - 1e-6 for i in range(len(b))):
                return None
            applied_n = x[0] * blend1["n"] / 100 + x[1] * blend2["n"] / 100
            applied_p = x[0] * blend1["p"] / 100 + x[1] * blend2["p"] / 100
            applied_k = x[0] * blend1["k"] / 100 + x[1] * blend2["k"] / 100
            return {
                "type": "combo",
                "blends": [(blend1, x[0]), (blend2, x[1])],
                "names": [blend1["name"], blend2["name"]],
                "applied_n": applied_n,
                "applied_p": applied_p,
                "applied_k": applied_k,
                "custom": blend1.get("custom", False) or blend2.get("custom", False),
            }
        except Exception:
            return None
    else:
        # bag mode (lbs)
        A = np.array([
            [blend1["n"] / 100, blend2["n"] / 100],
            [blend1["p"] / 100, blend2["p"] / 100],
            [blend1["k"] / 100, blend2["k"] / 100]
        ])
        b = np.array([n_req, p_req, k_req])
        rows = [i for i, req in enumerate(b) if req > 0]
        if not rows:
            return None
        A = A[rows]
        b = b[rows]
        try:
            res = np.linalg.lstsq(A, b, rcond=None)
            x = np.maximum(res[0], 0)
            if any(xi < 1e-3 for xi in x):
                return None
            applied = A @ x
            if any(applied[i] < b[i] - 1e-6 for i in range(len(b))):
                return None
            applied_n = x[0] * blend1["n"] / 100 + x[1] * blend2["n"] / 100
            applied_p = x[0] * blend1["p"] / 100 + x[1] * blend2["p"] / 100
            applied_k = x[0] * blend1["k"] / 100 + x[1] * blend2["k"] / 100
            return {
                "type": "combo",
                "blends": [(blend1, x[0]), (blend2, x[1])],
                "names": [blend1["name"], blend2["name"]],
                "applied_n": applied_n,
                "applied_p": applied_p,
                "applied_k": applied_k,
                "custom": blend1.get("custom", False) or blend2.get("custom", False),
            }
        except Exception:
            return None

def apply_bag_ton_rounding(candidate, area, area_type, total_n, total_p, total_k, price_unit="bag"):
    if area_type == 'acres':
        total_area = area
    else:
        total_area = area / 43560

    blends_result = []
    if price_unit == "ton":
        # Tons purchase scenario
        for blend, tons_per_acre in candidate["blends"]:
            total_tons = tons_per_acre * total_area
            tons_rounded = round_half(total_tons)
            blends_result.append({
                "name": blend["name"],
                "tons": tons_rounded,
                "cost": tons_rounded * blend["price_ton"],
                "applied_n": tons_rounded * blend["n"] / 100 * 2000,
                "applied_p": tons_rounded * blend["p"] / 100 * 2000,
                "applied_k": tons_rounded * blend["k"] / 100 * 2000,
                "n": blend["n"],
                "p": blend["p"],
                "k": blend["k"],
                "price_ton": blend["price_ton"],
                "custom": blend.get("custom", False)
            })
        total_cost = sum(b["cost"] for b in blends_result)
        total_applied_n = sum(b["applied_n"] for b in blends_result)
        total_applied_p = sum(b["applied_p"] for b in blends_result)
        total_applied_k = sum(b["applied_k"] for b in blends_result)
        # 20% rule after rounding
        if (total_applied_n < total_n - 1e-6 or
            total_applied_p < total_p - 1e-6 or
            total_applied_k < total_k - 1e-6):
            return None
        if (total_applied_n > total_n * 1.2 + 1e-6 or
            total_applied_p > total_p * 1.2 + 1e-6 or
            total_applied_k > total_k * 1.2 + 1e-6):
            return None
        return {
            "type": candidate["type"],
            "names": candidate["names"],
            "blends": blends_result,
            "total_cost": total_cost,
            "applied_n": total_applied_n,
            "applied_p": total_applied_p,
            "applied_k": total_applied_k,
            "custom": candidate.get("custom", False),
            "unit": "ton"
        }
    else:
        # Bags purchase scenario
        for blend, lbs_per_acre in candidate["blends"]:
            lbs_total = lbs_per_acre * total_area
            bags = math.ceil(lbs_total / blend["bag_size"])
            blends_result.append({
                "name": blend["name"],
                "bags": bags,
                "cost": bags * blend["price_bag"],
                "applied_n": bags * blend["bag_size"] * (blend["n"]/100),
                "applied_p": bags * blend["bag_size"] * (blend["p"]/100),
                "applied_k": bags * blend["bag_size"] * (blend["k"]/100),
                "n": blend["n"],
                "p": blend["p"],
                "k": blend["k"],
                "price_bag": blend["price_bag"],
                "bag_size": blend["bag_size"],
                "custom": blend.get("custom", False)
            })
        total_cost = sum(b["cost"] for b in blends_result)
        total_applied_n = sum(b["applied_n"] for b in blends_result)
        total_applied_p = sum(b["applied_p"] for b in blends_result)
        total_applied_k = sum(b["applied_k"] for b in blends_result)
        if (total_applied_n < total_n - 1e-6 or
            total_applied_p < total_p - 1e-6 or
            total_applied_k < total_k - 1e-6):
            return None
        if (total_applied_n > total_n * 1.2 + 1e-6 or
            total_applied_p > total_p * 1.2 + 1e-6 or
            total_applied_k > total_k * 1.2 + 1e-6):
            return None
        return {
            "type": candidate["type"],
            "names": candidate["names"],
            "blends": blends_result,
            "total_cost": total_cost,
            "applied_n": total_applied_n,
            "applied_p": total_applied_p,
            "applied_k": total_applied_k,
            "custom": candidate.get("custom", False),
            "unit": "bag"
        }

def deduplicate_solutions(solutions):
    seen = set()
    unique = []
    for sol in solutions:
        if sol.get("unit") == "ton":
            key = tuple(sorted((b['name'], b['tons']) for b in sol['blends']))
        else:
            key = tuple(sorted((b['name'], b['bags']) for b in sol['blends']))
        if key not in seen:
            seen.add(key)
            unique.append(sol)
    return unique

def extract_custom_blends_from_form(form, price_unit="bag"):
    custom_blends = []
    for j in range(2):
        name = form.get(f"custom_blend_{j}_name", "").strip()
        if not name:
            continue
        try:
            n_val = parse_float(form.get(f"custom_blend_{j}_n", 0))
            p_val = parse_float(form.get(f"custom_blend_{j}_p", 0))
            k_val = parse_float(form.get(f"custom_blend_{j}_k", 0))
        except Exception:
            continue
        try:
            price = parse_float(form.get(f"custom_blend_{j}_price", 0))
        except Exception:
            price = 0
        if price_unit == "ton":
            bag_size = 2000  # in lbs, but hidden
            price_bag = price_per_bag(price, bag_size)
            price_ton_val = price
        else:
            bag_size = parse_float(form.get(f"custom_blend_{j}_bag_size", 50))
            if bag_size <= 0:
                bag_size = 50
            price_ton_val = price_per_ton(price, bag_size)
            price_bag = price
        custom_blends.append({
            "name": name,
            "n": n_val,
            "p": p_val,
            "k": k_val,
            "price_bag": price_bag,
            "price_ton": price_ton_val,
            "bag_size": bag_size,
            "price": price if price_unit == "bag" else price_ton_val,
            "custom": True
        })
    return custom_blends

def get_recommendations(n_req, p_req, k_req, all_blends, area, area_type, user_selected_blends=None, custom_blends=None, price_unit="bag"):
    if area_type == "acres":
        total_n = n_req * area
        total_p = p_req * area
        total_k = k_req * area
    else:
        total_n = n_req * area / 43560
        total_p = p_req * area / 43560
        total_k = k_req * area / 43560

    filtered_blends = [b for b in all_blends if (not user_selected_blends or b["name"] in user_selected_blends)]
    candidate_blends = list(filtered_blends)
    if custom_blends is not None and len(custom_blends) > 0:
        candidate_blends += custom_blends

    single_blend_candidates = []
    for blend in candidate_blends:
        res = solve_single_blend(n_req, p_req, k_req, blend, price_unit=price_unit)
        if res:
            rounded = apply_bag_ton_rounding(res, area, area_type, total_n, total_p, total_k, price_unit=price_unit)
            if rounded:
                rounded['excess_n'] = rounded["applied_n"] - total_n
                rounded['excess_p'] = rounded["applied_p"] - total_p
                rounded['excess_k'] = rounded["applied_k"] - total_k
                rounded['excess_warning'] = ""
                single_blend_candidates.append(rounded)

    combo_candidates = []
    for blend1, blend2 in itertools.combinations(candidate_blends, 2):
        if blend1["name"] == blend2["name"]:
            continue
        res = solve_two_blend(n_req, p_req, k_req, blend1, blend2, price_unit=price_unit)
        if res:
            rounded = apply_bag_ton_rounding(res, area, area_type, total_n, total_p, total_k, price_unit=price_unit)
            if rounded:
                rounded['excess_n'] = rounded["applied_n"] - total_n
                rounded['excess_p'] = rounded["applied_p"] - total_p
                rounded['excess_k'] = rounded["applied_k"] - total_k
                rounded['excess_warning'] = ""
                combo_candidates.append(rounded)

    all_candidates = single_blend_candidates + combo_candidates
    all_candidates = deduplicate_solutions(all_candidates)

    avg_cost = sum(r["total_cost"] for r in all_candidates) / len(all_candidates) if all_candidates else 1
    for r in all_candidates:
        r["total_excess"] = r["excess_n"] + r["excess_p"] + r["excess_k"]
        r["score"] = (r["total_cost"] - avg_cost) * 2 + r["total_excess"]

    # Sort for cost and waste, then reorder if least wasteful and least cost are not the same
    if all_candidates:
        sorted_by_cost = sorted(all_candidates, key=lambda r: r["total_cost"])
        sorted_by_waste = sorted(all_candidates, key=lambda r: r["total_excess"])
        least_cost = sorted_by_cost[0]
        least_waste = sorted_by_waste[0]
        if least_cost is least_waste:
            top3 = sorted_by_cost[:3]
        else:
            top3 = [least_waste, least_cost]
            # fill in a third (unique) if available
            for cand in all_candidates:
                if cand is not least_waste and cand is not least_cost:
                    top3.append(cand)
                    break
    else:
        top3 = []

    # Highlight assignment
    if top3:
        min_cost = min(r["total_cost"] for r in top3)
        min_waste = min(r["total_excess"] for r in top3)
        for r in top3:
            r["highlight"] = ""
            is_min_cost = abs(r["total_cost"] - min_cost) < 1e-3
            is_min_waste = abs(r["total_excess"] - min_waste) < 1e-3
            if is_min_cost and is_min_waste:
                r["highlight"] = "highlight-green"
            elif is_min_cost:
                r["highlight"] = "highlight-orange"
            elif is_min_waste:
                r["highlight"] = "highlight-blue"

    note = None
    if not top3:
        note = "No solution could be found with the selected blends and prescription. Try selecting more blends or adjusting your N, P, K requirements."

    custom_results = []
    if custom_blends is not None and len(custom_blends) > 0:
        if len(custom_blends) == 1:
            blend = custom_blends[0]
            res = solve_single_blend(n_req, p_req, k_req, blend, price_unit=price_unit)
            if res:
                rounded = apply_bag_ton_rounding(res, area, area_type, total_n, total_p, total_k, price_unit=price_unit)
                if rounded:
                    rounded['excess_n'] = rounded["applied_n"] - total_n
                    rounded['excess_p'] = rounded["applied_p"] - total_p
                    rounded['excess_k'] = rounded["applied_k"] - total_k
                    rounded['custom'] = True
                    rounded['note'] = "Custom blend result (may not fully meet all requirements)."
                    custom_results.append(rounded)
        elif len(custom_blends) == 2:
            blend1, blend2 = custom_blends
            res = solve_two_blend(n_req, p_req, k_req, blend1, blend2, price_unit=price_unit)
            if res:
                rounded = apply_bag_ton_rounding(res, area, area_type, total_n, total_p, total_k, price_unit=price_unit)
                if rounded:
                    rounded['excess_n'] = rounded["applied_n"] - total_n
                    rounded['excess_p'] = rounded["applied_p"] - total_p
                    rounded['excess_k'] = rounded["applied_k"] - total_k
                    rounded['custom'] = True
                    rounded['note'] = "Custom combination result (may not fully meet all requirements)."
                    custom_results.append(rounded)
    return top3, custom_results, note

@app.route('/', methods=['GET', 'POST'])
def index():
    blends = load_fertilizer_blends()
    form_defaults = session.get('inputs', {})
    error = None

    # Support for global price unit switching
    price_unit = request.form.get("price_unit") if request.method == "POST" else form_defaults.get("price_unit", "bag")
    if not price_unit:
        price_unit = "bag"

    unit_change = False
    if request.method == "POST":
        unit_change = request.form.get('unit_change') == '1'

    for blend in blends:
        if price_unit == "ton":
            blend["price"] = blend["price_ton"]
        else:
            blend["price"] = blend["price_bag"]

    if request.method == 'POST':
        if unit_change:
            form_defaults = request.form.copy()
            form_defaults = dict(form_defaults)
            form_defaults['price_unit'] = price_unit
            return render_template('index.html', blends=blends, error=None, form_defaults=form_defaults, price_unit=price_unit)
        try:
            n = parse_float(request.form.get('n'), 0)
            p = parse_float(request.form.get('p'), 0)
            k = parse_float(request.form.get('k'), 0)
            area = parse_float(request.form.get('area'), 0)
        except ValueError:
            error = "All numeric fields must be valid numbers greater than zero."
            return render_template('index.html', blends=blends, error=error, form_defaults=request.form, price_unit=price_unit)
        area_type = request.form.get('area_type')
        selected_blends = request.form.getlist('blends')

        custom_enabled = request.form.get("custom_enabled")
        custom_blends = []
        if custom_enabled:
            custom_blends = extract_custom_blends_from_form(request.form, price_unit=price_unit)
        session['custom_blends'] = custom_blends

        for blend in blends:
            bname_key = blend['name'].replace(' ', '_')
            price_key = f"price_{bname_key}"
            bag_size_key = f"bag_size_{bname_key}"
            if price_key in request.form:
                try:
                    price_val = parse_float(request.form[price_key])
                    if price_unit == 'ton':
                        blend['price_ton'] = price_val
                        blend['price_bag'] = price_per_bag(price_val, blend['bag_size'])
                        blend['price'] = price_val
                    else:
                        blend['price_bag'] = price_val
                        blend['price_ton'] = price_per_ton(price_val, blend['bag_size'])
                        blend['price'] = price_val
                except Exception:
                    pass
            if bag_size_key in request.form and price_unit == "bag":
                try:
                    bag_size_val = parse_float(request.form[bag_size_key], 50)
                    if bag_size_val > 0:
                        blend['bag_size'] = bag_size_val
                        blend['price_ton'] = price_per_ton(blend['price_bag'], blend['bag_size'])
                    else:
                        blend['bag_size'] = 50
                except Exception:
                    blend['bag_size'] = 50

        session['inputs'] = {
            'n': n, 'p': p, 'k': k,
            'area': area, 'area_type': area_type,
            'selected_blends': selected_blends,
            'price_unit': price_unit
        }
        session['blend_prices'] = {blend['name']: blend['price'] for blend in blends}
        session['blend_bagsizes'] = {blend['name']: blend['bag_size'] for blend in blends}
        return redirect(url_for('results'))

    blend_prices = session.get('blend_prices', {})
    blend_bagsizes = session.get('blend_bagsizes', {})
    for blend in blends:
        if blend['name'] in blend_prices:
            blend['price'] = blend_prices[blend['name']]
        if blend['name'] in blend_bagsizes:
            blend['bag_size'] = blend_bagsizes[blend['name']]
    return render_template('index.html', blends=blends, error=error, form_defaults=form_defaults, price_unit=price_unit)

@app.route('/results', methods=['GET', 'POST'])
def results():
    blends = load_fertilizer_blends()
    inputs = session.get('inputs', {})
    n = parse_float(inputs.get('n'), 0)
    p = parse_float(inputs.get('p'), 0)
    k = parse_float(inputs.get('k'), 0)
    area = parse_float(inputs.get('area'), 0)
    area_type = inputs.get('area_type', 'acres')
    selected_blend_names = inputs.get('selected_blends', [])
    custom_blends = session.get('custom_blends', [])
    price_unit = inputs.get('price_unit', 'bag')

    top3, custom_results, note = get_recommendations(
        n, p, k, blends, area, area_type, user_selected_blends=selected_blend_names, custom_blends=custom_blends, price_unit=price_unit
    )
    for rec in top3:
        rec['total_applied_n'] = rec.get('applied_n', 0)
        rec['total_applied_p'] = rec.get('applied_p', 0)
        rec['total_applied_k'] = rec.get('applied_k', 0)
    for rec in custom_results:
        rec['total_applied_n'] = rec.get('applied_n', 0)
        rec['total_applied_p'] = rec.get('applied_p', 0)
        rec['total_applied_k'] = rec.get('applied_k', 0)
    all_results = []
    if top3:
        all_results.extend(top3)
    return render_template('results.html', results=all_results, custom_results=custom_results, not_typical=False, fallback_used=False, best_note=note, price_unit=price_unit)

@app.route('/download_pdf')
def download_pdf():
    blends = load_fertilizer_blends()
    inputs = session.get('inputs', {})
    n = parse_float(inputs.get('n'), 0)
    p = parse_float(inputs.get('p'), 0)
    k = parse_float(inputs.get('k'), 0)
    area = parse_float(inputs.get('area'), 0)
    area_type = inputs.get('area_type', 'acres')
    selected_blend_names = inputs.get('selected_blends', [])
    custom_blends = session.get('custom_blends', [])
    price_unit = inputs.get('price_unit', 'bag')

    top3, custom_results, note = get_recommendations(
        n, p, k, blends, area, area_type, user_selected_blends=selected_blend_names, custom_blends=custom_blends, price_unit=price_unit
    )
    for rec in top3:
        rec['total_applied_n'] = rec.get('applied_n', 0)
        rec['total_applied_p'] = rec.get('applied_p', 0)
        rec['total_applied_k'] = rec.get('applied_k', 0)
    for rec in custom_results:
        rec['total_applied_n'] = rec.get('applied_n', 0)
        rec['total_applied_p'] = rec.get('applied_p', 0)
        rec['total_applied_k'] = rec.get('applied_k', 0)
    all_results = []
    if top3:
        all_results.extend(top3)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Davidson Fertilizer Calculator Results", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Prescription: N={n} P={p} K={k} Area={area} {area_type}", ln=True)
    if note:
        pdf.set_text_color(200,80,0)
        pdf.cell(0, 10, note, ln=True)
        pdf.set_text_color(0,0,0)
    for idx, result in enumerate(all_results):
        pdf.ln(4)
        pdf.set_font("Arial", 'B', 12)
        if result.get("excess_warning"):
            pdf.set_text_color(200,80,0)
            pdf.cell(0, 8, result["excess_warning"], ln=True)
            pdf.set_text_color(0,0,0)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, f"Type: {result['type'].capitalize()} | {' + '.join(result['names'])}", ln=True)
        pdf.cell(0, 8, f"  Total Cost: ${result['total_cost']:.2f}", ln=True)
        pdf.cell(0, 8, f"  Total N applied: {result.get('total_applied_n', 0):.2f} lbs", ln=True)
        pdf.cell(0, 8, f"  Total P applied: {result.get('total_applied_p', 0):.2f} lbs", ln=True)
        pdf.cell(0, 8, f"  Total K applied: {result.get('total_applied_k', 0):.2f} lbs", ln=True)
        pdf.cell(0, 8, f"  Excess N: {result.get('excess_n',0):.2f} lbs", ln=True)
        pdf.cell(0, 8, f"  Excess P: {result.get('excess_p',0):.2f} lbs", ln=True)
        pdf.cell(0, 8, f"  Excess K: {result.get('excess_k',0):.2f} lbs", ln=True)
        for blend in result["blends"]:
            if result.get("unit") == "ton":
                pdf.cell(0, 8, f"    - {blend['name']}: {blend['tons']} tons @ ${blend['cost']:.2f} (N: {blend['applied_n']:.1f}, P: {blend['applied_p']:.1f}, K: {blend['applied_k']:.1f})", ln=True)
            else:
                pdf.cell(0, 8, f"    - {blend['name']}: {blend['bags']} bags @ ${blend['cost']:.2f} (N: {blend['applied_n']:.1f}, P: {blend['applied_p']:.1f}, K: {blend['applied_k']:.1f})", ln=True)
        pdf.ln(2)
    if custom_results:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 12, "Custom Solutions", ln=True)
    for rec in custom_results:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Custom Solution", ln=True)
        if rec.get("note"):
            pdf.set_text_color(200,80,0)
            pdf.cell(0, 8, rec["note"], ln=True)
            pdf.set_text_color(0,0,0)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, f"Type: {rec['type'].replace('custom-', '').capitalize()} | {' + '.join(rec['names'])}", ln=True)
        pdf.cell(0, 8, f"  Total Cost: ${rec['total_cost']:.2f}", ln=True)
        pdf.cell(0, 8, f"  Total N applied: {rec.get('applied_n', 0):.2f} lbs", ln=True)
        pdf.cell(0, 8, f"  Total P applied: {rec.get('applied_p', 0):.2f} lbs", ln=True)
        pdf.cell(0, 8, f"  Total K applied: {rec.get('applied_k', 0):.2f} lbs", ln=True)
        pdf.cell(0, 8, f"  Excess N: {rec.get('excess_n',0):.2f} lbs", ln=True)
        pdf.cell(0, 8, f"  Excess P: {rec.get('excess_p',0):.2f} lbs", ln=True)
        pdf.cell(0, 8, f"  Excess K: {rec.get('excess_k',0):.2f} lbs", ln=True)
        for blend in rec["blends"]:
            if rec.get("unit") == "ton":
                pdf.cell(0, 8, f"    - {blend['name']}: {blend['tons']} tons @ ${blend['cost']:.2f} (N: {blend['applied_n']:.1f}, P: {blend['applied_p']:.1f}, K: {blend['applied_k']:.1f})", ln=True)
            else:
                pdf.cell(0, 8, f"    - {blend['name']}: {blend['bags']} bags @ ${blend['cost']:.2f} (N: {blend['applied_n']:.1f}, P: {blend['applied_p']:.1f}, K: {blend['applied_k']:.1f})", ln=True)
        pdf.ln(2)
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_output = io.BytesIO(pdf_bytes)
    pdf_output.seek(0)
    return send_file(pdf_output, mimetype='application/pdf', as_attachment=True, download_name="fertilizer_recommendations.pdf")

@app.route('/share')
def share():
    return render_template('share.html')

if __name__ == '__main__':
    app.run(debug=True)