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

def load_fertilizer_blends(filename="fertilizers.csv"):
    blends = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            blends.append({
                "name": row["name"],
                "n": float(row["n"]),
                "p": float(row["p"]),
                "k": float(row["k"]),
                "price": float(row["price"]),
                "bag_size": float(row.get("bag_size", 50)),
            })
    return blends

def solve_single_blend(n_req, p_req, k_req, blend):
    if (n_req > 0 and blend["n"] == 0) or (p_req > 0 and blend["p"] == 0) or (k_req > 0 and blend["k"] == 0):
        return None
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

def solve_two_blend(n_req, p_req, k_req, blend1, blend2):
    if blend1["name"] == blend2["name"]:
        return None
    for idx, req in enumerate([n_req, p_req, k_req]):
        if req > 0 and blend1[["n", "p", "k"][idx]] == 0 and blend2[["n", "p", "k"][idx]] == 0:
            return None
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

def apply_bag_rounding(candidate, area, area_type, total_n, total_p, total_k):
    if area_type == 'acres':
        total_area = area
    else:
        total_area = area / 43560
    blends_result = []
    for blend, lbs_per_acre in candidate["blends"]:
        lbs_total = lbs_per_acre * total_area
        bags = math.ceil(lbs_total / blend["bag_size"])
        blends_result.append({
            "name": blend["name"],
            "bags": bags,
            "cost": bags * blend["price"],
            "applied_n": bags * blend["bag_size"] * (blend["n"]/100),
            "applied_p": bags * blend["bag_size"] * (blend["p"]/100),
            "applied_k": bags * blend["bag_size"] * (blend["k"]/100),
            "n": blend["n"],
            "p": blend["p"],
            "k": blend["k"],
            "price": blend["price"],
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
    return {
        "type": candidate["type"],
        "names": candidate["names"],
        "blends": blends_result,
        "total_cost": total_cost,
        "applied_n": total_applied_n,
        "applied_p": total_applied_p,
        "applied_k": total_applied_k,
        "custom": candidate.get("custom", False)
    }

def deduplicate_solutions(solutions):
    seen = set()
    unique = []
    for sol in solutions:
        key = tuple(sorted((b['name'], b['bags']) for b in sol['blends']))
        if key not in seen:
            seen.add(key)
            unique.append(sol)
    return unique

def extract_custom_blends_from_form(form):
    custom_blends = []
    for j in range(2):
        name = form.get(f"custom_blend_{j}_name", "").strip()
        if not name:
            continue
        try:
            n_val = float(form.get(f"custom_blend_{j}_n", 0))
            p_val = float(form.get(f"custom_blend_{j}_p", 0))
            k_val = float(form.get(f"custom_blend_{j}_k", 0))
        except Exception:
            continue
        try:
            price = float(form.get(f"custom_blend_{j}_price", 0))
        except Exception:
            price = 0
        try:
            bag_size = float(form.get(f"custom_blend_{j}_bag_size", 50))
            if bag_size <= 0:
                bag_size = 50
        except Exception:
            bag_size = 50
        custom_blends.append({
            "name": name,
            "n": n_val,
            "p": p_val,
            "k": k_val,
            "price": price,
            "bag_size": bag_size,
            "custom": True
        })
    return custom_blends

def get_recommendations(n_req, p_req, k_req, all_blends, area, area_type, user_selected_blends=None, custom_blends=None):
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
        res = solve_single_blend(n_req, p_req, k_req, blend)
        if res:
            rounded = apply_bag_rounding(res, area, area_type, total_n, total_p, total_k)
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
        res = solve_two_blend(n_req, p_req, k_req, blend1, blend2)
        if res:
            rounded = apply_bag_rounding(res, area, area_type, total_n, total_p, total_k)
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

    all_candidates.sort(key=lambda r: r["score"])
    top3 = all_candidates[:3]

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
            res = solve_single_blend(n_req, p_req, k_req, blend)
            if res:
                rounded = apply_bag_rounding(res, area, area_type, total_n, total_p, total_k)
                if rounded:
                    rounded['excess_n'] = rounded["applied_n"] - total_n
                    rounded['excess_p'] = rounded["applied_p"] - total_p
                    rounded['excess_k'] = rounded["applied_k"] - total_k
                    rounded['custom'] = True
                    rounded['note'] = "Custom blend result (may not fully meet all requirements)."
                    custom_results.append(rounded)
        elif len(custom_blends) == 2:
            blend1, blend2 = custom_blends
            res = solve_two_blend(n_req, p_req, k_req, blend1, blend2)
            if res:
                rounded = apply_bag_rounding(res, area, area_type, total_n, total_p, total_k)
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

    if request.method == 'POST':
        try:
            n = float(request.form.get('n', 0))
            p = float(request.form.get('p', 0))
            k = float(request.form.get('k', 0))
            area = float(request.form.get('area', 0))
        except ValueError:
            error = "All numeric fields must be valid numbers greater than zero."
            return render_template('index.html', blends=blends, error=error, form_defaults=request.form)
        area_type = request.form.get('area_type')
        selected_blends = request.form.getlist('blends')

        # Handle custom blends: only process if user enabled and provided at least one valid blend
        custom_enabled = request.form.get("custom_enabled")
        custom_blends = []
        if custom_enabled:
            custom_blends = extract_custom_blends_from_form(request.form)
        # Save only valid custom blends, else save empty
        session['custom_blends'] = custom_blends

        for blend in blends:
            bname_key = blend['name'].replace(' ', '_')
            price_key = f"price_{bname_key}"
            bag_size_key = f"bag_size_{bname_key}"
            if price_key in request.form:
                try:
                    blend['price'] = float(request.form[price_key])
                except Exception:
                    pass
            if bag_size_key in request.form:
                try:
                    bag_size_val = float(request.form[bag_size_key])
                    if bag_size_val > 0:
                        blend['bag_size'] = bag_size_val
                    else:
                        blend['bag_size'] = 50
                except Exception:
                    blend['bag_size'] = 50

        session['inputs'] = {
            'n': n, 'p': p, 'k': k,
            'area': area, 'area_type': area_type,
            'selected_blends': selected_blends
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
    return render_template('index.html', blends=blends, error=error, form_defaults=form_defaults)

@app.route('/results', methods=['GET', 'POST'])
def results():
    blends = load_fertilizer_blends()
    inputs = session.get('inputs', {})
    n = float(inputs.get('n', 0))
    p = float(inputs.get('p', 0))
    k = float(inputs.get('k', 0))
    area = float(inputs.get('area', 0))
    area_type = inputs.get('area_type', 'acres')
    selected_blend_names = inputs.get('selected_blends', [])
    custom_blends = session.get('custom_blends', [])

    top3, custom_results, note = get_recommendations(
        n, p, k, blends, area, area_type, user_selected_blends=selected_blend_names, custom_blends=custom_blends
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
    return render_template('results.html', results=all_results, custom_results=custom_results, not_typical=False, fallback_used=False, best_note=note)

@app.route('/download_pdf')
def download_pdf():
    blends = load_fertilizer_blends()
    inputs = session.get('inputs', {})
    n = float(inputs.get('n', 0))
    p = float(inputs.get('p', 0))
    k = float(inputs.get('k', 0))
    area = float(inputs.get('area', 0))
    area_type = inputs.get('area_type', 'acres')
    selected_blend_names = inputs.get('selected_blends', [])
    custom_blends = session.get('custom_blends', [])

    top3, custom_results, note = get_recommendations(
        n, p, k, blends, area, area_type, user_selected_blends=selected_blend_names, custom_blends=custom_blends
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
            pdf.cell(0, 8, f"    - {blend['name']}: {blend['bags']} bags @ ${blend['cost']:.2f}", ln=True)
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
            pdf.cell(0, 8, f"    - {blend['name']}: {blend['bags']} bags @ ${blend['cost']:.2f}", ln=True)
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