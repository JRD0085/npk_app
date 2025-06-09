import os
import csv
import itertools
import math
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mail import Mail, Message
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Flask-Mail configuration (update EMAIL_USER and EMAIL_PASS in your environment)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('EMAIL_USER', 'your_email@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('EMAIL_PASS', 'your_app_password')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('EMAIL_USER', 'your_email@gmail.com')
mail = Mail(app)

FEEDBACK_LIMIT = 1000
FEEDBACK_COUNTER_FILE = "feedback_count.txt"

def can_accept_feedback():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    count = 0
    if os.path.exists(FEEDBACK_COUNTER_FILE):
        with open(FEEDBACK_COUNTER_FILE, "r") as f:
            lines = f.readlines()
            if lines and lines[0].startswith(today_str):
                count = int(lines[0].strip().split(",")[1])
            else:
                count = 0
    return count < FEEDBACK_LIMIT

def increment_feedback_counter():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    count = 0
    if os.path.exists(FEEDBACK_COUNTER_FILE):
        with open(FEEDBACK_COUNTER_FILE, "r") as f:
            lines = f.readlines()
            if lines and lines[0].startswith(today_str):
                count = int(lines[0].strip().split(",")[1])
    count += 1
    with open(FEEDBACK_COUNTER_FILE, "w") as f:
        f.write(f"{today_str},{count}\n")

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
    reqs = []
    for nutrient, perc in zip([n_req, p_req, k_req], [blend["n"], blend["p"], blend["k"]]):
        if perc > 0:
            reqs.append(nutrient / (perc / 100))
        else:
            reqs.append(0 if nutrient <= 0 else float('inf'))
    lbs_per_acre = max(reqs)
    if math.isinf(lbs_per_acre) or lbs_per_acre <= 0:
        return None
    if (blend["n"] <= 0 and n_req > 0) or (blend["p"] <= 0 and p_req > 0) or (blend["k"] <= 0 and k_req > 0):
        return None
    applied_n = lbs_per_acre * (blend["n"] / 100)
    applied_p = lbs_per_acre * (blend["p"] / 100)
    applied_k = lbs_per_acre * (blend["k"] / 100)
    excess_n = max(0, applied_n - n_req)
    excess_p = max(0, applied_p - p_req)
    excess_k = max(0, applied_k - k_req)
    return {
        "type": "single",
        "blends": [(blend, lbs_per_acre)],
        "excess_n": excess_n,
        "excess_p": excess_p,
        "excess_k": excess_k,
        "names": [blend["name"]],
    }

def solve_two_blend(n_req, p_req, k_req, blend1, blend2):
    if blend1["name"] == blend2["name"]:
        return None
    A = np.array([
        [blend1["n"]/100, blend2["n"]/100],
        [blend1["p"]/100, blend2["p"]/100],
        [blend1["k"]/100, blend2["k"]/100]
    ])
    req = np.array([n_req, p_req, k_req])
    valid_rows = [i for i in range(3) if A[i,0] > 0 or A[i,1] > 0]
    if not valid_rows:
        return None
    A_reduced = A[valid_rows, :]
    req_reduced = req[valid_rows]
    try:
        x, residuals, rank, s = np.linalg.lstsq(A_reduced, req_reduced, rcond=None)
        x = np.maximum(x, 0)
        if any(xi < 0.01 for xi in x):
            return None
        applied = A @ x
        if any(applied[i] < req[i] and req[i] > 0 for i in range(3)):
            return None
        excess = applied - req
        excess_n = max(0, excess[0])
        excess_p = max(0, excess[1])
        excess_k = max(0, excess[2])
        return {
            "type": "combo",
            "blends": [(blend1, x[0]), (blend2, x[1])],
            "excess_n": excess_n,
            "excess_p": excess_p,
            "excess_k": excess_k,
            "names": [blend1["name"], blend2["name"]],
        }
    except Exception:
        return None

def best_matches(n_req, p_req, k_req, blends, area, area_type):
    results = []
    if area_type == 'acres':
        total_area = area
    else:
        total_area = area / 43560

    # Only recommend the cheapest single blend if all nutrients are equal
    if n_req == p_req == k_req:
        cheapest = None
        for blend in blends:
            res = solve_single_blend(n_req, p_req, k_req, blend)
            if res is not None:
                lbs_needed = res["blends"][0][1] * total_area
                bags_needed = math.ceil(lbs_needed / blend["bag_size"])
                cost = bags_needed * blend["price"]
                res.update({
                    "total_cost": cost,
                    "bags": [(blend["name"], bags_needed)],
                    "applied_n": res["blends"][0][1] * total_area * (blend["n"] / 100),
                    "applied_p": res["blends"][0][1] * total_area * (blend["p"] / 100),
                    "applied_k": res["blends"][0][1] * total_area * (blend["k"] / 100),
                    "details": [(blend["name"], res["blends"][0][1])],
                })
                if cheapest is None or cost < cheapest["total_cost"]:
                    cheapest = res
        if cheapest:
            cheapest["is_most_cost_effective"] = True
            cheapest["is_most_precise"] = False
            cheapest["total_excess"] = cheapest["excess_n"] + cheapest["excess_p"] + cheapest["excess_k"]
            return [cheapest]
        else:
            return []
    # Otherwise, allow combos as before
    for blend in blends:
        res = solve_single_blend(n_req, p_req, k_req, blend)
        if res is not None:
            lbs_needed = res["blends"][0][1] * total_area
            bags_needed = math.ceil(lbs_needed / blend["bag_size"])
            cost = bags_needed * blend["price"]
            res.update({
                "total_cost": cost,
                "bags": [(blend["name"], bags_needed)],
                "applied_n": res["blends"][0][1] * total_area * (blend["n"] / 100),
                "applied_p": res["blends"][0][1] * total_area * (blend["p"] / 100),
                "applied_k": res["blends"][0][1] * total_area * (blend["k"] / 100),
                "details": [(blend["name"], res["blends"][0][1])],
            })
            results.append(res)

    for blend1, blend2 in itertools.combinations(blends, 2):
        res = solve_two_blend(n_req, p_req, k_req, blend1, blend2)
        if res is not None:
            lbs1 = res["blends"][0][1] * total_area
            lbs2 = res["blends"][1][1] * total_area
            bags1 = math.ceil(lbs1 / blend1["bag_size"])
            bags2 = math.ceil(lbs2 / blend2["bag_size"])
            cost = bags1 * blend1["price"] + bags2 * blend2["price"]
            res.update({
                "total_cost": cost,
                "bags": [(blend1["name"], bags1), (blend2["name"], bags2)],
                "applied_n": lbs1 * (blend1["n"] / 100) + lbs2 * (blend2["n"] / 100),
                "applied_p": lbs1 * (blend1["p"] / 100) + lbs2 * (blend2["p"] / 100),
                "applied_k": lbs1 * (blend1["k"] / 100) + lbs2 * (blend2["k"] / 100),
                "details": [(blend1["name"], res["blends"][0][1]), (blend2["name"], res["blends"][1][1])],
            })
            results.append(res)

    if not results:
        return []
    for r in results:
        r["total_excess"] = r["excess_n"] + r["excess_p"] + r["excess_k"]
    min_cost = min(r["total_cost"] for r in results)
    min_excess = min(r["total_excess"] for r in results)
    for r in results:
        r["is_most_cost_effective"] = math.isclose(r["total_cost"], min_cost)
        r["is_most_precise"] = math.isclose(r["total_excess"], min_excess)
    results.sort(key=lambda r: (r["total_cost"], r["total_excess"]))
    return results[:3]

@app.route('/', methods=['GET', 'POST'])
def index():
    blends = load_fertilizer_blends()
    error = None
    if request.method == 'POST':
        try:
            n = float(request.form.get('n', 0))
            p = float(request.form.get('p', 0))
            k = float(request.form.get('k', 0))
            area = float(request.form.get('area', 0))
        except ValueError:
            error = "All numeric fields must be valid numbers greater than zero."
            return render_template('index.html', blends=blends, error=error)
        area_type = request.form.get('area_type')
        selected_blends = request.form.getlist('blends')
        if not selected_blends or n < 0 or p < 0 or k < 0 or area <= 0:
            error = "Please ensure all numeric fields are filled with valid positive values and at least one blend is selected."
            return render_template('index.html', blends=blends, error=error)
        session['inputs'] = {
            'n': n, 'p': p, 'k': k,
            'area': area, 'area_type': area_type,
            'selected_blends': selected_blends
        }
        return redirect(url_for('recommendations'))
    return render_template('index.html', blends=blends, error=error)

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    blends = load_fertilizer_blends()
    inputs = session.get('inputs', {})
    n = inputs.get('n', 0)
    p = inputs.get('p', 0)
    k = inputs.get('k', 0)
    area = inputs.get('area', 0)
    area_type = inputs.get('area_type', 'acres')
    selected_blend_names = inputs.get('selected_blends', [])

    selected_blends = [b for b in blends if b["name"] in selected_blend_names]
    matches = best_matches(n, p, k, selected_blends, area, area_type)
    all_matches = best_matches(n, p, k, blends, area, area_type)
    best_overall = all_matches[0] if all_matches else None

    impossible = not matches

    # Only show the green best overall box if the user's matches do NOT contain the best overall solution.
    def match_key(m):
        return set(m["names"])
    matches_keys = [match_key(m) for m in matches]
    best_key = match_key(best_overall) if best_overall else None
    show_best_overall = best_overall and best_key not in matches_keys

    return render_template(
        'recommendations.html',
        matches=matches,
        best_overall=best_overall if show_best_overall else None,
        impossible=impossible,
        inputs=inputs
    )

@app.route('/customize', methods=['GET', 'POST'])
def customize():
    if request.method == 'POST':
        recommendations = []
        selected = request.form.getlist('select')
        rec_count = int(request.form.get('rec_count', 0))
        for i in range(rec_count):
            rec_prefix = f"rec_{i}_"
            rec_selected = (str(i) in selected)
            blend_count = int(request.form.get(f"{rec_prefix}blend_count", 1))
            blends = []
            for j in range(blend_count):
                try:
                    name = request.form[f"{rec_prefix}blend_{j}_name"]
                    n = float(request.form[f"{rec_prefix}blend_{j}_n"])
                    p = float(request.form[f"{rec_prefix}blend_{j}_p"])
                    k = float(request.form[f"{rec_prefix}blend_{j}_k"])
                    price = float(request.form[f"{rec_prefix}blend_{j}_price"])
                    bag_size = float(request.form[f"{rec_prefix}blend_{j}_bag_size"])
                    lbs_per_acre = float(request.form[f"{rec_prefix}blend_{j}_lbs_per_acre"])
                except:
                    continue
                blends.append({
                    "name": name, "n": n, "p": p, "k": k,
                    "price": price, "bag_size": bag_size,
                    "lbs_per_acre": lbs_per_acre,
                })
            recommendations.append({
                "selected": rec_selected,
                "blends": blends,
                "type": request.form[f"{rec_prefix}type"],
                "note": request.form.get(f"{rec_prefix}note", ""),
            })
        custom_recs = []
        custom_count = int(request.form.get('custom_count', 0))
        for i in range(custom_count):
            prefix = f"custom_{i}_"
            if not request.form.get(f"{prefix}enabled"): continue
            blend_count = int(request.form.get(f"{prefix}blend_count", 1))
            blends = []
            for j in range(blend_count):
                try:
                    name = request.form[f"{prefix}blend_{j}_name"]
                    n = float(request.form[f"{prefix}blend_{j}_n"])
                    p = float(request.form[f"{prefix}blend_{j}_p"])
                    k = float(request.form[f"{prefix}blend_{j}_k"])
                    price = float(request.form[f"{prefix}blend_{j}_price"])
                    bag_size = float(request.form[f"{prefix}blend_{j}_bag_size"])
                    # lbs_per_acre removed for custom blends
                    # lbs_per_acre = float(request.form[f"{prefix}blend_{j}_lbs_per_acre"])
                except:
                    continue
                blends.append({
                    "name": name, "n": n, "p": p, "k": k,
                    "price": price, "bag_size": bag_size,
                    # "lbs_per_acre": lbs_per_acre,
                })
            custom_recs.append({
                "selected": True,
                "blends": blends,
                "type": "custom",
                "note": "",
            })

        session['customize'] = {
            "recommendations": recommendations,
            "custom_recs": custom_recs,
        }
        return redirect(url_for('results'))

    blends = load_fertilizer_blends()
    inputs = session.get('inputs', {})
    n = inputs.get('n', 0)
    p = inputs.get('p', 0)
    k = inputs.get('k', 0)
    area = inputs.get('area', 0)
    area_type = inputs.get('area_type', 'acres')
    selected_blend_names = inputs.get('selected_blends', [])

    selected_blends = [b for b in blends if b["name"] in selected_blend_names]
    matches = best_matches(n, p, k, selected_blends, area, area_type)
    all_matches = best_matches(n, p, k, blends, area, area_type)
    best_overall = all_matches[0] if all_matches else None

    def match_key(m):
        return set(m["names"])
    matches_keys = [match_key(m) for m in matches]
    overall_key = match_key(best_overall) if best_overall else None
    show_best_overall = best_overall and overall_key not in matches_keys

    recommendations = []
    for m in matches:
        blends_data = []
        for blend, lbs_per_acre in m["blends"]:
            blends_data.append({
                "name": blend["name"], "n": blend["n"], "p": blend["p"], "k": blend["k"],
                "price": blend["price"], "bag_size": blend["bag_size"], "lbs_per_acre": lbs_per_acre
            })
        recommendations.append({
            "type": m["type"],
            "note": "",
            "blends": blends_data,
            "selected": True,
        })
    if show_best_overall:
        blends_data = []
        for blend, lbs_per_acre in best_overall["blends"]:
            blends_data.append({
                "name": blend["name"], "n": blend["n"], "p": blend["p"], "k": blend["k"],
                "price": blend["price"], "bag_size": blend["bag_size"], "lbs_per_acre": lbs_per_acre
            })
        recommendations.append({
            "type": best_overall["type"],
            "note": "This is the best overall match, but you did not select it as available to you.",
            "blends": blends_data,
            "selected": False,
        })

    return render_template(
        'customize.html',
        recommendations=recommendations,
        max_custom=3
    )

@app.route('/results')
def results():
    inputs = session.get('inputs', {})
    customize = session.get('customize', {})
    n_req = float(inputs.get('n', 0))
    p_req = float(inputs.get('p', 0))
    k_req = float(inputs.get('k', 0))
    area = float(inputs.get('area', 0))
    area_type = inputs.get('area_type', 'acres')
    if area_type == 'acres':
        total_area = area
    else:
        total_area = area / 43560

    recs = customize.get('recommendations', [])
    selected_recs = [r for r in recs if r.get('selected')]
    custom_recs = customize.get('custom_recs', [])
    all_recs = selected_recs + custom_recs

    results = []
    for idx, rec in enumerate(all_recs):
        blends = rec['blends']
        blend_calcs = []
        total_cost = 0
        total_applied_n = 0
        total_applied_p = 0
        total_applied_k = 0
        for blend in blends:
            lbs_per_acre = float(blend['lbs_per_acre']) if 'lbs_per_acre' in blend else 0
            bag_size = float(blend['bag_size'])
            price = float(blend['price'])
            lbs_total = lbs_per_acre * total_area
            bags = int(-(-lbs_total // bag_size)) if lbs_per_acre > 0 else 0
            actual_supplied_lbs = bags * bag_size if lbs_per_acre > 0 else 0
            cost = bags * price if lbs_per_acre > 0 else 0
            applied_n = actual_supplied_lbs * (float(blend['n']) / 100)
            applied_p = actual_supplied_lbs * (float(blend['p']) / 100)
            applied_k = actual_supplied_lbs * (float(blend['k']) / 100)
            blend_calcs.append({
                'name': blend['name'],
                'lbs_per_acre': lbs_per_acre,
                'bag_size': bag_size,
                'bags': bags,
                'price': price,
                'cost': cost,
                'applied_n': applied_n,
                'applied_p': applied_p,
                'applied_k': applied_k,
                'lbs_total': actual_supplied_lbs,
            })
            total_cost += cost
            total_applied_n += applied_n
            total_applied_p += applied_p
            total_applied_k += applied_k
        results.append({
            'type': rec['type'] if 'type' in rec else 'custom',
            'note': rec.get('note', ''),
            'blends': blend_calcs,
            'total_cost': total_cost,
            'total_applied_n': total_applied_n,
            'total_applied_p': total_applied_p,
            'total_applied_k': total_applied_k,
            'excess_n': max(0, total_applied_n - n_req * total_area),
            'excess_p': max(0, total_applied_p - p_req * total_area),
            'excess_k': max(0, total_applied_k - k_req * total_area),
        })

    # Add best overall if not among selected
    blends = load_fertilizer_blends()
    all_matches = best_matches(n_req, p_req, k_req, blends, area, area_type)
    best_overall = all_matches[0] if all_matches else None

    def match_key(blends_list):
        return set(b['name'] for b in blends_list)

    selected_keys = [match_key([b for b in rec['blends']]) for rec in results]
    best_key = match_key([b for b, _ in best_overall['blends']]) if best_overall else None
    show_best_overall = best_overall and best_key not in selected_keys

    if results:
        min_cost = min(r["total_cost"] for r in results)
        min_excess = min(r["excess_n"] + r["excess_p"] + r["excess_k"] for r in results)
        if n_req == p_req == k_req:
            for r in results:
                r["is_most_cost_effective"] = math.isclose(r["total_cost"], min_cost)
                r["is_most_precise"] = False
        else:
            for r in results:
                r["is_most_cost_effective"] = math.isclose(r["total_cost"], min_cost)
                r["is_most_precise"] = math.isclose(r["excess_n"] + r["excess_p"] + r["excess_k"], min_excess)

    best_overall_result = None
    if show_best_overall:
        blend_calcs = []
        total_cost = 0
        total_applied_n = 0
        total_applied_p = 0
        total_applied_k = 0
        for blend, lbs_per_acre in best_overall['blends']:
            bag_size = blend['bag_size']
            price = blend['price']
            lbs_total = lbs_per_acre * total_area
            bags = int(-(-lbs_total // bag_size))
            actual_supplied_lbs = bags * bag_size
            cost = bags * price
            applied_n = actual_supplied_lbs * (blend['n'] / 100)
            applied_p = actual_supplied_lbs * (blend['p'] / 100)
            applied_k = actual_supplied_lbs * (blend['k'] / 100)
            blend_calcs.append({
                'name': blend['name'],
                'lbs_per_acre': lbs_per_acre,
                'bag_size': bag_size,
                'bags': bags,
                'price': price,
                'cost': cost,
                'applied_n': applied_n,
                'applied_p': applied_p,
                'applied_k': applied_k,
                'lbs_total': actual_supplied_lbs,
            })
            total_cost += cost
            total_applied_n += applied_n
            total_applied_p += applied_p
            total_applied_k += applied_k
        best_overall_result = {
            'type': best_overall['type'],
            'note': "This is the best overall match, but you did not select it as available to you.",
            'blends': blend_calcs,
            'total_cost': total_cost,
            'total_applied_n': total_applied_n,
            'total_applied_p': total_applied_p,
            'total_applied_k': total_applied_k,
            'excess_n': max(0, total_applied_n - n_req * total_area),
            'excess_p': max(0, total_applied_p - p_req * total_area),
            'excess_k': max(0, total_applied_k - k_req * total_area),
            'is_most_cost_effective': True,
            'is_most_precise': False
        }

    return render_template(
        'results.html',
        results=results,
        best_overall_result=best_overall_result,
        n_req=n_req,
        p_req=p_req,
        k_req=k_req,
        total_area=total_area,
        area_type=area_type,
    )

@app.route('/share', methods=['GET', 'POST'])
def share():
    feedback_sent = False
    feedback_limit_reached = False
    if request.method == 'POST':
        if can_accept_feedback():
            email = request.form.get('email')
            feedback = request.form.get('feedback')
            msg = Message(
                subject="Feedback from Davidson Fertilizer Calculator",
                recipients=["jrd0085@auburn.edu"],
                body=f"From: {email or 'Anonymous'}\n\n{feedback}"
            )
            try:
                mail.send(msg)
                increment_feedback_counter()
                feedback_sent = True
            except Exception as e:
                feedback_sent = False
                print("Email send failed:", e)
        else:
            feedback_limit_reached = True
    return render_template('share.html', feedback_sent=feedback_sent, feedback_limit_reached=feedback_limit_reached)

if __name__ == '__main__':
    app.run(debug=True)