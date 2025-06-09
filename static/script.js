document.getElementById("npk-form").onsubmit = async function(e) {
    e.preventDefault();
    const N = document.getElementById("N").value;
    const P = document.getElementById("P").value;
    const K = document.getElementById("K").value;

    const response = await axios.post("/best-fit", {N, P, K});
    const blends = response.data;

    let formHtml = "";
    blends.forEach((blend, i) => {
        formHtml += `<input type="checkbox" name="blend" value="${i}" id="blend${i}">
                     <label for="blend${i}">${blend.label} (${blend.N}-${blend.P}-${blend.K})</label><br>`;
    });
    formHtml += `<button type="submit">Select Blend(s)</button>`;

    const blendForm = document.getElementById("blend-select-form");
    blendForm.innerHTML = formHtml;
    document.getElementById("blend-options").style.display = "block";
    document.getElementById("compute-options").style.display = "none";
    document.getElementById("results").innerHTML = "";

    blendForm.onsubmit = function(ev) {
        ev.preventDefault();
        const selected = [];
        blends.forEach((blend, i) => {
            if (document.getElementById(`blend${i}`).checked) selected.push(blend);
        });
        if (selected.length === 0) {
            alert("Please select at least one blend.");
            return;
        }
        // Store selected blends for compute step
        window.selectedBlends = selected;
        document.getElementById("compute-options").style.display = "block";
    };
};

document.getElementById("compute-form").onsubmit = async function(e) {
    e.preventDefault();
    const price = document.getElementById("price").value;
    const bag_weight = document.getElementById("bag_weight").value;
    const N = document.getElementById("N").value;
    const P = document.getElementById("P").value;
    const K = document.getElementById("K").value;
    const blends = window.selectedBlends;

    const response = await axios.post("/compute", {
        N, P, K, price, bag_weight, blends
    });
    const results = response.data;

    let html = `<h2>Results (sorted by lowest excess):</h2><table border="1">
        <tr><th>Blend</th><th>Bags Required</th><th>Total Cost</th><th>Excess (lbs)</th>
            <th>Provided N</th><th>Provided P</th><th>Provided K</th></tr>`;
    results.forEach(result => {
        html += `<tr>
            <td>${result.label}</td>
            <td>${result.bags_required}</td>
            <td>$${result.total_cost.toFixed(2)}</td>
            <td>${result.excess}</td>
            <td>${result.provided_N}</td>
            <td>${result.provided_P}</td>
            <td>${result.provided_K}</td>
        </tr>`;
    });
    html += `</table>`;
    document.getElementById("results").innerHTML = html;
};