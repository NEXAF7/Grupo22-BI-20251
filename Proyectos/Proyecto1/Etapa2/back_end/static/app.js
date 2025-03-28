document
  .getElementById("predict-form")
  .addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const result = await response.json();
    document.getElementById("predict-result").textContent = JSON.stringify(
      result,
      null,
      2
    );
  });

document
  .getElementById("retrain-form")
  .addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const response = await fetch("/retrain", {
      method: "POST",
      body: formData,
    });
    const result = await response.json();
    document.getElementById("retrain-result").textContent = JSON.stringify(
      result,
      null,
      2
    );
  });
