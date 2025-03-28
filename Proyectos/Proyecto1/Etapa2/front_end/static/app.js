// Manejo del formulario de predicción
document
  .getElementById("predict-form")
  .addEventListener("submit", async (e) => {
    e.preventDefault(); // Evita recargar la página al enviar el formulario

    const formData = new FormData(e.target); // Captura datos del formulario
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    }); // Llama al endpoint predict del backend

    const result = await response.json(); // Convierte la respuesta a JSON
    document.getElementById("predict-result").textContent = JSON.stringify(
      result,
      null,
      2
    );
  });

// Manejo del formulario de reentrenamiento
document
  .getElementById("retrain-form")
  .addEventListener("submit", async (e) => {
    e.preventDefault(); // Evita recargar la página al enviar el formulario

    const formData = new FormData(e.target); // Captura datos del formulario
    const response = await fetch("/retrain", {
      method: "POST",
      body: formData,
    }); // Llama al endpoint retrain del backend

    const result = await response.json(); // Convierte la respuesta a JSON
    document.getElementById("retrain-result").textContent = JSON.stringify(
      result,
      null,
      2
    );
  });
