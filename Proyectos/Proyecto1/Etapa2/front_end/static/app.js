// Manejo del formulario para cargar el modelo
document
  .getElementById("load-model-form")
  .addEventListener("submit", async (e) => {
    e.preventDefault(); // Evitar recarga de la página
    const formData = new FormData(e.target);

    try {
      const response = await fetch("/load_model", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      document.getElementById("load-model-result").textContent = JSON.stringify(
        result,
        null,
        2
      );
    } catch (error) {
      console.error("Error al cargar el modelo:", error);
      document.getElementById("load-model-result").textContent =
        "Ocurrió un error al cargar el modelo.";
    }
  });

// Manejo del formulario de predicción
document
  .getElementById("predict-form")
  .addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);

    try {
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
    } catch (error) {
      console.error("Error en la predicción:", error);
      document.getElementById("predict-result").textContent =
        "Ocurrió un error en la predicción.";
    }
  });

// Manejo del formulario de reentrenamiento
document
  .getElementById("retrain-form")
  .addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);

    try {
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
    } catch (error) {
      console.error("Error en el reentrenamiento:", error);
      document.getElementById("retrain-result").textContent =
        "Ocurrió un error en el reentrenamiento.";
    }
  });
