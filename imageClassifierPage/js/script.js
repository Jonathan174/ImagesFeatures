window.addEventListener('DOMContentLoaded', (event) => {
    const imageInput = document.getElementById('imageInput');
    const selectedImage = document.getElementById('selectedImage');
    const defaultImagesContainer = document.getElementById('defaultImages');
  
    imageInput.addEventListener('change', (event) => {
      const file = event.target.files[0];
      const reader = new FileReader();
  
      reader.onload = function (event) {
        selectedImage.src = event.target.result;
        defaultImagesContainer.style.display = 'flex'; // Mostramos las imágenes predefinidas
      };
  
      if (file) {
        reader.readAsDataURL(file);
      }
    });
  });
  