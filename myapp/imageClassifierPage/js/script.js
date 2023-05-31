const dropArea = document.querySelector(".drop-area");
const dragText = dropArea.querySelector("h2");
const input = dropArea.querySelector("#input-file");
let files;


input.addEventListener("change", (e) =>{
    files = this.files;
    dropArea.classList.add("active");
    showFiles(files);
    dropArea.classList.remove("active");
});

dropArea.addEventListener("dragover", (e)=>{
    e.preventDefault();
    dropArea.classList.add("active");
    dragText.textContent = "Drop to upload files";
});

dropArea.addEventListener("dragleave", (e)=>{
    e.preventDefault();
    dropArea.classList.remove("active");
    dragText.textContent = "Drag and drop images";

});

dropArea.addEventListener("drop", (e)=>{
    e.preventDefault();
    files = e.dataTransfer.files;
    showFiles(files);
    dropArea.classList.remove("active");
    dragText.textContent = "Drag and drop images";
});

function showFiles(files){
    if(files.length === undefined){
        processFile(files);
    }else{
        for(const file of files){
            processFile(file);
        }
    }
}

function processFile(file){
    const docType = file.type;
    const validExtensions = ["image/jpeg", "image/jpg", "image/png", "image/gif"];

    if(validExtensions.includes(docType)){
        //valid file

        const fileReader = new FileReader();
        const id = `file-${Math.random().toString(32).substring(7)}`;
        fileReader.addEventListener('load', e=>{
            const fileUrl = fileReader.result;
            const image = `
                <div id= "${id}" class="file-container">
                    <img src = "${fileUrl}" alt = "${file.name}" width = 50px>
                    <div class"status">
                        <span>${file.name}</span>
                        <span class = "status-text">
                            loading...
                        </span>
                    </div>
                </div>
            `;
            const html = document.querySelector('#preview').innerHTML;
            document.querySelector('#preview').innerHTML = image + html;
        });

        fileReader.readAsDataURL(file);
        uploadFile(file, id);

    }else{
        alert('Not a valid file');
    }
}

async function uploadFile(file, id){
    const formData = new FormData();
    formData.append("file", file);
    
    try{
        const response = await fetch ("http://localhost:3000/upload", {
            method: "POST",
            body: formData,
        });
        
        const responseText = await response.text();
        console.log(responseText)
        document.querySelector(`#${id} .status-text`).innerHTML = `<span class="success"> Archivo subido correctamente... </span>`;
    }catch(error){
        document.querySelector(`#${id} .status-text`).innerHTML = `<span class="success"> Archivo no se ha subido correctamente... </span>`;
    }
}

window.addEventListener('DOMContentLoaded', (event) => {
    const imageInput = document.getElementById('input-file');
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