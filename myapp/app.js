const express = require('express');
const app = express();
const port = 3000
const fileUpload = require('express-fileupload');
const cors = require('cors');
/*const { PythonShell } = require('python-shell');

let options = {
  scriptPath: 'C:/Users/Sebas/Documents/code/simulacionGrafica/featureImages', // Ruta al directorio que contiene el archivo Python
  args: ["asti", 23],
};

PythonShell.run('main.py', options, (err, res) => {
  if (err) console.log(err);
  if (res) console.log(res);
});*/

app.use(fileUpload());
app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
    res.send("Server running!!!!");
});

app.post('/upload', (req, res) => {
  console.log(req.files.file.name);

  let imageFile = req.files.file;

  imageFile.mv(`imageClassifierPage/images/${req.files.file.name}`, err => {
    if (err) {
     return res.status(500).send(err);
    }
  
    // res.json({ file: `fotos/${req.body.filename}.jpg` });
    // console.log(res.json);
   });
});

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})
