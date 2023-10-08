const express = require('express');
const multer = require('multer');
const app = express();
const port = 3000;

const storage = multer.diskStorage({
    destination: 'uploads/', // Specify the directory where you want to save the photo
    filename: (req, file, callback) => {
        callback(null, file.originalname);
    },
});

const upload = multer({ storage: storage });

app.post('/upload', upload.single('photo'), (req, res) => {
    const imageUrl = `/uploads/${req.file.originalname}`;
    res.send(imageUrl);
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
