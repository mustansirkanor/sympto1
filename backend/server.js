const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const cors = require('cors');
const fs = require('fs');

const app = express();
const PORT = 5000;

// Middleware
app.use(cors());
app.use(express.json());

// File upload setup
const upload = multer({ dest: 'uploads/' });

// ROOT ROUTE - Add this!
app.get('/', (req, res) => {
    res.json({ 
        message: 'Disease Prediction Backend API',
        status: 'running',
        endpoints: {
            health: '/api/health',
            predict: 'POST /api/predict'
        }
    });
});

// Health check
app.get('/api/health', (req, res) => {
    res.json({ status: 'OK', message: 'Backend is healthy' });
});

// Predict endpoint
app.post('/api/predict', upload.single('image'), async (req, res) => {
    console.log('Received request');
    console.log('File:', req.file);

    if (!req.file) {
        return res.status(400).json({ 
            success: false,
            error: 'No image uploaded' 
        });
    }

    try {
        // Send to FastAPI
        const formData = new FormData();
        formData.append('file', fs.createReadStream(req.file.path));

        const response = await axios.post('http://localhost:8000/predict', formData, {
            headers: formData.getHeaders(),
            timeout: 30000
        });

        // Delete temp file
        fs.unlinkSync(req.file.path);

        res.json({
            success: true,
            data: response.data
        });

    } catch (error) {
        // Delete temp file
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }
        
        console.error('Error:', error.message);
        res.status(500).json({ 
            success: false,
            error: error.message 
        });
    }
});

app.listen(PORT, () => {
    console.log(`Node.js backend running on http://localhost:${PORT}`);
    console.log(`Endpoints available:`);
    console.log(`  - GET  http://localhost:${PORT}/`);
    console.log(`  - GET  http://localhost:${PORT}/api/health`);
    console.log(`  - POST http://localhost:${PORT}/api/predict`);
});
