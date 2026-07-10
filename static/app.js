document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const datasetGallery = document.getElementById('dataset-gallery');
    const datasetSearch = document.getElementById('dataset-search');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const btnClearImage = document.getElementById('btn-clear-image');
    const activeFilename = document.getElementById('active-filename');
    const btnRunEval = document.getElementById('btn-run-eval');
    
    // Modal Elements
    const evalModal = document.getElementById('eval-modal');
    const closeModal = document.querySelector('.close-modal');
    const evalLogs = document.getElementById('eval-logs');
    
    // Tab Elements
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Image element cache to refresh them
    const imgResults = document.getElementById('img-results');
    const imgGrid = document.getElementById('img-grid');

    // Scenarios IDs list
    const scenarios = ['S1_LR_Besar', 'S2_LR_Kecil', 'S5_Epoch_Tinggi'];

    let allDatasetFiles = [];

    // Initialize Page
    fetchDatasetList();
    setupDropZone();
    setupTabSwitching();
    setupEvaluationModal();

    // 1. Fetch & Render Dataset
    async function fetchDatasetList() {
        try {
            const response = await fetch('/api/dataset');
            const data = await response.json();
            if (data.success) {
                allDatasetFiles = data.files;
                renderGallery(allDatasetFiles);
            } else {
                datasetGallery.innerHTML = `<div class="error-text">Failed to load dataset: ${data.error}</div>`;
            }
        } catch (err) {
            datasetGallery.innerHTML = `<div class="error-text">Failed to fetch dataset.</div>`;
        }
    }

    function renderGallery(files) {
        if (files.length === 0) {
            datasetGallery.innerHTML = `<div class="loading-spinner">No images found.</div>`;
            return;
        }

        datasetGallery.innerHTML = '';
        files.forEach(filename => {
            const card = document.createElement('div');
            card.className = 'image-thumb-card';
            card.dataset.filename = filename;
            
            // Image source using direct static server
            card.innerHTML = `
                <img src="/api/dataset/image/${filename}" alt="${filename}" loading="lazy">
                <span class="image-name-tag">${filename}</span>
            `;
            
            card.addEventListener('click', () => {
                selectDatasetImage(filename, card);
            });
            
            datasetGallery.appendChild(card);
        });
    }

    // Filter gallery on search input
    datasetSearch.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        const filtered = allDatasetFiles.filter(f => f.toLowerCase().includes(query));
        renderGallery(filtered);
    });

    // 2. Select Image from Sidebar
    function selectDatasetImage(filename, cardElement) {
        // Highlight active thumbnail
        document.querySelectorAll('.image-thumb-card').forEach(el => el.classList.remove('active'));
        if (cardElement) {
            cardElement.classList.add('active');
        }

        // Show image preview
        imagePreview.src = `/api/dataset/image/${filename}`;
        previewContainer.style.display = 'flex';
        dropZone.style.display = 'none';
        
        activeFilename.textContent = filename;

        // Trigger prediction
        runPredictionForDataset(filename);
    }

    // 3. Upload Functions
    function setupDropZone() {
        dropZone.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleUploadedFile(e.target.files[0]);
            }
        });

        // Drag events
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
            }, false);
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                handleUploadedFile(files[0]);
            }
        });

        // Clear image button
        btnClearImage.addEventListener('click', () => {
            fileInput.value = '';
            imagePreview.src = '';
            previewContainer.style.display = 'none';
            dropZone.style.display = 'flex';
            activeFilename.textContent = 'No File Selected';
            document.querySelectorAll('.image-thumb-card').forEach(el => el.classList.remove('active'));
            resetGauges();
        });
    }

    function handleUploadedFile(file) {
        // Show local preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.style.display = 'flex';
            dropZone.style.display = 'none';
        };
        reader.readAsDataURL(file);

        activeFilename.textContent = file.name;
        // Deselect dataset active states
        document.querySelectorAll('.image-thumb-card').forEach(el => el.classList.remove('active'));

        // Run prediction
        runPredictionForUpload(file);
    }

    // 4. API Prediction Calls
    async function runPredictionForDataset(filename) {
        showLoadingState();
        try {
            const formData = new FormData();
            formData.append('source', 'dataset');
            formData.append('filename', filename);

            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.success) {
                updateResults(data.predictions);
            } else {
                showErrorState(data.error);
            }
        } catch (err) {
            showErrorState(err.message);
        }
    }

    async function runPredictionForUpload(file) {
        showLoadingState();
        try {
            const formData = new FormData();
            formData.append('source', 'upload');
            formData.append('image', file);

            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.success) {
                updateResults(data.predictions);
            } else {
                showErrorState(data.error);
            }
        } catch (err) {
            showErrorState(err.message);
        }
    }

    // 5. Update GUI States
    function showLoadingState() {
        scenarios.forEach(id => {
            const textPct = document.getElementById(`val-${id.split('_')[0]}-pct`);
            textPct.textContent = '...';
            const textClass = document.getElementById(`val-${id.split('_')[0]}-class`);
            textClass.className = 'prediction-status safe';
            textClass.textContent = 'Predicting';
        });
    }

    function showErrorState(errMsg) {
        scenarios.forEach(id => {
            const textPct = document.getElementById(`val-${id.split('_')[0]}-pct`);
            textPct.textContent = 'Err';
            const textClass = document.getElementById(`val-${id.split('_')[0]}-class`);
            textClass.className = 'prediction-status safe';
            textClass.textContent = 'Failed';
        });
        alert("Error running inference: " + errMsg);
    }

    function updateResults(predictions) {
        scenarios.forEach(id => {
            const alias = id.split('_')[0]; // S1, S2, S5
            const data = predictions[id];
            
            const pctText = document.getElementById(`val-${alias}-pct`);
            const classText = document.getElementById(`val-${alias}-class`);
            const latencyText = document.getElementById(`val-${alias}-latency`);
            const progressCircle = document.querySelector(`.${alias.toLowerCase()}-progress`);
            
            if (data.error) {
                pctText.textContent = 'Err';
                classText.textContent = 'Error';
                classText.className = 'prediction-status safe';
                latencyText.textContent = '-';
                return;
            }

            const prob = data.probability;
            const pct = Math.round(prob * 100);
            
            // Update gauge percentage text
            pctText.textContent = `${pct}%`;
            
            // Update SVG circle dashoffset
            // Circumference is 2 * PI * r = 2 * 3.14159 * 40 = 251.2
            const offset = 251.2 - (251.2 * prob);
            progressCircle.style.strokeDashoffset = offset;
            
            // Update latency
            latencyText.textContent = `${data.latency_ms} ms`;
            
            // Update classification status badge
            classText.textContent = pct >= 50 ? 'Wildfire Detected' : 'Safe / No Fire';
            if (pct >= 50) {
                classText.className = 'prediction-status wildfire';
            } else {
                classText.className = 'prediction-status safe';
            }
        });
    }

    function resetGauges() {
        scenarios.forEach(id => {
            const alias = id.split('_')[0];
            document.getElementById(`val-${alias}-pct`).textContent = '0%';
            document.querySelector(`.${alias.toLowerCase()}-progress`).style.strokeDashoffset = 251.2;
            document.getElementById(`val-${alias}-class`).textContent = 'Ready';
            document.getElementById(`val-${alias}-class`).className = 'prediction-status safe';
            document.getElementById(`val-${alias}-latency`).textContent = '-';
        });
    }

    // 6. Tab switching logic
    function setupTabSwitching() {
        tabButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                // Remove active classes
                tabButtons.forEach(b => b.classList.remove('active'));
                tabContents.forEach(c => c.style.display = 'none');
                
                // Set active tab
                btn.classList.add('active');
                const target = btn.dataset.target;
                document.getElementById(target).style.display = 'block';
            });
        });
    }

    // 7. Full Evaluation Script Execution
    function setupEvaluationModal() {
        btnRunEval.addEventListener('click', async () => {
            evalModal.classList.add('show');
            evalLogs.textContent = 'Initializing evaluation...\nRunning `evaluate.py` via Python backend...\n\n';
            
            try {
                const response = await fetch('/api/evaluate', {
                    method: 'POST'
                });
                const data = await response.json();
                
                if (data.success) {
                    evalLogs.textContent += data.output;
                    evalLogs.textContent += '\n\nEvaluation finished successfully! Updating plots...';
                    
                    // Force refresh charts images by appending timestamp to URL
                    const t = new Date().getTime();
                    imgResults.src = `/static/evaluation_results.png?t=${t}`;
                    imgGrid.src = `/static/predictions_grid.png?t=${t}`;
                } else {
                    evalLogs.textContent += `\nError executing script:\n${data.error}`;
                }
            } catch (err) {
                evalLogs.textContent += `\nConnection error: ${err.message}`;
            }
        });
        
        closeModal.addEventListener('click', () => {
            evalModal.classList.remove('show');
        });

        // Close on overlay click
        evalModal.addEventListener('click', (e) => {
            if (e.target === evalModal) {
                evalModal.classList.remove('show');
            }
        });
    }
});
