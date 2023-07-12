let droppedFiles = [];

    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    const startOCRButton = document.getElementById('start-ocr-button');
    const predictionResult = document.getElementById('prediction-result');
    const browseButton = document.getElementById('browse-button');
    const paragraph = document.getElementById('paragraph');

    // Set initial dimensions based on CSS variables
    function setInitialDimensions() {
      const dropZoneWidth = getComputedStyle(document.documentElement).getPropertyValue('--drop-zone-width');
      const dropZoneHeight = getComputedStyle(document.documentElement).getPropertyValue('--drop-zone-height');

      dropZone.style.width = dropZoneWidth;
      dropZone.style.height = dropZoneHeight;
    }

    // Update file list
    function updateFileList() {
      fileList.innerHTML = '';
      for (let i = 0; i < droppedFiles.length; i++) {
        const file = droppedFiles[i];

        const fileItem = document.createElement('div');
        fileItem.classList.add('file-item');

        const deleteButton = document.createElement('span');
        deleteButton.classList.add('delete-button-small');
        deleteButton.innerHTML = '&times;';
        deleteButton.addEventListener('click', () => deleteFile(file));

        const image = document.createElement('img');
        image.src = URL.createObjectURL(file);


        fileItem.appendChild(image);
        fileItem.appendChild(deleteButton);

        fileList.appendChild(fileItem);

      }
      if (dropZone.contains(fileList)) {
    dropZone.removeChild(fileList);
  }

      if (droppedFiles.length > 0) {
          dropZone.appendChild(fileList);
      }
    }

    // Handle file selection from file input
    function handleFileSelection(event) {
      droppedFiles = Array.from(event.target.files);
      updateFileList()
      startOCRButton.style.display = 'flex';
      browseButton.style.display = 'none';
      paragraph.style.display = 'none';
    }

    function setBrowseButtonStyle(){
        browseButton.style.text_align = 'center';
        browseButton.style.background_position = 'center';
        browseButton.style.font_size = '32px';
    }

    // Handle file deletion
    function deleteFile(file) {
      const index = droppedFiles.indexOf(file);
      if (index !== -1) {
        droppedFiles.splice(index, 1);
        updateFileList();
      }
      if (droppedFiles.length === 0) {
        startOCRButton.style.display = 'none';
        browseButton.style.display = 'flex';
        setBrowseButtonStyle()
        paragraph.style.display = 'flex';
      }
      dropZone.appendChild(fileList);

    }

    // Send files to server for prediction
    async function makePrediction() {
      for (let i = 0; i < droppedFiles.length; i++) {
        const file = droppedFiles[i];
        const formData = new FormData();
        formData.append('file', file);

        try {
          const response = await fetch('OCR/predict', {
            method: 'POST',
            body: formData
          });

          if (response.ok) {
            const data = await response.json();
            const prediction = data.prediction;
            predictionResult.value = prediction;
          } else {
            console.error('Prediction failed');
          }
        } catch (error) {
          console.error('Error:', error);
        }
      }
    }

    // Add event listeners
    function addEventListeners() {
      fileInput.addEventListener('change', handleFileSelection);

      document.addEventListener('click', (event) => {
  if (event.target === dropZone || event.target === fileInput) {
    fileInput.click();
  }
      });


      dropZone.addEventListener('dragover', (event) => {
        event.preventDefault();
        dropZone.classList.add('dragover');
      });

      dropZone.addEventListener('dragleave', (event) => {
        event.preventDefault();
        dropZone.classList.remove('dragover');
      });

      dropZone.addEventListener('drop', (event) => {
        event.preventDefault();
        dropZone.classList.remove('dragover');
        droppedFiles = Array.from(event.dataTransfer.files);

        const existingImage = dropZone.querySelector('img');
        if (existingImage) {
          existingImage.remove();
        }

        updateFileList();
        browseButton.style.display = 'none';
        paragraph.style.display = 'none';
        startOCRButton.style.display = 'block';

      });

      startOCRButton.addEventListener('click', () => {
        makePrediction()
      });
    }

    // Initialize the page
    function initializePage() {
      setInitialDimensions();
      addEventListeners();
    }

    initializePage();